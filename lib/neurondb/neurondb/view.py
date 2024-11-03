from __future__ import annotations

from collections import defaultdict
from time import time
from typing import Any, cast

import numpy as np
import torch
from neurondb.filters import (
    QTILE_KEYS,
    QTILE_KEYS_TYPE,
    ActivationPercentileFilter,
    AttributionFilter,
    AttributionResult,
    ComplexFilter,
    IdFilter,
    Neuron,
    NeuronDBFilter,
    NeuronDescription,
    NeuronFilter,
    NeuronGeneralMetadata,
    NeuronPolarity,
    NeuronRunMetadata,
    NeuronsMetadataDict,
    TokenFilter,
)
from neurondb.postgres import DBManager
from neurondb.schemas import SQLANeuron, SQLANeuronDescription, SQLANeuronQuantiles
from util.chat_input import ChatConversation, ModelInput, make_chat_input
from util.subject import Subject
from util.types import NDFloatArray, NDIntArray

###############
# Neuron View #
###############


class NeuronView:
    """
    NeuronView represents a view of neurons in a neural network subject.
    It allows filtering and retrieval of specific neurons based on various criteria.

    The class maintains a set of all neurons in the subject and supports applying
    filters to this set. Filters can be simple (e.g., selecting specific neuron IDs)
    or complex (combining multiple filters with logical operations).
    """

    def __init__(
        self,
        subject: Subject,
        db: DBManager,
        model_input: ModelInput,
        percentiles_PLI: dict[str, NDFloatArray] | None = None,
    ):
        self._subject, self._db, self._model_input = (
            subject,
            db,
            model_input,
        )

        # State of the chat conversation
        self._acts_LIT: NDFloatArray | None = None
        self._neuron_interventions: dict[tuple[int, int, int], float] | None = None

        # State of the view
        self._neurons: set[Neuron] | None = set()  # None indicates catch-all
        self._filter: NeuronFilter | None = None

        # Cache for attribution effects
        self._attrs_LTfTsI: dict[tuple[int, int, int | None], list[AttributionResult]] = {}

        # Cache percentiles in memory (takes <10MB each)
        self._percentiles_PLI: dict[str, NDFloatArray] = percentiles_PLI or dict()
        missing_pctiles = cast(
            list[QTILE_KEYS_TYPE], [p for p in QTILE_KEYS if p not in self._percentiles_PLI]
        )
        if missing_pctiles:
            loaded_percentiles = self.load_percentiles(db, subject, missing_pctiles)
            self._percentiles_PLI.update(loaded_percentiles)

        # Immediately gather activations
        self._update_activations()

    @staticmethod
    def load_percentiles(
        db: DBManager, subject: Subject, percentiles: list[QTILE_KEYS_TYPE]
    ) -> dict[str, NDFloatArray]:
        print(f"Loading percentiles for {percentiles}...", end="", flush=True)
        pctiles = db.get(
            [
                SQLANeuron.layer,
                SQLANeuron.neuron,
                *(SQLANeuronQuantiles.quantiles[p] for p in percentiles),
            ],
            joins=[(SQLANeuron, SQLANeuron.id == SQLANeuronQuantiles.neuron_id)],
        )

        results = {p: np.full((subject.L, subject.I), np.nan) for p in percentiles}
        for row in pctiles:
            l, n, *values = row
            for p, v in zip(percentiles, values):
                results[p][l, n] = v

        print(" Done")

        return results

    @property
    def model_input(self) -> ModelInput:
        return self._model_input

    @property
    def filter(self):
        return self._filter

    def num_tokens(self) -> int:
        return len(self._model_input.tokenize(self._subject))

    def clear_neuron_interventions(self):
        self._neuron_interventions = None

    def set_neuron_interventions(self, interventions: dict[tuple[int, int, int], float]):
        self._neuron_interventions = interventions

    def send_message(self, *args: Any, **kwargs: Any):
        """
        Sends messages using the ChatConversation.
        Handles interventions and updates activations after each token.
        """

        if not isinstance(self._model_input, ChatConversation):
            raise ValueError(f"Cannot send message on {type(self._model_input)}")
        cc = self._model_input  # For some reason this is required for the type checker??

        # Include neuron interventions if applicable
        if self._neuron_interventions is not None:
            kwargs["neuron_interventions"] = self._neuron_interventions

        # Handle streaming
        if kwargs.get("stream", False):

            def _generator():
                for update in cc.send_message(*args, **kwargs):
                    yield update

            return _generator()
        else:
            ans = cc.send_message(*args, **kwargs)
            return ans

    def _update_activations(self):
        if self._model_input.is_empty(self._subject):
            return

        acts = self._subject.collect_acts(
            [self._model_input], layers=list(range(self._subject.L)), include=["neurons_BTI"]
        ).to("cpu")

        self._acts_LIT = np.zeros((self._subject.L, self._subject.I, self.num_tokens()))
        for layer in range(self._subject.L):
            # Get the first batch only
            self._acts_LIT[layer] = acts[layer].neurons_BTI[0].T.float().numpy()  # type: ignore

    def get_attribution(
        self,
        target_token_idx: int,
        target_token_id: int | None = None,
        distractor_token_id: int | None = None,
        top_k: int = 1000,
    ):
        if target_token_id is None:
            target_token_id = self._model_input.tokenize(self._subject)[target_token_idx + 1]

        if (
            target_token_idx,
            target_token_id,
            distractor_token_id,
        ) not in self._attrs_LTfTsI:
            PREFIX_LEN = (
                len(make_chat_input(None, "", "").tokenize(self._subject)) - 5
            )  # FIXME ?????

            acts: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            model = self._subject.model
            with model.trace(self._model_input.tokenize(self._subject)):  # type: ignore
                # Requires grad
                for layer in range(self._subject.L):
                    model.model.layers[layer].mlp.down_proj.input.requires_grad_(True)

                # Forward pass logits
                logits_V = model.lm_head.output[0, target_token_idx]
                # Compute probabilities
                probs_V = logits_V.softmax(dim=0)
                prob = probs_V[target_token_id]

                # Save things
                for layer in range(self._subject.L):
                    acts.append(
                        cast(torch.Tensor, model.model.layers[layer].mlp.down_proj.input.save())
                    )
                    grads.append(
                        cast(
                            torch.Tensor, model.model.layers[layer].mlp.down_proj.input.grad.save()
                        )
                    )

                # Backward pass
                prob.backward()

            all_activations_t, gradients_t = torch.concatenate(acts), torch.concatenate(grads)

            vals, indices = torch.topk(
                (all_activations_t[:, PREFIX_LEN:] * gradients_t[:, PREFIX_LEN:]).abs().flatten(),
                top_k,
            )
            indices_3N = cast(
                tuple[NDIntArray, ...],
                np.unravel_index(
                    indices.cpu().numpy(), all_activations_t[:, PREFIX_LEN:].shape  # type: ignore
                ),
            )
            indices_N3 = np.asarray(indices_3N).T + [0, PREFIX_LEN, 0]

            self._attrs_LTfTsI[(target_token_idx, target_token_id, distractor_token_id)] = [
                AttributionResult(
                    layer=l,
                    neuron=n,
                    src_token_idx=t,
                    tgt_token_idx=target_token_idx,
                    attribution=vals[i].item(),
                )
                for i, (l, t, n) in enumerate(indices_N3)
            ]

            del acts, grads, all_activations_t, gradients_t, vals, indices

        return self._attrs_LTfTsI[(target_token_idx, target_token_id, distractor_token_id)]

    def set_filter(self, filter: NeuronFilter | None):
        """
        Set and apply a new filter.
        You may presume that the neuron mask is always fresh, since it's updated when the filter is set.
        """

        self._filter = filter
        self._neurons = self._get_filtered_ids(None, self._filter)

    def get_neurons(self, with_tokens: bool = False) -> list[Neuron]:
        """
        TODO perform some sorting
        """

        neurons: set[Neuron] = set()
        if self._neurons is None:
            for l in range(self._subject.L):
                for n in range(self._subject.I):
                    for t in range(self.num_tokens()):
                        neurons.add(Neuron(layer=l, neuron=n, token=t))
        else:
            for n in self._neurons:
                if with_tokens:  # Tokens require activations
                    assert (
                        self._acts_LIT is not None
                    ), "Activations not computed; cannot get neurons using `with_tokens=True`"

                    if n.token is not None:  # If token specified, add as is
                        neurons.add(
                            n.model_validate(
                                n.model_dump()
                                | {"activation": self._acts_LIT[n.layer][n.neuron][n.token]}
                            )
                        )
                    else:  # If no token specified, add neurons at all tokens
                        for t in range(self.num_tokens()):
                            neurons.add(
                                n.model_validate(
                                    n.model_dump()
                                    | {
                                        "token": t,
                                        "activation": self._acts_LIT[n.layer][n.neuron][t],
                                    }
                                )
                            )
                else:
                    if n.token is not None:  # If token specified, remove specification
                        neurons.add(n.model_validate(n.model_dump() | {"token": None}))
                    else:  # If no token specified, add as is
                        neurons.add(n)

        return list(neurons)

    def get_neurons_metadata_dict(
        self, neurons: list[Neuron], include_run_metadata: bool = False
    ) -> NeuronsMetadataDict:
        return NeuronsMetadataDict(
            general=self._get_neurons_metadata_general(neurons),
            run=self._get_neurons_metadata_run(neurons) if include_run_metadata else None,
        )

    def _get_neurons_metadata_run(self, neurons: list[Neuron]):
        """
        Here's how we get run metadata:
        - All activations are returned for each (layer, neuron, token)
        - The current filter may have attribution filters. We want attribution metadata too.
            - Each attribution filter computes all neuron effects from preceding tokens to a target token.
            - We return the influence of each neuron from its firing token to all possible target token idxs.
        """

        assert self._acts_LIT is not None, "Activations not computed"

        # Collect attribution data as requested by the filter
        if self._filter is not None:
            attributed_tokens = set(
                f.target_token_idx for f in self._filter.get_attribution_filters()
            )
            attrs = [
                self.get_attribution(target_token_idx) for target_token_idx in attributed_tokens
            ]
            attrs_dict: dict[tuple[int, int, int], dict[int, AttributionResult]] = defaultdict(dict)
            for sublist in attrs:
                for a in sublist:
                    attrs_dict[(a.layer, a.neuron, a.src_token_idx)][a.tgt_token_idx] = a
        else:
            attrs_dict = {}

        metadata_run = [
            NeuronRunMetadata(
                layer=n.layer,
                neuron=n.neuron,
                token=n.token,
                activation=self._acts_LIT[n.layer][n.neuron][n.token],
                attributions=attrs_dict.get((n.layer, n.neuron, n.token)),
            )
            for n in neurons
            if n.token is not None  # This should always be true
        ]

        return {(mdr.layer, mdr.neuron, mdr.token): mdr for mdr in metadata_run}

    def _get_neurons_metadata_general(
        self, neurons: list[Neuron]
    ) -> dict[tuple[int, int], NeuronGeneralMetadata]:
        # Retrieve from DB
        unique_neuron_tuples = list(set((n.layer, n.neuron) for n in neurons))
        descs = self._db.get(
            [
                SQLANeuron.layer,
                SQLANeuron.neuron,
                SQLANeuronDescription.polarity,
                SQLANeuronDescription.description,
                SQLANeuronDescription.description_cleaned,
                SQLANeuronDescription.description_summary,
                SQLANeuronDescription.description_metadata["score"],
                SQLANeuronDescription.is_interesting,
            ],
            joins=[
                (SQLANeuron, SQLANeuron.id == SQLANeuronDescription.neuron_id),
                (
                    SQLANeuronQuantiles,
                    SQLANeuronQuantiles.neuron_id == SQLANeuron.id,
                ),
            ],
            # filter=SQLANeuronDescription.version.is_(None),
            # filter=SQLANeuronDescription.version == 'dami',
            layer_neuron_tuples=unique_neuron_tuples,
        )
        descs_dict = {
            (l, n, p): NeuronDescription(
                text=d,
                text_cleaned=dc,
                summary=su,
                score=sc,
                # is_interesting=i or True,
                is_interesting=i,
            )
            for l, n, p, d, dc, su, sc, i in descs
        }

        metadata_general = [
            NeuronGeneralMetadata(
                layer=n.layer,
                neuron=n.neuron,
                descriptions={
                    NeuronPolarity.POS: descs_dict.get((n.layer, n.neuron, 1)),
                    NeuronPolarity.NEG: descs_dict.get((n.layer, n.neuron, -1)),
                },
                activation_percentiles={
                    k: self._percentiles_PLI[k][n.layer, n.neuron] for k in QTILE_KEYS
                },
            )
            for n in neurons
        ]

        return {(mdg.layer, mdg.neuron): mdg for mdg in metadata_general}

    def _get_filtered_ids(
        self, neurons: set[Neuron] | None, filter: NeuronFilter | None, depth: int = 0
    ) -> set[Neuron] | None:
        """
        Recursively applies a filter to a list of neurons.
        If neurons is None, that means all neurons are in the set.
        """

        # Shortcut for no filter
        if filter is None:
            return neurons

        ans: set[Neuron] | None = None
        start_time = time()

        # These filters depend on current neurons
        if isinstance(filter, TokenFilter):
            if neurons is None:
                print(f"Warning: applying TokenFilter first is likely very slow")
                ans = set(
                    [
                        Neuron(layer=l, neuron=n, token=t)
                        for l in range(self._subject.L)
                        for n in range(self._subject.I)
                        for t in filter.tokens
                    ]
                )
            else:
                ans = set()
                for n in neurons:
                    if n.token is None:
                        for t in filter.tokens:
                            ans.add(Neuron(layer=n.layer, neuron=n.neuron, token=t))
                    else:
                        if n.token in filter.tokens:
                            ans.add(n)
        # These don't depend on `neurons`
        elif isinstance(filter, IdFilter):
            ans = filter.ids
        elif isinstance(filter, ActivationPercentileFilter):
            assert self._acts_LIT is not None, "Activations not computed"
            ans = filter.get_matching_ids(self._percentiles_PLI, self._acts_LIT)
        elif isinstance(filter, AttributionFilter):
            assert self._acts_LIT is not None, "Activations not computed"
            ans = filter.get_matching_ids(
                self.get_attribution, self._percentiles_PLI, self._acts_LIT
            )
        elif isinstance(filter, NeuronDBFilter):
            ans = filter.get_matching_ids(self._db)
        # Complex filters
        elif isinstance(filter, ComplexFilter):
            if filter.op == "and":
                ans = neurons
                for sub_filter in filter.filters:
                    sub_ids = self._get_filtered_ids(ans, sub_filter, depth + 1)
                    ans = self._and_ids(ans, sub_ids) if ans is not None else sub_ids
            elif filter.op == "or":
                ans = set()
                for sub_filter in filter.filters:
                    sub_ids = self._get_filtered_ids(ans, sub_filter, depth + 1)
                    ans = self._or_ids(ans, sub_ids)
            else:
                raise ValueError(f"Unknown filter op: {filter.op}")
        else:
            raise ValueError(f"Unknown filter type: {type(filter)}")

        elapsed = time() - start_time
        print(f"{' ' * depth * 4}Applying {filter.__class__.__name__} took {elapsed * 1000:.2f}ms")

        return ans

    @staticmethod
    def _and_ids(l1: set[Neuron] | None, l2: set[Neuron] | None):
        """
        AND over all neurons in l1 and l2 in O(N).
        This is complex since the neurons may have `None` tokens or `None` polarity, which are catch-alls.

        TODO FIX POLARITY ERRORS
        """

        # If one is None (catch-all), return the other
        if l1 is None:
            return l2
        if l2 is None:
            return l1

        # Group by (layer, neuron, polarity) and collect all tokens for each pair
        d1: dict[tuple[int, int, NeuronPolarity | None], set[int | None]] = defaultdict(set)
        d2: dict[tuple[int, int, NeuronPolarity | None], set[int | None]] = defaultdict(set)
        for n in l1:
            d1[(n.layer, n.neuron, n.polarity)].add(n.token)
            if n.token is None:
                assert (
                    len(d1[(n.layer, n.neuron, n.polarity)]) == 1
                ), f"Neuron {n} has multiple `None` tokens"
        for n in l2:
            d2[(n.layer, n.neuron, n.polarity)].add(n.token)
            if n.token is None:
                assert (
                    len(d2[(n.layer, n.neuron, n.polarity)]) == 1
                ), f"Neuron {n} has multiple `None` tokens"

        # Casework
        ans: set[Neuron] = set()
        for (l, n, p), ts1 in d1.items():
            ts2 = d2[l, n, p]

            # ts1 is a catch-all
            if None in ts1:
                # Both are catch-alls
                if None in ts2:
                    ans.add(_make_neuron(l, n, p, None))
                # ts2 is a bunch of tokens; take all of ts2
                else:
                    ans.update([_make_neuron(l, n, p, t) for t in ts2])
            # ts1 is a bunch of tokens
            else:
                # ts2 is a catch-all; take all of ts1
                if None in ts2:
                    ans.update([_make_neuron(l, n, p, t) for t in ts1])
                # ts2 is a bunch of tokens; take the ts1/ts2 intersection
                else:
                    ans.update([_make_neuron(l, n, p, t) for t in ts1.intersection(ts2)])

        return ans

    @staticmethod
    def _or_ids(l1: set[Neuron] | None, l2: set[Neuron] | None):
        """
        OR over all neurons in l1 and l2 in O(N).
        This is complex since the neurons may have `None` tokens, which is a catch-all.

        TODO FIX POLARITY ERRORS
        """

        # If either is None (catch-all), return None
        if l1 is None or l2 is None:
            return None
        # If either are empty, return the other
        if len(l1) == 0:
            return l2
        if len(l2) == 0:
            return l1

        # Group by (layer, neuron) and collect all tokens
        d1: dict[tuple[int, int, NeuronPolarity | None], set[int | None]] = defaultdict(set)
        d2: dict[tuple[int, int, NeuronPolarity | None], set[int | None]] = defaultdict(set)
        for n in l1:
            d1[(n.layer, n.neuron, n.polarity)].add(n.token)
            if n.token is None:
                assert (
                    len(d1[(n.layer, n.neuron, n.polarity)]) == 1
                ), f"Neuron {n} has multiple `None` tokens"
        for n in l2:
            d2[(n.layer, n.neuron, n.polarity)].add(n.token)
            if n.token is None:
                assert (
                    len(d2[(n.layer, n.neuron, n.polarity)]) == 1
                ), f"Neuron {n} has multiple `None` tokens"

        # Casework
        ans: set[Neuron] = set()
        for cur, other in [(d1, d2), (d2, d1)]:  # Scan in both directions
            for (l, n, p), ts1 in cur.items():
                ts2 = other[l, n, p]

                # ts1 is a catch-all; always take everything
                if None in ts1:
                    ans.add(_make_neuron(l, n, p, None))
                # ts1 is a bunch of tokens
                else:
                    # ts2 is a catch-all; take everything
                    if None in ts2:
                        ans.add(_make_neuron(l, n, p, None))
                    # ts2 is a bunch of tokens; take the union of ts1 and ts2
                    else:
                        ans.update([_make_neuron(l, n, p, t) for t in ts1.union(ts2)])

        return ans


def _make_neuron(l: int, n: int, p: NeuronPolarity | None, t: int | None):
    return Neuron(layer=l, neuron=n, token=t, polarity=p)
