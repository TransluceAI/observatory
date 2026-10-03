import { useLazyClusterNeuronsQuery } from '@/app/store/api/neuronsApi';
import {
  addLinterMessage,
  CustomLinterMessage,
  EndLinterMessage,
  IntroLinterMessage,
  NeuronClustersLinterMessage,
  resetAILinterState,
  setClusters,
  setLoadingTokenSelectionLinterMessageId,
  setSelectedClusterId,
  SolutionQuestionLinterMessage,
  SteeringHintLinterMessage,
  SwitchModeLinterMessage,
  TokenSelectionLinterMessage,
  updateLinterMessage,
} from '@/app/store/slices/aiLinterSlice';
import store, { RootState } from '@/app/store/store';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from '@/hooks/use-toast';
import {
  useState,
  useEffect,
  useRef,
  useMemo,
  useCallback,
  Suspense,
} from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  AlertCircle,
  BeakerIcon,
  CheckCircle,
  FlaskConicalIcon,
  HelpCircle,
  PlusCircle,
} from 'lucide-react';
import { PRESET_FLOWS, PresetFlow } from '@/app/types/presetFlow';
import {
  resetUIState,
  setFlowState,
  setShowChatArea,
  setShowLinterPanel,
  setShowNeuronsPanel,
  setShowSteeringPanel,
  setSteeringDialogSpec,
} from '@/app/store/slices/uiStateSlice';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  addSteeringSpec,
  resetSteeringState,
} from '@/app/store/slices/steeringSlice';
import { v4 as uuidv4 } from 'uuid';
import {
  resetNeuronsState,
  setMousedOverNeurons,
  setSelectedTokenRange,
  setTableHighlightedNeuronIds,
} from '@/app/store/slices/neuronsSlice';
import { NeuronRunMetadata } from '@/app/types/neuronData';
import { resetChatState } from '@/app/store/slices/chatSlice';
import { useSearchParams } from 'next/navigation'; // Add this import

import { usePostHog } from 'posthog-js/react';

interface DisplayedLinterMessage {
  type: 'assistant' | 'user';
  sticky?: boolean;
  content: JSX.Element;
}

function FlowIdHandler({ onFlowId }: { onFlowId: (flowId: string) => void }) {
  const searchParams = useSearchParams();

  useEffect(() => {
    const flowId = searchParams.get('flowId');
    if (flowId) {
      onFlowId(flowId);
    }
  }, [searchParams, onFlowId]);

  return null;
}

export default function LinterPanel() {
  const dispatch = useDispatch();
  const initialFlowHandled = useRef(false);

  const handleFlowIdChange = useCallback((flowId: string) => {
    if (!initialFlowHandled.current) {
      if (PRESET_FLOWS[flowId]) {
        initialFlowHandled.current = true;
        handlePresetFlowClick(flowId);
      } else {
        toast({
          title: 'Invalid preset flow',
          description: `Preset flow ${flowId} not found`,
        });
      }
    }
  }, []);

  /**
   * Global state
   */
  const sessionId = useSelector((state: RootState) => state.uiState.sessionId);
  const neuronsMetadataDict = useSelector(
    (state: RootState) => state.neurons.metadataDict
  );
  const globalNeuronFilter = useSelector(
    (state: RootState) => state.neurons.globalNeuronFilter
  );
  const steeringSpecs = useSelector(
    (state: RootState) => state.steering.steeringSpecs
  );
  const showNeuronsPanel = useSelector(
    (state: RootState) => state.uiState.showNeuronsPanel
  );
  const showSteeringPanel = useSelector(
    (state: RootState) => state.uiState.showSteeringPanel
  );
  const selectedTokenRange = useSelector(
    (state: RootState) => state.neurons.displayModulation.selectedTokenRange
  );
  const neuronClusters = useSelector(
    (state: RootState) => state.aiLinter.clusters
  );
  const selectedAttributionToken = useSelector(
    (state: RootState) =>
      state.neurons.displayModulation.selectedAttributionToken
  );

  const neuronIdsInTokenRange = useMemo(() => {
    if (!neuronsMetadataDict) {
      return new Set();
    }

    return new Set(
      Object.values(neuronsMetadataDict.run)
        .filter(
          (metadata) =>
            selectedTokenRange === undefined ||
            (selectedTokenRange[0] <= metadata.token &&
              metadata.token <= selectedTokenRange[1])
        )
        .map((metadata) => `${metadata.layer},${metadata.neuron}`)
    );
  }, [neuronsMetadataDict, selectedTokenRange]);

  /**
   * Preset flow
   */
  const flowState = useSelector((state: RootState) => state.uiState.flowState);
  const presetFlow = useMemo(() => {
    if (flowState && PRESET_FLOWS[flowState.presetFlowId]) {
      return PRESET_FLOWS[flowState.presetFlowId];
    }
    return undefined;
  }, [flowState]);
  const flowStateRef = useRef(flowState);
  useEffect(() => {
    flowStateRef.current = flowState;
  }, [flowState]);

  /**
   * Load the linter once neurons are populated
   */
  const [
    clusterNeurons,
    { data: clusteringResult, isLoading: loadingClusters },
  ] = useLazyClusterNeuronsQuery();
  const [clusterHintShown, setClusterHintShown] = useState(false);

  // Store the abortable promise returned by clusterNeurons
  const prevClusterNeuronsPromiseRef =
    useRef<ReturnType<typeof clusterNeurons>>();
  const curLinterMessageIdRef = useRef<string>();
  const [clusterRetryCount, setClusterRetryCount] = useState(0);
  // Store the requestId of the current clusterNeurons request; do not update clusters less it matches this
  const [curClusterRequestId, setCurClusterRequestId] = useState<
    string | undefined
  >(undefined);
  // Store the id of the linter message to update once clusters are returned
  const [curLinterMessageId, setCurLinterMessageId] = useState<
    string | undefined
  >(undefined);

  // Trigger a clustering request whenever the filter is changed
  useEffect(() => {
    if (
      sessionId &&
      neuronsMetadataDict &&
      globalNeuronFilter &&
      // In a preset flow, only scan for clusters if user has selected tokens at least once
      (flowState === undefined || flowState.tokensSelected)
    ) {
      // Abort the previous request if it exists
      let aborted = false;
      if (prevClusterNeuronsPromiseRef.current) {
        prevClusterNeuronsPromiseRef.current.abort();
        aborted = true;
      }

      const filters = [globalNeuronFilter];
      // if (selectedTokenRange) {
      //   filters.push({
      //     type: 'token',
      //     tokens: Array.from(
      //       { length: selectedTokenRange[1] - selectedTokenRange[0] + 1 },
      //       (_, i) => selectedTokenRange[0] + i
      //     ),
      //   });
      // }

      // Trigger the new request and store the promise & requestId
      const requestId = uuidv4();
      const promise = clusterNeurons({
        sessionId,
        filter: {
          type: 'complex',
          op: 'and',
          filters: filters,
        },
        requestId,
      });
      prevClusterNeuronsPromiseRef.current = promise;
      setCurClusterRequestId(requestId);

      promise.unwrap().catch((error) => {
        if (prevClusterNeuronsPromiseRef.current !== promise) return;
        prevClusterNeuronsPromiseRef.current = undefined;
        dispatch(setLoadingTokenSelectionLinterMessageId(undefined));
        const messageId = curLinterMessageIdRef.current;
        if (messageId) {
          dispatch(
            updateLinterMessage({
              id: messageId,
              message: {
                type: 'neuronClusters',
                id: messageId,
                error:
                  error?.status === 'TIMEOUT_ERROR'
                    ? 'Clustering took too long. Please try again.'
                    : 'Could not scan for clusters. Please try again.',
                showNeuronsFrom,
              } as NeuronClustersLinterMessage,
            })
          );
        }
      });

      // Send linter message that can be overwritten later
      if (!aborted) {
        const linterMessageId = uuidv4();
        curLinterMessageIdRef.current = linterMessageId;
        setCurLinterMessageId(linterMessageId);
        dispatch(
          addLinterMessage({
            type: 'neuronClusters',
            id: linterMessageId,
            clusters: undefined,
            showNeuronsFrom: showNeuronsFrom,
          } as NeuronClustersLinterMessage)
        );
      } else if (curLinterMessageIdRef.current) {
        dispatch(
          updateLinterMessage({
            id: curLinterMessageIdRef.current,
            message: {
              type: 'neuronClusters',
              id: curLinterMessageIdRef.current,
              showNeuronsFrom,
            } as NeuronClustersLinterMessage,
          })
        );
      }
    }
    // }, [neuronsMetadataDict, globalNeuronFilter, selectedTokenRange]);
  }, [neuronsMetadataDict, globalNeuronFilter, flowState?.tokensSelected, clusterRetryCount]);
  const showNeuronsFrom = useSelector(
    (state: RootState) => state.neurons.displayModulation.showNeuronsFrom
  );
  useEffect(() => {
    if (
      clusteringResult &&
      curLinterMessageId &&
      clusteringResult.requestId == curClusterRequestId
    ) {
      // Reset ref
      prevClusterNeuronsPromiseRef.current = undefined;

      // Indicate that the token selection linter message is no longer loading
      dispatch(setLoadingTokenSelectionLinterMessageId(undefined));

      // Update the clusters
      const clusters = clusteringResult.clusters;
      dispatch(setClusters(clusters));

      // Update the linter message
      const clonedClusters = JSON.parse(JSON.stringify(clusters));
      dispatch(
        updateLinterMessage({
          id: curLinterMessageId,
          message: {
            type: 'neuronClusters',
            id: curLinterMessageId,
            clusters: clonedClusters,
            showNeuronsFrom: showNeuronsFrom,
          } as NeuronClustersLinterMessage,
        })
      );

      // If in a preset flow, show a hint
      if (presetFlow && !clusterHintShown) {
        setTimeout(() => {
          dispatch(
            addLinterMessage({
              type: 'custom',
              role: 'assistant',
              message:
                'Mouse over a cluster to see which tokens its neurons fire on. Click any cluster to learn more.',
            } as CustomLinterMessage)
          );
        }, 1000);
        setClusterHintShown(true);
      }
    }
  }, [clusteringResult]);

  const chatTokens = useSelector((state: RootState) => state.chat.tokens);
  const isLoadingChat = useSelector(
    (state: RootState) => state.chat.isLoadingChat
  );
  const isStreamingTokens = useSelector(
    (state: RootState) => state.chat.isStreamingTokens
  );
  const selectedClusterId = useSelector(
    (state: RootState) => state.aiLinter.selectedClusterId
  );

  /**
   * Summed attribution data for sorting clusters
   */
  const summedAttributionDict = useMemo(() => {
    if (!neuronsMetadataDict) {
      return {};
    }

    const dict: Record<string, number> = {};
    for (const metadata of Object.values(neuronsMetadataDict.run)) {
      const attributions = metadata.attributions;
      const attribution = attributions
        ? Object.values(attributions)[0].attribution
        : 0;

      const key = `${metadata.layer},${metadata.neuron}`;
      dict[key] = (dict[key] ?? 0) + attribution;
    }
    return dict;
  }, [neuronsMetadataDict]);

  const posthog = usePostHog();

  /**
   * Handling clicks
   */
  const handlePresetFlowClick = (presetFlowId?: string) => {
    setPresetFlowClicked(true);
    posthog.capture('Started flow', {
      flowId: presetFlowId,
      sessionId: sessionId,
    });

    // If already in a preset flow, just navigate to the new one
    if (presetFlow) {
      if (presetFlowId) {
        window.location.href = `/dashboard/chat?flowId=${presetFlowId}`;
      } else {
        window.location.href = `/dashboard/chat`;
      }
      return;
    }

    // If no preset flow, show the chat area
    if (presetFlowId === undefined) {
      dispatch(setShowChatArea(true));
      return;
    }

    dispatch(
      addLinterMessage({
        type: 'custom',
        role: 'user',
        message: `Show me how to: ${PRESET_FLOWS[presetFlowId].title}.`,
      } as CustomLinterMessage)
    );

    setTimeout(() => {
      dispatch(
        addLinterMessage({
          type: 'custom',
          role: 'assistant',
          message: `Sure! Let me ask the model ${PRESET_FLOWS[presetFlowId].llmAsk}...`,
        } as CustomLinterMessage)
      );

      setTimeout(() => {
        dispatch(setFlowState({ presetFlowId: presetFlowId }));
      }, 500);
    }, 500);
  };

  /**
   * Preventing double clicks
   */
  const [presetFlowClicked, setPresetFlowClicked] = useState(false);
  const [clusterClicked, setClusterClicked] = useState(false);
  const [showMeClicked, setShowMeClicked] = useState(false);
  const [shouldShowEndMessage, setShouldShowEndMessage] = useState(false);
  const [endMessageShown, setEndMessageShown] = useState(false);

  /**
   * Trigger end if all steering specs are set
   */
  useEffect(() => {
    if (
      presetFlow &&
      steeringSpecs?.length === presetFlow.solutionSteeringSpecs.length &&
      presetFlow.solutionSteeringSpecs.every((solutionSpec) =>
        steeringSpecs.some((spec) => solutionSpec.id === spec.id)
      )
    ) {
      console.log('showing end message');
      console.log(presetFlow, endMessageShown);
      // Skip if the end message has already been shown
      if (!presetFlow || endMessageShown) {
        return;
      }
      setEndMessageShown(true);
      setShouldShowEndMessage(true);
    }
  }, [steeringSpecs]);

  useEffect(() => {
    if (!isStreamingTokens && shouldShowEndMessage && presetFlow) {
      // If all steering specs are set, show the end messages
      presetFlow.endMessages.forEach((endMessage, index) => {
        setTimeout(
          () => {
            dispatch(
              addLinterMessage({
                type: 'end',
                message: endMessage,
                nextFlowId:
                  index === presetFlow.endMessages.length - 1
                    ? presetFlow.nextFlowId
                    : null,
              } as EndLinterMessage)
            );
          },
          1000 * (index + 1)
        );
      });
      setShouldShowEndMessage(false);
    }
  }, [isStreamingTokens]);

  /**
   * Linter messages
   * Modules the chat messages shown in the panel
   */
  const linterMessages = useSelector(
    (state: RootState) => state.aiLinter.messages
  );
  const messages: DisplayedLinterMessage[] = useMemo(() => {
    return linterMessages
      .flatMap((message, messageIndex) => {
        if (message.type === 'intro') {
          const curMessage = message as IntroLinterMessage;
          const ans = [
            {
              type: 'assistant',
              content: (
                <>Hello! 🕵️ I&apos;m Transluce&apos;s model investigator.</>
              ),
            } as DisplayedLinterMessage,
            {
              type: 'assistant',
              content: (
                <>
                  I can help you understand and debug language model behaviors.
                </>
              ),
            } as DisplayedLinterMessage,
            {
              type: 'assistant',
              content: (
                <>
                  Let me walk you through a tutorial of my interface. Together,
                  we&apos;ll:
                  <ol className="list-decimal list-inside pl-4">
                    <li>
                      <b>Observe</b> a peculiar model behavior
                    </li>
                    <li>
                      <b>Understand</b> why it occurred
                    </li>
                    <li>
                      <b>Fix</b> the issue
                    </li>
                  </ol>
                </>
              ),
            } as DisplayedLinterMessage,
          ];

          if (true) {
            ans.push({
              type: 'assistant',
              content: (
                <div className="space-y-1">
                  <div>Click on a demo to get started!</div>
                  <div className="grid xl:grid-cols-2 grid-cols-1 gap-2 w-full">
                    {curMessage.presetFlowIds.map((presetFlowId) => (
                      <Button
                        key={presetFlowId}
                        size="sm"
                        variant="default"
                        className="w-full text-left"
                        disabled={
                          isLoadingChat || presetFlowClicked || !sessionId
                        }
                        onClick={() => handlePresetFlowClick(presetFlowId)}
                      >
                        {PRESET_FLOWS[presetFlowId].title}
                      </Button>
                    ))}
                    <Button
                      size="sm"
                      variant="secondary"
                      className="w-full text-left"
                      disabled={
                        isLoadingChat || presetFlowClicked || !sessionId
                      }
                      onClick={() => handlePresetFlowClick(undefined)}
                    >
                      Try my own prompt!
                    </Button>
                  </div>
                </div>
              ),
            } as DisplayedLinterMessage);
          }

          return ans;
        } else if (message.type === 'neuronClusters') {
          // // Find the last index of 'neuronClusters' message
          // const lastNeuronClustersIndex = linterMessages.findLastIndex(
          //   (msg) => msg.type === 'neuronClusters'
          // );
          // // Only process if this is the last 'neuronClusters' message
          // if (index !== lastNeuronClustersIndex) {
          //   return [];
          // }

          const curMessage = message as NeuronClustersLinterMessage;

          if (curMessage.error) {
            return {
              type: 'assistant',
              content: (
                <>
                  <div>{curMessage.error}</div>
                  <Button size="sm" variant="secondary" onClick={() => setClusterRetryCount((count) => count + 1)}>
                    Retry scan
                  </Button>
                </>
              ),
            } as DisplayedLinterMessage;
          } else if (curMessage.clusters === undefined) {
            return {
              type: 'assistant',
              content: (
                <>
                  <div>Scanning for clusters...</div>
                  <div className="flex space-x-3 overflow-x-scroll p-1">
                    {Array(3)
                      .fill(null)
                      .map((_, index) => (
                        <SkeletonCard key={index} />
                      ))}
                  </div>
                </>
              ),
            } as DisplayedLinterMessage;
          } else if (curMessage.clusters.length === 0) {
            return {
              type: 'assistant',
              content: <>No clusters found.</>,
            } as DisplayedLinterMessage;
          } else {
            const isLastNeuronClustersMessage =
              linterMessages.findLastIndex(
                (msg) => msg.type === 'neuronClusters'
              ) === messageIndex;

            return {
              type: 'assistant',
              content: (
                <>
                  <div>
                    I found some neurons that
                    {curMessage.showNeuronsFrom === 'attribution'
                      ? ` highly influenced your selection.`
                      : ' fired highest on the input.'}{' '}
                    I grouped their behaviors into these clusters:
                  </div>
                  <div className="flex space-x-3 overflow-x-scroll p-1">
                    {[...curMessage.clusters]
                      .sort((a, b) => {
                        const aSum = a.neurons.reduce(
                          (acc, neuron) =>
                            acc +
                            (summedAttributionDict[
                              `${neuron.layer},${neuron.neuron}`
                            ] ?? 0),
                          0
                        );
                        const bSum = b.neurons.reduce(
                          (acc, neuron) =>
                            acc +
                            (summedAttributionDict[
                              `${neuron.layer},${neuron.neuron}`
                            ] ?? 0),
                          0
                        );
                        return bSum - aSum;
                      })
                      .map((cluster, index) => {
                        const neuronsInTokenRange = cluster.neurons.filter(
                          (neuron) =>
                            neuronIdsInTokenRange.has(
                              `${neuron.layer},${neuron.neuron}`
                            )
                        );
                        return (
                          <div
                            key={cluster.id}
                            onClick={() => {
                              posthog.capture('Clicked cluster', {
                                clusterId: cluster.id,
                                clusterDescription: cluster.description,
                                sessionId: sessionId,
                              });

                              // Queue a hint about the clusters
                              if (!clusterClicked && presetFlow) {
                                setClusterClicked(true);
                                dispatch(
                                  addLinterMessage({
                                    type: 'custom',
                                    role: 'assistant',
                                    message: `I've highlighted neurons in your cluster in green. Click on a row to learn about the individual neuron.`,
                                  } as CustomLinterMessage)
                                );

                                setTimeout(() => {
                                  dispatch(setShowNeuronsPanel(true));

                                  setTimeout(() => {
                                    dispatch(
                                      addLinterMessage({
                                        type: 'solutionQuestion',
                                        question: presetFlow.solutionQuestion,
                                      } as SolutionQuestionLinterMessage)
                                    );
                                  }, 5000);
                                }, 1000);
                              }

                              // Reset token selection if no selected token has neurons in this cluster
                              if (!neuronsInTokenRange.length) {
                                dispatch(setSelectedTokenRange(undefined));
                              }

                              // Actually select the neurons
                              dispatch(setSelectedClusterId(cluster.id));
                              dispatch(
                                setTableHighlightedNeuronIds(
                                  cluster.neurons.map((neuron) => {
                                    return `${neuron.layer},${neuron.neuron}`;
                                  })
                                )
                              );
                            }}
                            onMouseOver={() => {
                              if (!isLastNeuronClustersMessage) return;
                              dispatch(setMousedOverNeurons(cluster.neurons));
                            }}
                            onMouseOut={() => {
                              if (!isLastNeuronClustersMessage) return;
                              dispatch(setMousedOverNeurons(undefined));
                            }}
                            className={cn(
                              'rounded-md flex-shrink-0 w-52 bg-white shadow-md transition-shadow duration-200 p-2',
                              isLastNeuronClustersMessage
                                ? 'hover:shadow-lg cursor-pointer hover:bg-[#d6ebe3A0]'
                                : 'opacity-50 cursor-not-allowed',
                              selectedClusterId === cluster.id &&
                                isLastNeuronClustersMessage &&
                                'ring-2 ring-[#67b398ff] bg-[#d6ebe3ff]'
                            )}
                          >
                            <h3 className="text-sm font-semibold leading-tight mb-2 text-gray-800 line-clamp-3">
                              {cluster.description}
                            </h3>
                            <p className="text-xs text-gray-600">{`${neuronsInTokenRange.length} neurons${
                              selectedTokenRange
                                ? ' in token range'
                                : ' matching'
                            }`}</p>
                          </div>
                        );
                      })}
                  </div>
                </>
              ),
            } as DisplayedLinterMessage;
          }
        } else if (message.type === 'tokenSelection') {
          const curMessage = message as TokenSelectionLinterMessage;
          if (curMessage.tokenIdx === null) {
            return {
              type: 'user',
              content: <>Clear token selection.</>,
            } as DisplayedLinterMessage;
          } else {
            return {
              type: 'user',
              content: (
                <>
                  {curMessage.mode === 'attribution'
                    ? 'Find neurons influencing the prediction of'
                    : 'Find neurons activating highly on'}{' '}
                  {curMessage.tokenIdx !== undefined &&
                  curMessage.tokenString ? (
                    <b>
                      &quot;{curMessage.tokenString}&quot; (index{' '}
                      {curMessage.tokenIdx})
                    </b>
                  ) : (
                    'the prompt'
                  )}
                  .
                </>
              ),
            } as DisplayedLinterMessage;
          }
        } else if (message.type === 'switchMode') {
          const curMessage = message as SwitchModeLinterMessage;
          return [
            {
              type: 'user',
              content: (
                <>
                  {curMessage.mode === 'attribution'
                    ? 'Switch to attribution mode. '
                    : 'Switch to activation mode. '}
                </>
              ),
            } as DisplayedLinterMessage,
            {
              type: 'assistant',
              content: (
                <>
                  {curMessage.mode === 'attribution' ? (
                    <>
                      Now in attribution mode! Click a token to see which
                      neurons are <b>influencing its prediction</b>.
                    </>
                  ) : (
                    <>
                      Now in activation mode! Click a token to see which neurons{' '}
                      <b>activate highly on it</b>.
                    </>
                  )}
                </>
              ),
            } as DisplayedLinterMessage,
          ];
        } else if (message.type === 'custom') {
          const curMessage = message as CustomLinterMessage;
          return {
            type: curMessage.role,
            content: <div>{curMessage.message}</div>,
          } as DisplayedLinterMessage;
        } else if (message.type === 'steeringHint') {
          const curMessage = message as SteeringHintLinterMessage;
          return {
            type: 'assistant',
            sticky: !curMessage.steeringSpecs.every((spec) =>
              steeringSpecs?.some((s) => s.id === spec.id)
            ),
            content: (
              <>
                <div>{curMessage.steeringNarration}</div>
                <div className="flex flex-wrap space-x-2 mt-1 items-center">
                  {curMessage.steeringSpecs.map((spec) => (
                    <>
                      <Button
                        key={spec.id}
                        size="sm"
                        onClick={() => {
                          posthog.capture('Opened steering dialog', {
                            specId: spec.id,
                            specName: spec.name,
                            sessionId: sessionId,
                          });
                          dispatch(setSteeringDialogSpec(spec));
                        }}
                        disabled={
                          isLoadingChat ||
                          steeringSpecs?.some((s) => s.id === spec.id)
                        }
                      >
                        <div>
                          {spec.strength === 0 ? 'Suppress' : 'Strengthen'}{' '}
                          {spec.name}
                        </div>
                      </Button>
                      {steeringSpecs?.some((s) => s.id === spec.id) ? (
                        <CheckCircle size={20} className="text-green-500" />
                      ) : (
                        <AlertCircle size={20} className="text-red-500" />
                      )}
                    </>
                  ))}
                </div>
              </>
            ),
          } as DisplayedLinterMessage;
        } else if (message.type === 'solutionQuestion') {
          const curMessage = message as SolutionQuestionLinterMessage;
          return {
            type: 'assistant',
            sticky: !showMeClicked,
            content: (
              <div className="space-y-1">
                <div>{curMessage.question}</div>
                <Button
                  onClick={() => {
                    if (presetFlow && !showMeClicked) {
                      posthog.capture('Clicked Show Me', {
                        flow: presetFlow.title,
                        sessionId: sessionId,
                      });
                      setShowMeClicked(true);
                      dispatch(
                        addLinterMessage({
                          type: 'custom',
                          role: 'user',
                          message: 'Show me!',
                        } as CustomLinterMessage)
                      );

                      setTimeout(() => {
                        dispatch(
                          addLinterMessage({
                            type: 'custom',
                            role: 'assistant',
                            message: presetFlow.solutionAnswer,
                          } as CustomLinterMessage)
                        );

                        setTimeout(() => {
                          dispatch(
                            addLinterMessage({
                              type: 'steeringHint',
                              steeringNarration: presetFlow.steeringNarration,
                              steeringSpecs: presetFlow.solutionSteeringSpecs,
                            } as SteeringHintLinterMessage)
                          );
                          dispatch(setShowSteeringPanel(true));
                        }, 1000);
                      }, 1000);
                    }
                  }}
                  size="sm"
                  disabled={showMeClicked}
                >
                  Show me!
                </Button>
              </div>
            ),
          } as DisplayedLinterMessage;
        } else if (message.type === 'end') {
          const curMessage = message as EndLinterMessage;
          return {
            type: 'assistant',
            sticky: true,
            content: (
              <div className="space-y-1">
                <div>{curMessage.message}</div>
                {curMessage.nextFlowId && (
                  <Button
                    size="sm"
                    onClick={() => {
                      window.location.href = `/dashboard/chat?flowId=${curMessage.nextFlowId}`;
                    }}
                  >
                    {PRESET_FLOWS[curMessage.nextFlowId].title}
                  </Button>
                )}
                {curMessage.nextFlowId === undefined && (
                  <Button
                    size="sm"
                    onClick={() => {
                      window.location.href = `/dashboard/chat`;
                    }}
                  >
                    Go home
                  </Button>
                )}
              </div>
            ),
          } as DisplayedLinterMessage;
        }
      })
      .filter((message) => message !== undefined)
      .sort((a, b) => {
        if (a.sticky && !b.sticky) return 1;
        if (!a.sticky && b.sticky) return -1;
        return 0;
      });
  }, [
    linterMessages,
    chatTokens,
    selectedClusterId,
    isLoadingChat,
    dispatch,
    showNeuronsPanel,
    showSteeringPanel,
    presetFlow,
    showMeClicked,
    neuronIdsInTokenRange,
    sessionId,
  ]);

  /**
   * Deselect the cluster if the highlighted neurons are not in the current token range
   */
  const tableHighlightedNeuronIds = useSelector(
    (state: RootState) =>
      state.neurons.displayModulation.tableHighlightedNeuronIds
  );
  useEffect(() => {
    if (
      tableHighlightedNeuronIds &&
      !tableHighlightedNeuronIds.some((neuron) =>
        neuronIdsInTokenRange.has(neuron)
      )
    ) {
      dispatch(setSelectedClusterId(undefined));
      dispatch(setTableHighlightedNeuronIds(undefined));
    }
  }, [neuronIdsInTokenRange]);

  /**
   * Auto-sticking messages to top or bottom
   */
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [hasOverflow, setHasOverflow] = useState(false);
  const checkOverflow = useCallback(() => {
    if (messagesContainerRef.current) {
      const { scrollHeight, clientHeight } = messagesContainerRef.current;
      setHasOverflow(scrollHeight > clientHeight);
    }
  }, []);
  useEffect(() => {
    const resizeObserver = new ResizeObserver(checkOverflow);
    if (messagesContainerRef.current) {
      resizeObserver.observe(messagesContainerRef.current);
    }

    return () => {
      resizeObserver.disconnect();
    };
  }, [checkOverflow]);
  useEffect(() => {
    checkOverflow();
  }, [messages, showNeuronsPanel, checkOverflow]);
  const reverseMessages = useMemo(
    () => showNeuronsPanel && hasOverflow,
    [showNeuronsPanel, hasOverflow]
  );

  return (
    <div className="flex flex-col h-full w-full">
      <Suspense>
        <FlowIdHandler onFlowId={handleFlowIdChange} />
      </Suspense>
      <div className="flex items-center space-x-2 mb-2">
        <div className="text-lg font-bold">Transluce Model Investigator</div>
        {/*
          <Tooltip>
            <TooltipTrigger>
              <HelpCircle
                size={14}
                className="text-muted-foreground cursor-help"
              />
            </TooltipTrigger>
            <TooltipContent className="text-sm max-w-[30rem]">
              <ul className="list-disc pl-4">
                <li>Description coming soon!</li>
              </ul>
            </TooltipContent>
          </Tooltip>
         */}
      </div>
      <div className="flex flex-1 flex-col overflow-y-auto p-1.5 bg-gradient-to-r from-blue-100 to-purple-100 font-mono text-[0.8rem] rounded-lg">
        <div
          ref={messagesContainerRef}
          className={cn(
            'flex flex-1 text-left overflow-y-auto w-full',
            reverseMessages ? 'flex-col-reverse' : 'flex-col'
          )}
        >
          {(reverseMessages ? [...messages].reverse() : messages).map(
            (message, index) => (
              <div
                key={index}
                className={cn(
                  'flex',
                  message.type === 'assistant' ? 'justify-start' : 'justify-end'
                )}
              >
                <div
                  className={cn(
                    'rounded-lg p-1.5 py-1.5 max-w-[80%]',
                    message.type === 'assistant'
                      ? 'bg-white'
                      : 'bg-[#dae4fbff]',
                    index !== (reverseMessages ? messages.length - 1 : 0) &&
                      'mt-1.5'
                  )}
                >
                  {message.content}
                </div>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="rounded-md flex-shrink-0 w-52 bg-white shadow-md p-2">
      <Skeleton className="h-4 w-24 mb-2" />
      <Skeleton className="h-3 w-32 mb-1" />
      <Skeleton className="h-3 w-28 mb-1" />
      <Skeleton className="h-3 w-24" />
    </div>
  );
}
