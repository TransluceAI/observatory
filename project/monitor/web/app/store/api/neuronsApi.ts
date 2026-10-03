import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { BASE_URL } from '@/app/constants';
import { Neuron, NeuronsMetadataDict } from '@/app/types/neuronData';
import { AttributionFilter, ComplexFilter, NeuronDBFilter, NeuronFilter } from '@/app/types/neuronFilters';
import { NeuronCluster } from '@/app/types/neuronData';

interface FetchNeuronsMetadataParams {
  sessionId: string;
  filter: NeuronFilter;
}

interface ClusterNeuronsParams {
  sessionId: string;
  filter: NeuronFilter;
  requestId: string;
}

interface ClusterNeuronsResponse {
  clusters: NeuronCluster[];
  n_failures: number;
}


interface ClusterNeuronsResult {
  clusters: NeuronCluster[];
  n_failures: number;
  requestId: string;
}

export const neuronsApi = createApi({
  reducerPath: 'neuronsApi',
  baseQuery: fetchBaseQuery({ baseUrl: BASE_URL }),
  endpoints: (builder) => ({
    fetchNeuronsMetadata: builder.query<
      NeuronsMetadataDict,
      FetchNeuronsMetadataParams
    >({
      query: (params: FetchNeuronsMetadataParams) => {
        return {
          url: `neurons/${params.sessionId}`,
          method: 'POST',
          body: params.filter,
        }
      },
      transformResponse: (response: { neurons_metadata_dict: NeuronsMetadataDict }) => response.neurons_metadata_dict,
    }),
    fetchNeuronsAndMetadata: builder.query<
      { neurons: Neuron[]; metadata: NeuronsMetadataDict },
      FetchNeuronsMetadataParams
    >({
      query: (params: FetchNeuronsMetadataParams) => {
        return {
          url: `neurons/${params.sessionId}`,
          method: 'POST',
          body: params.filter,
        }
      },
      transformResponse: (response: { neurons: Neuron[]; neurons_metadata_dict: NeuronsMetadataDict }) => {
        return {
          neurons: response.neurons,
          metadata: response.neurons_metadata_dict
        }
      },
    }),
    clusterNeurons: builder.query<ClusterNeuronsResult, ClusterNeuronsParams>({
      query: (params) => ({
        url: `neurons/cluster/${params.sessionId}`,
        method: 'POST',
        body: params.filter,
        timeout: 90000,
      }),
      transformResponse: (response: ClusterNeuronsResponse, meta, arg) => ({
        clusters: response.clusters,
        n_failures: response.n_failures,
        requestId: arg.requestId,
      }),
    }),
  }),
});

export const { useFetchNeuronsMetadataQuery, useLazyFetchNeuronsMetadataQuery, useClusterNeuronsQuery, useLazyClusterNeuronsQuery, useFetchNeuronsAndMetadataQuery, useLazyFetchNeuronsAndMetadataQuery } = neuronsApi;
