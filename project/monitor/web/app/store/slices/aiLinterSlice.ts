import { NeuronCluster } from '@/app/types/neuronData';
import { SteeringSpec } from '@/app/types/neuronFilters';
import { PRESET_FLOWS, PresetFlow } from '@/app/types/presetFlow';
import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface LinterMessage {
  type: 'intro' | 'solutionQuestion' | 'neuronClusters' | 'tokenSelection' | 'custom' | "switchMode" | 'steeringHint' | 'end';
  id?: string;
}

export interface EndLinterMessage extends LinterMessage {
  type: 'end';
  message: string;
  nextFlowId?: string | null;
}

export interface SolutionQuestionLinterMessage extends LinterMessage {
  type: 'solutionQuestion';
  question: string;
}

export interface IntroLinterMessage extends LinterMessage {
  type: 'intro';
  presetFlowIds: string[];
}

export interface SteeringHintLinterMessage extends LinterMessage {
  type: 'steeringHint';
  steeringNarration: string;
  steeringSpecs: SteeringSpec[];
}

export interface SwitchModeLinterMessage extends LinterMessage {
  type: 'switchMode';
  mode: "attribution" | "activation";
}

export interface CustomLinterMessage extends LinterMessage {
  type: 'custom';
  role: 'assistant' | 'user'
  message: string;
}

export interface TokenSelectionLinterMessage extends LinterMessage {
  type: 'tokenSelection';
  tokenIdx?: number;
  tokenString?: string;
  mode: "attribution" | "activation";
}

export interface NeuronClustersLinterMessage extends LinterMessage {
  type: 'neuronClusters';
  clusters?: NeuronCluster[];
  error?: string;
  showNeuronsFrom: 'attribution' | 'activation';
}

interface AILinterState {
  selectedClusterId?: string;
  clusters?: NeuronCluster[];
  loadingTokenSelectionLinterMessageId?: string;
  messages: LinterMessage[];
}

const initialState: AILinterState = {
  selectedClusterId: undefined,
  clusters: undefined,
  loadingTokenSelectionLinterMessageId: undefined,
  messages: [
    {
      type: 'intro',
      // presetFlowIds: ['comparison_911', 'carlini_elk', 'alice_bob_steering'],
      presetFlowIds: ['comparison_911', 'sort_numbers', 'carlini_elk', 'alice_bob_steering'],
      // presetFlowIds: ['comparison_911', 'alice_bob_steering'],
    } as IntroLinterMessage,
  ],
};

const aiLinterSlice = createSlice({
  name: 'aiLinter',
  initialState,
  reducers: {
    resetAILinterState(state) {
      Object.assign(state, initialState);
    },
    setSelectedClusterId(state, action: PayloadAction<string | undefined>) {
      state.selectedClusterId = action.payload;
    },
    setClusters(state, action: PayloadAction<NeuronCluster[] | undefined>) {
      state.clusters = action.payload;
    },
    setLoadingTokenSelectionLinterMessageId(state, action: PayloadAction<string | undefined>) {
      state.loadingTokenSelectionLinterMessageId = action.payload;
    },
    clearLinterMessages(state) {
      state.messages = [];
    },
    addLinterMessage(state, action: PayloadAction<LinterMessage>) {
      state.messages.push(action.payload);
    },
    updateLinterMessage(state, action: PayloadAction<{ id: string, message: LinterMessage }>) {
      const idx = state.messages.findIndex(m => m.id === action.payload.id);
      if (idx !== -1) {
        state.messages[idx] = action.payload.message;
      }
    },
  },
});

export const { resetAILinterState, setSelectedClusterId, setClusters, setLoadingTokenSelectionLinterMessageId, clearLinterMessages, addLinterMessage, updateLinterMessage } =
  aiLinterSlice.actions;
export default aiLinterSlice.reducer;
