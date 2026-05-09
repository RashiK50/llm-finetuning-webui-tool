export const api = {
  status: async () => {
    return { status: 'idle', model: null, dataset: null, gpu: 'NVIDIA A100 80GB (Idle)' };
  },
  loadModel: async (modelId) => {
    return new Promise(resolve => setTimeout(() => resolve({ success: true, modelId }), 1500));
  },
  uploadDataset: async (file) => {
    return new Promise(resolve => setTimeout(() => resolve({ 
      success: true, 
      size: '2.4 MB',
      preview: [
        { instruction: "Translate to French: Hello", output: "Bonjour" },
        { instruction: "Summarize: The quick brown fox jumps over the lazy dog.", output: "A fox jumps over a dog." },
        { instruction: "Write a haiku about React.", output: "Components on screen\nState is changing everywhere\nRender is complete." }
      ]
    }), 1500));
  },
  train: async (config) => {
    return new Promise(resolve => setTimeout(() => resolve({ success: true }), 1000));
  },
  progress: async () => {
    return { loss: Math.max(0.5, Math.random() * 2), epoch: 1, step: Math.floor(Math.random() * 100) };
  },
  evaluate: async () => {
    return new Promise(resolve => setTimeout(() => resolve({
      rouge: 0.85, bert: 0.89, f1: 0.82,
      samples: [
        { target: "Bonjour", predicted: "Bonjour" },
        { target: "A fox jumps over a dog.", predicted: "The fox jumped." }
      ]
    }), 2000));
  },
  export: async (type) => {
    return new Promise(resolve => setTimeout(() => resolve({ success: true, url: '#' }), 2000));
  },
  chat: async (message) => {
    return new Promise(resolve => setTimeout(() => resolve({
      response: `This is a simulated model response to: "${message}"`
    }), 1500));
  }
};
