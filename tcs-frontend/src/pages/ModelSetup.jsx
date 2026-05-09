import React, { useState } from 'react';
import { useGlobalState } from '../context/GlobalState';
import StepFlow from '../components/ui/StepFlow';
import { api } from '../api';
import { Server, Save, AlertCircle, Loader2, CheckCircle2 } from 'lucide-react';
import clsx from 'clsx';

const predefinedModels = [
  "mistralai/Mistral-7B-v0.3",
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "meta-llama/Llama-2-7b-chat-hf",
  "Custom Model"
];

export default function ModelSetup() {
  const { model, setModel, status, setStatus } = useGlobalState();
  const [selectedModel, setSelectedModel] = useState(model || predefinedModels[0]);
  const [customModelPath, setCustomModelPath] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [toast, setToast] = useState(null);

  const handleLoad = async () => {
    const modelToLoad = selectedModel === "Custom Model" ? customModelPath : selectedModel;
    if (!modelToLoad) {
      setToast({ type: 'error', msg: 'Please select or enter a model name' });
      return;
    }
    setIsLoading(true);
    setStatus('loading_model');
    setToast(null);

    const res = await api.loadModel(modelToLoad);
    setIsLoading(false);

    if (res.success) {
      setModel(modelToLoad);
      setStatus('model_ready');
      setToast({ type: 'success', msg: 'Model loaded successfully!' });
      setTimeout(() => setToast(null), 3000);
    } else {
      setStatus('error');
      setToast({ type: 'error', msg: 'Failed to load model.' });
    }
  };

  return (
    <div className="max-w-4xl mx-auto flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <StepFlow />
      
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold text-textMain tracking-tight">Model Setup</h1>
        <p className="text-textMuted">Select a pre-trained base model or provide a path to a custom HuggingFace model.</p>
      </div>

      <div className="bg-surface border border-border/50 rounded-3xl p-8 shadow-2xl relative overflow-hidden group">
        <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl -mr-16 -mt-16 transition-all duration-700 group-hover:bg-primary/10 pointer-events-none"></div>
        
        <div className="relative z-10 flex flex-col gap-6">
          <div className="flex flex-col gap-2">
            <label className="text-sm font-semibold text-textMain flex items-center gap-2">
              <Server size={16} className="text-primary" />
              Base Model
            </label>
            <div className="relative">
              <select 
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-surfaceHover border border-border/50 text-textMain text-sm rounded-xl px-4 py-3.5 appearance-none focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all shadow-sm"
              >
                {predefinedModels.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
              <div className="absolute inset-y-0 right-4 flex items-center pointer-events-none text-textMuted">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
              </div>
            </div>
          </div>

          <div className={clsx("flex flex-col gap-2 transition-all duration-300", selectedModel === "Custom Model" ? "opacity-100 max-h-40" : "opacity-30 max-h-40 pointer-events-none")}>
            <label className="text-sm font-semibold text-textMain">HuggingFace Model Path</label>
            <input 
              type="text" 
              placeholder="e.g., openai-community/gpt2"
              value={selectedModel === "Custom Model" ? customModelPath : selectedModel}
              onChange={(e) => setCustomModelPath(e.target.value)}
              disabled={selectedModel !== "Custom Model"}
              className="w-full bg-background border border-border/50 text-textMain text-sm rounded-xl px-4 py-3.5 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary disabled:opacity-50 shadow-inner"
            />
          </div>

          <div className="pt-4 border-t border-border/30">
            <button 
              onClick={handleLoad}
              disabled={isLoading || status === 'training'}
              className="w-full sm:w-auto px-8 py-3.5 bg-primary hover:bg-primaryHover disabled:bg-surfaceHover disabled:text-textMuted text-white font-medium rounded-xl transition-all shadow-[0_4px_14px_0_rgba(59,130,246,0.39)] hover:shadow-[0_6px_20px_rgba(59,130,246,0.23)] hover:-translate-y-0.5 disabled:shadow-none disabled:transform-none flex items-center justify-center gap-2"
            >
              {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Save size={18} />}
              {isLoading ? 'Loading Model...' : 'Load Model'}
            </button>
          </div>
        </div>
      </div>

      {toast && (
        <div className={clsx(
          "flex items-center gap-3 p-4 rounded-xl border animate-in slide-in-from-bottom-8 fade-in shadow-lg",
          toast.type === 'error' ? "bg-error/10 border-error/20 text-error" : "bg-success/10 border-success/20 text-success"
        )}>
          {toast.type === 'error' ? <AlertCircle size={20} /> : <CheckCircle2 size={20} />}
          <span className="font-medium text-sm">{toast.msg}</span>
        </div>
      )}
    </div>
  );
}
