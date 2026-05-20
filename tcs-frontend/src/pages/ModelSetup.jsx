import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGlobalState } from '../context/GlobalState';
import StepFlow from '../components/ui/StepFlow';
import { api } from '../api';
import {
  Save,
  Edit2,
  XCircle,
  Loader2,
  Box,
  Layers,
  Cpu,
  CheckCircle2,
  ArrowRight,
  Sparkles,
  HardDrive,
  Database,
} from 'lucide-react';
import toast from 'react-hot-toast';

const predefinedModels = [
  'meta-llama/Meta-Llama-3-8B',
  'mistralai/Mistral-7B-v0.3',
  'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
  'Qwen/Qwen2.5-0.5B-Instruct',
];

export default function ModelSetup() {
  const navigate = useNavigate();
  const {
    model,
    setModel,
    status,
    setStatus,
    modelConfig,
    setModelConfig,
    loadingProgress,
    loadingMessage,
    gpuName,
    gpuMem,
    gpuTotalMem,
    ram,
    ramTotal,
    currentStep,
  } = useGlobalState();

  const initialBaseModel = useMemo(() => {
    if (modelConfig?.modelName === 'custom') return 'custom';
    if (predefinedModels.includes(modelConfig?.modelName)) return modelConfig.modelName;
    if (predefinedModels.includes(model)) return model;
    return predefinedModels[0];
  }, [model, modelConfig]);

  const [modelType, setModelType] = useState(initialBaseModel);
  const [customPath, setCustomPath] = useState(modelConfig?.customPath || '');
  const [isLoading, setIsLoading] = useState(false);
  const [isWaitingForModel, setIsWaitingForModel] = useState(false);
  const [reloadMode, setReloadMode] = useState(false);
  const [reloadLoraRepo, setReloadLoraRepo] = useState('');

  const isCustomModel = modelType === 'custom';
  const hfModelPath = isCustomModel ? customPath : modelType;
  const isLoaded = !!model && status !== 'loading_model';
  const isBusy = isLoading || status === 'loading_model';
  const gpuPercent = gpuTotalMem > 0 ? Math.min(100, Math.round((gpuMem / gpuTotalMem) * 100)) : 0;
  const ramPercent = ramTotal > 0 ? Math.min(100, Math.round((ram / ramTotal) * 100)) : 0;

  useEffect(() => {
    if (modelConfig?.modelName === 'custom') {
      setModelType('custom');
      setCustomPath(modelConfig?.customPath || '');
      return;
    }
    if (predefinedModels.includes(modelConfig?.modelName)) {
      setModelType(modelConfig.modelName);
      setCustomPath(modelConfig.modelName);
    }
  }, [modelConfig]);

  useEffect(() => {
    if (!isWaitingForModel) return;
    if (status === 'model_ready') {
      setIsWaitingForModel(false);
      setIsLoading(false);
      toast.success('Model loaded successfully.');
    } else if (status === 'error') {
      setIsWaitingForModel(false);
      setIsLoading(false);
      toast.error('Failed to load model.');
    }
  }, [status, isWaitingForModel]);

  const handleBaseModelChange = (value) => {
    setModelType(value);
    if (value === 'custom') {
      setModelConfig((prev) => ({ ...prev, modelName: 'custom', customPath: prev?.customPath || '' }));
      return;
    }
    setCustomPath(value);
    setModelConfig({ modelName: value, customPath: value });
  };

  const handleCustomPathChange = (value) => {
    setCustomPath(value);
    setModelConfig((prev) => ({ ...prev, modelName: 'custom', customPath: value }));
  };

  const handleLoad = async () => {
    const finalModel = hfModelPath.trim();
    if (!finalModel) {
      toast.error('Enter a valid Hugging Face model path.');
      return;
    }

    setIsLoading(true);
    setIsWaitingForModel(true);
    try {
      let res;
      if (reloadMode) {
        if (!reloadLoraRepo.trim()) {
          toast.error('Enter the HF repo ID that holds your LoRA adapters.');
          setIsLoading(false);
          setIsWaitingForModel(false);
          return;
        }
        res = await api.loadFinetunedModel(finalModel, reloadLoraRepo.trim());
      } else {
        res = await api.loadModel(finalModel);
      }
      if (res?.success) {
        setStatus('loading_model');
        setModelConfig({
          modelName: isCustomModel ? 'custom' : modelType,
          customPath: finalModel,
        });
      }
    } catch {
      setIsLoading(false);
      setIsWaitingForModel(false);
    }
  };

  const handleCancelLoad = async () => {
    try {
      await api.cancelOperation('loading_model');
      setIsLoading(false);
      setIsWaitingForModel(false);
      setStatus('idle');
      toast.success('Cancel requested.');
    } catch {
      // API layer toast handles failure.
    }
  };

  const handleEdit = async () => {
    const loadingToastId = toast.loading('Unloading model from memory...');
    try {
      await api.unloadModel();
      setModel(null);
      setStatus('idle');
      toast.dismiss(loadingToastId);
      toast.success('Model unloaded. Configuration unlocked.');
    } catch {
      toast.dismiss(loadingToastId);
      toast.error('Failed to unload model.');
    }
  };

  return (
    <div className="max-w-[1600px] mx-auto min-h-0 flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-4 duration-500 px-2">
      <StepFlow />

      <div className="flex items-center justify-between gap-3 shrink-0">
        <div className="flex items-center gap-3 min-w-0">
          <h1 className="text-2xl md:text-3xl font-extrabold text-textMain tracking-tight">Model Setup</h1>
        </div>
        {!isLoaded && (
          <div className="hidden md:block px-3 py-2 rounded-xl border border-border/50 text-textMuted bg-surface/60 text-xs">
            Configure and load a model first
          </div>
        )}
      </div>

      <p className="text-sm text-textMuted shrink-0">
        Hugging Face path mirrors your base-model selection and is locked. Switch to custom to edit it manually.
      </p>

      <div className="grid grid-cols-1 xl:grid-cols-[470px_minmax(0,1fr)] gap-4 flex-1 min-h-0">
        <div className="bg-surface border border-border/50 rounded-3xl p-5 shadow-xl flex flex-col gap-4 min-h-0">
          <div className="flex flex-col gap-2">
            <label className="text-xs uppercase tracking-[0.22em] font-bold text-textMuted flex items-center gap-2">
              <Box size={13} className="text-primary" /> Base Model
            </label>
            <select
              value={modelType}
              onChange={(e) => handleBaseModelChange(e.target.value)}
              disabled={isBusy}
              className="w-full bg-background border border-border/50 text-textMain text-sm rounded-2xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary/50 appearance-none disabled:opacity-50"
            >
              {predefinedModels.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
              <option value="custom">Custom / Manual Path</option>
            </select>
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-xs uppercase tracking-[0.22em] font-bold text-textMuted">
              Hugging Face Model Path
            </label>
            <input
              type="text"
              value={hfModelPath}
              placeholder={isCustomModel ? 'org-name/model-name or /local/model/path' : modelType}
              onChange={(e) => isCustomModel && handleCustomPathChange(e.target.value)}
              disabled={!isCustomModel || isBusy}
              className="w-full bg-background border border-border/50 text-textMain rounded-2xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-55"
            />
            <p className="text-xs text-textMuted">
              {isCustomModel
                ? 'Editable: enter any valid repo ID or local model directory.'
                : 'Locked to base model selection.'}
            </p>
          </div>

          {/* ── Reload from HF (fine-tuned) toggle ── */}
          <div className="rounded-2xl border border-border/40 bg-background/60 p-3 flex flex-col gap-2">
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={reloadMode}
                onChange={(e) => setReloadMode(e.target.checked)}
                disabled={isBusy}
                className="accent-primary w-4 h-4"
              />
              <span className="text-xs font-semibold text-textMain">
                Reload from HF (attach existing LoRA adapters)
              </span>
            </label>
            {reloadMode && (
              <div className="flex flex-col gap-1">
                <input
                  type="text"
                  value={reloadLoraRepo}
                  onChange={(e) => setReloadLoraRepo(e.target.value)}
                  placeholder="your-username/my-finetuned-lora"
                  disabled={isBusy}
                  className="w-full bg-background border border-primary/40 text-textMain rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-55"
                />
                <p className="text-[11px] text-textMuted">
                  Enter the HF repo ID where your LoRA adapters were previously pushed.
                  The base model above will be loaded first, then adapters attached.
                </p>
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-2xl border border-border/40 bg-background/60 p-3">
              <div className="text-[10px] font-bold uppercase tracking-[0.22em] text-textMuted mb-1">Selected</div>
              <div className="text-xs font-semibold text-textMain break-words">{hfModelPath || '—'}</div>
            </div>
            <div className="rounded-2xl border border-border/40 bg-background/60 p-3">
              <div className="text-[10px] font-bold uppercase tracking-[0.22em] text-textMuted mb-1">Status</div>
              <div className="text-xs font-semibold text-textMain capitalize">{status.replace(/_/g, ' ')}</div>
            </div>
          </div>

          {isLoaded ? (
            <div className="mt-auto flex items-center justify-between gap-3 pt-3 border-t border-border/30">
              <div className="flex items-center gap-2 text-success text-sm font-medium">
                <CheckCircle2 size={18} /> Model configured
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => navigate('/dataset')}
                  className="px-4 py-2.5 bg-primary hover:bg-primaryHover text-white font-medium rounded-xl transition-all flex items-center gap-2 text-sm"
                >
                  Move to Dataset <ArrowRight size={14} />
                </button>
                <button
                  onClick={handleEdit}
                  className="px-4 py-2.5 bg-surfaceHover hover:bg-primary/20 text-primary border border-primary/20 font-medium rounded-xl transition-all flex items-center gap-2 text-sm"
                >
                  <Edit2 size={14} /> Edit Model
                </button>
              </div>
            </div>
          ) : (
            <div className="mt-auto grid grid-cols-1 sm:grid-cols-2 gap-2">
              <button
                onClick={handleLoad}
                disabled={isBusy || (isCustomModel && !customPath.trim()) || status === 'training'}
                className="w-full px-5 py-3 bg-primary hover:bg-primaryHover disabled:bg-surfaceHover disabled:text-textMuted text-white font-semibold rounded-xl transition-all flex items-center justify-center gap-2 text-sm"
              >
                {isBusy ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
                {isBusy ? 'Loading model...' : 'Load Model'}
              </button>
              <button
                onClick={handleCancelLoad}
                disabled={!isBusy}
                className="w-full px-5 py-3 bg-surfaceHover hover:bg-error/15 disabled:opacity-50 text-error font-semibold rounded-xl transition-all border border-error/20 flex items-center justify-center gap-2 text-sm"
              >
                <XCircle size={16} />
                Cancel
              </button>
            </div>
          )}
        </div>

        <div className="bg-surface border border-border/50 rounded-3xl p-5 shadow-xl flex flex-col items-center justify-center min-h-0 overflow-hidden relative">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_18%_16%,rgba(59,130,246,0.12),transparent_36%),radial-gradient(circle_at_86%_80%,rgba(16,185,129,0.10),transparent_30%)] pointer-events-none" />
          {isBusy ? (
            <div className="w-full max-w-2xl flex flex-col items-center gap-6">
              <div className="relative w-44 h-44 flex items-center justify-center">
                <div className="absolute inset-0 rounded-full border border-primary/30 animate-[spin_14s_linear_infinite]" />
                <div className="absolute inset-7 rounded-full border border-accent/25 animate-[spin_10s_linear_infinite_reverse]" />
                <div className="absolute inset-[3.4rem] rounded-2xl bg-primary/90 rotate-45 shadow-[0_0_40px_rgba(59,130,246,0.45)] flex items-center justify-center">
                  <Box size={22} className="text-white -rotate-45" />
                </div>
              </div>

              <div className="w-full">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-bold uppercase tracking-[0.24em] text-textMuted">Loading weights</span>
                  <span className="text-xs font-mono font-semibold text-textMain">{loadingProgress}%</span>
                </div>
                <div className="w-full h-2.5 rounded-full bg-background border border-border/50 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-success via-primary to-accent transition-all duration-300"
                    style={{ width: `${loadingProgress}%` }}
                  />
                </div>
                <p className="text-xs text-textMuted mt-3 text-center">{loadingMessage || 'Preparing model download.'}</p>
              </div>
            </div>
          ) : isLoaded ? (
            <div className="w-full max-w-2xl flex flex-col items-center gap-5">
              <div className="relative w-full max-w-2xl h-56 rounded-3xl border border-primary/20 bg-background/70 overflow-hidden">
                <div className="absolute top-5 left-6 right-6 h-px bg-primary/25" />
                <div className="absolute top-10 left-6 right-6 h-px bg-accent/20" />
                <div className="absolute top-4 left-4 w-2 h-2 rounded-full bg-primary animate-pulse" />
                <div className="absolute top-4 left-8 w-2 h-2 rounded-full bg-accent/80 animate-pulse" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="relative w-36 h-36">
                    <div className="absolute inset-0 rounded-3xl border border-primary/35 animate-[spin_10s_linear_infinite]" />
                    <div className="absolute inset-5 rounded-2xl border border-accent/30 animate-[spin_8s_linear_infinite_reverse]" />
                    <div className="absolute inset-[2.7rem] rounded-xl bg-primary/85 rotate-45 shadow-[0_0_36px_rgba(59,130,246,0.4)] flex items-center justify-center">
                      <Box size={20} className="text-white -rotate-45" />
                    </div>
                  </div>
                </div>
                <div className="absolute bottom-4 left-5 text-[10px] font-mono text-primary/80">weights mapped</div>
                <div className="absolute bottom-4 right-5 text-[10px] font-mono text-accent/80">pipeline ready</div>
                <div className="absolute top-4 left-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-background border border-border flex items-center justify-center">
                  <Layers size={12} className="text-primary" />
                </div>
                <div className="absolute bottom-6 right-7 w-7 h-7 rounded-full bg-background border border-border flex items-center justify-center">
                  <Cpu size={12} className="text-accent" />
                </div>
                <div className="absolute left-4 right-4 top-4 flex items-center justify-between pointer-events-none">
                  <span className="text-[10px] uppercase tracking-[0.2em] font-bold text-textMuted/80 bg-surface/70 border border-border/30 rounded-full px-2.5 py-1">
                    Step {currentStep || 0}
                  </span>
                  <span className="text-[10px] uppercase tracking-[0.2em] font-bold text-textMuted/80 bg-surface/70 border border-border/30 rounded-full px-2.5 py-1 max-w-[58%] truncate">
                    {hfModelPath}
                  </span>
                </div>
              </div>

              <div className="text-center flex flex-col items-center gap-2">
                <span className="inline-flex items-center gap-2 bg-success/10 text-success text-[10px] font-bold px-3 py-1 rounded-full uppercase tracking-[0.24em] border border-success/20">
                  <Sparkles size={11} /> Ready Pipeline
                </span>
                <h2 className="text-xl md:text-2xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-primary to-accent break-words">
                  {model}
                </h2>
              </div>

              <div className="w-full grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="rounded-2xl border border-border/40 bg-background/70 p-3">
                  <div className="text-[10px] uppercase tracking-[0.2em] font-bold text-textMuted flex items-center gap-1.5">
                    <Cpu size={12} /> Compute
                  </div>
                  <div className="text-xs text-textMain font-semibold mt-2 break-words">{gpuName || 'CPU Mode'}</div>
                </div>
                <div className="rounded-2xl border border-border/40 bg-background/70 p-3">
                  <div className="text-[10px] uppercase tracking-[0.2em] font-bold text-textMuted flex items-center gap-1.5">
                    <HardDrive size={12} /> GPU Memory
                  </div>
                  <div className="text-xs text-textMain font-semibold mt-2">{gpuMem.toFixed(2)} / {gpuTotalMem.toFixed(2)} GB</div>
                  <div className="mt-2 h-1.5 rounded-full bg-surfaceHover overflow-hidden">
                    <div className="h-full bg-primary" style={{ width: `${gpuPercent}%` }} />
                  </div>
                </div>
                <div className="rounded-2xl border border-border/40 bg-background/70 p-3">
                  <div className="text-[10px] uppercase tracking-[0.2em] font-bold text-textMuted flex items-center gap-1.5">
                    <Database size={12} /> RAM
                  </div>
                  <div className="text-xs text-textMain font-semibold mt-2">{ram.toFixed(2)} / {ramTotal.toFixed(2)} GB</div>
                  <div className="mt-2 h-1.5 rounded-full bg-surfaceHover overflow-hidden">
                    <div className="h-full bg-accent" style={{ width: `${ramPercent}%` }} />
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="w-full max-w-xl flex flex-col items-center gap-4 opacity-75">
              <div className="relative w-44 h-44 flex items-center justify-center">
                <div className="absolute inset-0 rounded-full border border-dashed border-border/30" />
                <div className="absolute inset-7 rounded-full border border-dashed border-border/20" />
                <div className="absolute inset-[3.4rem] rounded-2xl bg-background border border-border/40 rotate-45 flex items-center justify-center">
                  <Box size={20} className="text-textMuted -rotate-45" />
                </div>
              </div>
              <div className="text-center">
                <h2 className="text-lg font-bold text-textMuted">No model loaded</h2>
                <p className="text-xs text-textMuted mt-1">Load a model to unlock dataset and training.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
