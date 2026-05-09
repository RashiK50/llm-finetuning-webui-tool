import React, { useState } from 'react';
import { useGlobalState } from '../context/GlobalState';
import StepFlow from '../components/ui/StepFlow';
import { api } from '../api';
import { Play, RotateCcw, AlertTriangle, Loader2 } from 'lucide-react';
import clsx from 'clsx';
import { useNavigate } from 'react-router-dom';

const SliderConfig = ({ label, min, max, step, value, onChange, disabled, tooltip }) => (
  <div className="flex flex-col gap-2">
    <div className="flex justify-between items-center group relative">
      <label className="text-sm font-semibold text-textMain">{label}</label>
      <span className="text-xs font-mono font-bold bg-background px-2 py-1 rounded text-primary border border-border/50 shadow-sm">{value}</span>
      <div className="absolute left-0 -top-10 bg-surface border border-border px-3 py-1.5 rounded-lg text-xs font-medium text-textMain opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 shadow-xl whitespace-nowrap">
        {tooltip}
      </div>
    </div>
    <input 
      type="range"
      min={min} max={max} step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      disabled={disabled}
      className="w-full h-2 bg-background border border-border/50 rounded-lg appearance-none cursor-pointer disabled:opacity-50 accent-primary focus:outline-none focus:ring-2 focus:ring-primary/30"
    />
  </div>
);

export default function TrainingPanel() {
  const { model, dataset, status, setStatus } = useGlobalState();
  const navigate = useNavigate();

  const [config, setConfig] = useState({ lora_rank: 16, lora_alpha: 32, lora_dropout: 0.05, learning_rate: 0.0002, epochs: 3, batch_size: 2 });
  const [isLoading, setIsLoading] = useState(false);

  const resetToDefault = () => setConfig({ lora_rank: 16, lora_alpha: 32, lora_dropout: 0.05, learning_rate: 0.0002, epochs: 3, batch_size: 2 });

  const handleStartTraining = async () => {
    setIsLoading(true);
    setStatus('training');
    const res = await api.train(config);
    if (res.success) {
      navigate('/live-progress');
    }
    setIsLoading(false);
  };

  const isFormDisabled = status === 'training' || !model || !dataset;

  return (
    <div className="max-w-4xl mx-auto flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <StepFlow />
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold text-textMain tracking-tight">Training Configuration</h1>
        <p className="text-textMuted">Configure hyperparameters for LoRA Supervised Fine-Tuning.</p>
      </div>

      {(!model || !dataset) && (
        <div className="flex items-start gap-4 bg-warning/5 border border-warning/20 text-warning p-5 rounded-2xl shadow-sm">
          <AlertTriangle size={20} className="shrink-0 mt-0.5" />
          <div className="flex flex-col gap-1">
            <span className="font-bold text-sm tracking-wide uppercase">Prerequisites Missing</span>
            <span className="text-sm font-medium opacity-90">Please ensure a model is loaded and a dataset is uploaded before starting training.</span>
          </div>
        </div>
      )}

      <div className="bg-surface border border-border/50 rounded-3xl p-8 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-accent/5 rounded-full blur-3xl -mr-16 -mt-16 transition-all duration-700 pointer-events-none"></div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-16 gap-y-10 relative z-10">
          <div className="flex flex-col gap-8">
            <h3 className="text-xs font-bold text-textMuted uppercase tracking-wider border-b border-border/50 pb-3 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-primary/50"></span> LoRA Adapter
            </h3>
            <SliderConfig label="LoRA Rank" min={4} max={64} step={4} value={config.lora_rank} onChange={(v) => setConfig({...config, lora_rank: v})} disabled={isFormDisabled} tooltip="Rank of the update matrices." />
            <SliderConfig label="LoRA Alpha" min={8} max={128} step={8} value={config.lora_alpha} onChange={(v) => setConfig({...config, lora_alpha: v})} disabled={isFormDisabled} tooltip="Scaling factor for LoRA." />
            <SliderConfig label="LoRA Dropout" min={0} max={0.5} step={0.01} value={config.lora_dropout} onChange={(v) => setConfig({...config, lora_dropout: v})} disabled={isFormDisabled} tooltip="Dropout probability." />
          </div>

          <div className="flex flex-col gap-8">
             <h3 className="text-xs font-bold text-textMuted uppercase tracking-wider border-b border-border/50 pb-3 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-accent/50"></span> Training Params
             </h3>
             
             <div className="flex flex-col gap-2 group relative">
               <div className="flex justify-between items-center">
                 <label className="text-sm font-semibold text-textMain">Learning Rate</label>
                 <div className="absolute left-0 -top-10 bg-surface border border-border px-3 py-1.5 rounded-lg text-xs font-medium text-textMain opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 shadow-xl whitespace-nowrap">
                   Optimizer learning rate.
                 </div>
               </div>
               <input type="number" min="0" step="0.0001" value={config.learning_rate} onChange={(e) => setConfig({...config, learning_rate: e.target.value})} disabled={isFormDisabled}
                 className="w-full bg-background border border-border/50 text-textMain font-mono text-sm rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50 transition-colors shadow-inner"
               />
             </div>

             <SliderConfig label="Epochs" min={1} max={10} step={1} value={config.epochs} onChange={(v) => setConfig({...config, epochs: v})} disabled={isFormDisabled} tooltip="Number of training passes over dataset." />

             <div className="flex flex-col gap-2 group relative">
               <div className="flex justify-between items-center">
                 <label className="text-sm font-semibold text-textMain">Batch Size</label>
                 <div className="absolute left-0 -top-10 bg-surface border border-border px-3 py-1.5 rounded-lg text-xs font-medium text-textMain opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 shadow-xl whitespace-nowrap">
                   Samples per forward pass.
                 </div>
               </div>
               <div className="relative">
                 <select value={config.batch_size} onChange={(e) => setConfig({...config, batch_size: Number(e.target.value)})} disabled={isFormDisabled} className="w-full bg-background border border-border/50 text-textMain text-sm rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary/50 appearance-none disabled:opacity-50 shadow-inner">
                   {[1, 2, 4, 8].map(sz => <option key={sz} value={sz}>{sz}</option>)}
                 </select>
                 <div className="absolute inset-y-0 right-4 flex items-center pointer-events-none text-textMuted">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                 </div>
               </div>
             </div>
          </div>
        </div>

        <div className="mt-12 pt-6 border-t border-border/50 flex items-center justify-between relative z-10">
          <button onClick={resetToDefault} disabled={isFormDisabled} className="flex items-center gap-2 text-sm font-medium text-textMuted hover:text-textMain transition-colors disabled:opacity-50 px-4 py-2 rounded-lg hover:bg-surfaceHover">
            <RotateCcw size={16} /> Reset to Defaults
          </button>
          <button onClick={handleStartTraining} disabled={isFormDisabled || isLoading} className="px-8 py-3.5 bg-accent hover:bg-[#0EA5E9] disabled:bg-surfaceHover disabled:text-textMuted text-white font-bold rounded-xl transition-all shadow-[0_4px_14px_0_rgba(16,185,129,0.39)] hover:shadow-[0_6px_20px_rgba(16,185,129,0.23)] hover:-translate-y-0.5 disabled:shadow-none disabled:transform-none flex items-center justify-center gap-2">
            {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} className="fill-white" />}
            Start Training
          </button>
        </div>
      </div>
    </div>
  );
}
