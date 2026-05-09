import React, { useState } from 'react';
import StepFlow from '../components/ui/StepFlow';
import { api } from '../api';
import { Download, Layers, Box, Loader2, CheckCircle2 } from 'lucide-react';
import { useGlobalState } from '../context/GlobalState';

export default function ExportPanel() {
  const { status } = useGlobalState();
  const [downloading, setDownloading] = useState(null);
  const [toast, setToast] = useState(null);

  const handleExport = async (type) => {
    setDownloading(type);
    const res = await api.export(type);
    setDownloading(null);
    if (res.success) {
       setToast({ type: 'success', msg: `Successfully exported ${type === 'lora' ? 'LoRA Adapter' : 'Merged Model'}!` });
       setTimeout(() => setToast(null), 3000);
    }
  };

  const isReady = status === 'done';

  return (
    <div className="max-w-4xl mx-auto flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <StepFlow />
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold text-textMain tracking-tight">Export Model</h1>
        <p className="text-textMuted">Download your fine-tuned model artifacts.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
        <div className="bg-surface border border-border/50 p-8 rounded-3xl shadow-xl flex flex-col relative overflow-hidden group">
          <div className="absolute -top-10 -right-10 w-40 h-40 bg-primary/10 rounded-full blur-3xl group-hover:bg-primary/20 transition-all duration-700"></div>
          
          <div className="p-4 bg-primary/10 w-max rounded-2xl mb-6 relative z-10 border border-primary/20 shadow-inner">
             <Layers size={32} className="text-primary" />
          </div>
          
          <h3 className="text-xl font-bold text-textMain mb-2 relative z-10">LoRA Adapter</h3>
          <p className="text-sm text-textMuted mb-10 flex-1 relative z-10 leading-relaxed">The lightweight adapter weights generated during fine-tuning. Requires the original base model configuration to run.</p>
          
          <button 
             onClick={() => handleExport('lora')}
             disabled={!isReady || downloading !== null}
             className="w-full px-6 py-3.5 bg-surfaceHover border border-primary/50 text-textMain font-bold rounded-xl hover:bg-primary/10 hover:text-primary hover:border-primary transition-all disabled:opacity-50 disabled:hover:bg-surfaceHover disabled:hover:text-textMain disabled:hover:border-primary/50 flex items-center justify-center gap-2 relative z-10"
          >
             {downloading === 'lora' ? <Loader2 size={18} className="animate-spin" /> : <Download size={18} />}
             {downloading === 'lora' ? 'Preparing Download...' : 'Download Adapter'}
          </button>
        </div>

        <div className="bg-surface border border-border/50 p-8 rounded-3xl shadow-xl flex flex-col relative overflow-hidden group">
          <div className="absolute -top-10 -right-10 w-40 h-40 bg-accent/10 rounded-full blur-3xl group-hover:bg-accent/20 transition-all duration-700"></div>
          
          <div className="p-4 bg-accent/10 w-max rounded-2xl mb-6 relative z-10 border border-accent/20 shadow-inner">
             <Box size={32} className="text-accent" />
          </div>
          
          <h3 className="text-xl font-bold text-textMain mb-2 relative z-10">Merged Model</h3>
          <p className="text-sm text-textMuted mb-10 flex-1 relative z-10 leading-relaxed">Full standalone model with LoRA weights merged directly into the base model. Ready for immediate deployment.</p>
          
          <button 
             onClick={() => handleExport('merged')}
             disabled={!isReady || downloading !== null}
             className="w-full px-6 py-3.5 bg-accent hover:bg-[#0EA5E9] text-white font-bold rounded-xl transition-all shadow-[0_4px_14px_0_rgba(16,185,129,0.39)] hover:shadow-[0_6px_20px_rgba(16,185,129,0.23)] hover:-translate-y-0.5 disabled:opacity-50 disabled:shadow-none disabled:transform-none flex items-center justify-center gap-2 relative z-10"
          >
             {downloading === 'merged' ? <Loader2 size={18} className="animate-spin" /> : <Download size={18} />}
             {downloading === 'merged' ? 'Merging & Exporting...' : 'Export Full Model'}
          </button>
        </div>
      </div>

      {toast && (
        <div className="flex items-center gap-3 p-4 bg-success/10 border border-success/20 text-success rounded-xl shadow-lg animate-in slide-in-from-bottom-4 fade-in mt-4 border-l-4">
          <CheckCircle2 size={20} />
          <span className="font-medium text-sm">{toast.msg}</span>
        </div>
      )}
    </div>
  );
}
