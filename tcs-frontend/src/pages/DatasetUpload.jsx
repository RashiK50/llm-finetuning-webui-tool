import React, { useState, useRef } from 'react';
import { useGlobalState } from '../context/GlobalState';
import StepFlow from '../components/ui/StepFlow';
import { api } from '../api';
import { UploadCloud, FileJson, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import clsx from 'clsx';

export default function DatasetUpload() {
  const { dataset, setDataset, status, setStatus } = useGlobalState();
  const [isDragActive, setIsDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [toast, setToast] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => { 
    e.preventDefault(); 
    e.stopPropagation(); 
    if (e.type === "dragenter" || e.type === "dragover") { setIsDragActive(true); } 
    else if (e.type === "dragleave") { setIsDragActive(false); } 
  };

  const handleDrop = (e) => {
    e.preventDefault(); 
    e.stopPropagation(); 
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelected = (selected) => {
    if (selected.type !== "application/json" && !selected.name.endsWith(".json")) {
      setToast({ type: 'error', msg: 'Please upload a JSON file only.' });
      return;
    }
    setFile(selected);
    setToast(null);
  };

  const handleUploadClick = () => fileInputRef.current?.click();

  const handleUpload = async () => {
    if (!file) return;
    setIsLoading(true);
    const res = await api.uploadDataset(file);
    setIsLoading(false);
    if (res.success) {
      setDataset({ size: res.size, preview: res.preview });
      if(status === 'idle' || status === 'model_ready') {
         setStatus('dataset_ready');
      }
      setToast({ type: 'success', msg: 'Dataset processed successfully!' });
      setTimeout(() => setToast(null), 3000);
    } else {
      setToast({ type: 'error', msg: 'Failed to process dataset.' });
    }
  };

  return (
    <div className="max-w-4xl mx-auto flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <StepFlow />
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold text-textMain tracking-tight">Dataset Upload</h1>
        <p className="text-textMuted">Upload a JSON file containing instruction-output pairs for supervised fine-tuning.</p>
      </div>

      <div className="bg-surface border border-border/50 rounded-3xl p-8 shadow-2xl relative">
        <div 
          className={clsx(
            "w-full border-2 border-dashed rounded-2xl p-12 flex flex-col items-center justify-center gap-4 transition-all duration-300 cursor-pointer",
            isDragActive ? "border-primary bg-primary/5 shadow-[0_0_20px_rgba(59,130,246,0.15)]" : "border-border/60 hover:border-textMuted/50 bg-background/50 hover:bg-background"
          )}
          onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop} onClick={handleUploadClick}
        >
          <input type="file" className="hidden" ref={fileInputRef} accept=".json,application/json" onChange={(e) => { if (e.target.files && e.target.files[0]) handleFileSelected(e.target.files[0]); }} />
          
          <div className="p-4 bg-surfaceHover rounded-full text-primary mb-2 shadow-sm border border-border/50">
            <UploadCloud size={32} />
          </div>
          <div className="text-center">
            <p className="text-base font-semibold text-textMain mb-1">Click to upload or drag and drop</p>
            <p className="text-sm text-textMuted">JSON format only (max. 100MB)</p>
          </div>
          
          {file && (
            <div className="mt-4 flex items-center gap-3 bg-surface border border-border/50 px-4 py-2 rounded-xl text-sm font-medium text-textMain shadow-sm">
              <FileJson size={16} className="text-primary" />
              <span className="truncate max-w-[200px]">{file.name}</span>
              <button 
                onClick={(e) => { e.stopPropagation(); setFile(null); }}
                className="text-textMuted hover:text-error ml-2"
              >
                &times;
              </button>
            </div>
          )}
        </div>

        {file && (
           <div className="mt-6 flex justify-end">
             <button 
               onClick={handleUpload}
               disabled={isLoading || status === 'training'}
               className="px-8 py-3.5 bg-primary hover:bg-primaryHover disabled:bg-surfaceHover disabled:text-textMuted text-white font-medium rounded-xl transition-all shadow-[0_4px_14px_0_rgba(59,130,246,0.39)] hover:shadow-[0_6px_20px_rgba(59,130,246,0.23)] flex items-center gap-2 hover:-translate-y-0.5 disabled:shadow-none disabled:transform-none"
             >
               {isLoading ? <Loader2 size={18} className="animate-spin" /> : <UploadCloud size={18} />}
               {isLoading ? 'Processing...' : 'Upload & Process'}
             </button>
           </div>
        )}

        {dataset && dataset.preview && (
          <div className="mt-8 border-t border-border/30 pt-8 animate-in fade-in duration-500">
            <h3 className="text-xs font-semibold text-textMuted tracking-wider uppercase mb-5 flex items-center gap-2">
               <CheckCircle2 size={14} className="text-success" />
               Dataset Preview
            </h3>
            <div className="flex flex-col gap-4">
              {dataset.preview.map((item, idx) => (
                <div key={idx} className="bg-background border border-border/50 rounded-2xl p-5 overflow-hidden relative group shadow-sm hover:shadow-md transition-shadow">
                  <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary/30 group-hover:bg-primary transition-colors"></div>
                  <div className="mb-3">
                    <span className="text-[10px] font-bold text-textMuted uppercase bg-surface px-2 py-1 rounded inline-block mb-1 border border-border/50">Instruction</span>
                    <p className="text-sm text-textMain font-medium leading-relaxed">{item.instruction}</p>
                  </div>
                  <div>
                     <span className="text-[10px] font-bold text-textMuted uppercase bg-surface px-2 py-1 rounded inline-block mb-1 border border-border/50">Output</span>
                     <p className="text-sm text-textMain/80 leading-relaxed whitesapce-pre-wrap">{item.output}</p>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-5 text-center text-xs text-textMuted font-medium">Showing {dataset.preview.length} sample entries</div>
          </div>
        )}
      </div>

      {toast && (
        <div className={clsx(
          "flex items-center gap-3 p-4 rounded-xl border shadow-lg animate-in slide-in-from-bottom-8 fade-in",
          toast.type === 'error' ? "bg-error/10 border-error/20 text-error" : "bg-success/10 border-success/20 text-success"
        )}>
          {toast.type === 'error' ? <AlertCircle size={20} /> : <CheckCircle2 size={20} />}
          <span className="font-medium text-sm">{toast.msg}</span>
        </div>
      )}
    </div>
  );
}
