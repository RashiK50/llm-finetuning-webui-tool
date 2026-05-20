import React, { useEffect, useState } from 'react';
import StepFlow from '../components/ui/StepFlow';
import { api } from '../api';
import { Download, Layers, Box, Loader2, CheckCircle2, AlertCircle, FolderOpen, Link as LinkIcon, Upload } from 'lucide-react';
import { useGlobalState } from '../context/GlobalState';
import toast from 'react-hot-toast';

export default function ExportPanel() {
  const {
    status, model, modelType, loadingProgress, loadingMessage,
    exportDownloadUrl, exportArtifactPath, exportType, exportError,
    hfPushStatus, hfPushMessage, hfPushedRepo,
    ggufPushStatus, ggufPushMessage, ggufPushedRepo,
  } = useGlobalState();
  const [downloading, setDownloading] = useState(null);
  const [lastExport, setLastExport] = useState(null);
  const [lastDownloadedUrl, setLastDownloadedUrl] = useState('');
  // HF push repo inputs
  const [loraHfRepo, setLoraHfRepo] = useState('');
  const [mergedHfRepo, setMergedHfRepo] = useState('');
  // GGUF state
  const [ggufMergedDir, setGgufMergedDir] = useState('./outputs/_tmp_export_merged');
  const [ggufHfRepo, setGgufHfRepo] = useState('');
  const [ggufQuant, setGgufQuant] = useState('f16');
  const [ggufLoading, setGgufLoading] = useState(false);

  useEffect(() => {
    // Safety guard: if backend is no longer exporting, never keep local loading stuck.
    if (status !== 'exporting' && downloading !== null) {
      setDownloading(null);
    }
  }, [status, downloading]);

  useEffect(() => {
    if (status === 'error' && exportError) {
      toast.error(`Export failed: ${exportError}`);
    }
  }, [status, exportError]);

  useEffect(() => {
    const triggerDownload = async () => {
      if (!exportDownloadUrl || status !== 'finished') return;
      const fullUrl = `http://localhost:8000${exportDownloadUrl}`;
      if (fullUrl === lastDownloadedUrl) return;

      try {
        const res = await fetch(fullUrl);
        if (!res.ok) throw new Error(`Download failed (${res.status})`);
        const blob = await res.blob();
        const objectUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = objectUrl;
        const fallbackName = exportType ? `export_${exportType}.zip` : 'model_export.zip';
        a.download = exportArtifactPath?.split('/').pop() || fallbackName;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(objectUrl);

        setLastDownloadedUrl(fullUrl);
        setLastExport({
          type: exportType || downloading || 'model',
          message: `${(exportType || downloading || 'Model').toUpperCase()} export complete.`,
          path: exportArtifactPath || '',
          downloadUrl: fullUrl,
        });
        toast.success('Export ZIP downloaded.');
      } catch (e) {
        toast.error(String(e?.message || 'Export download failed.'));
      }
    };

    triggerDownload();
  }, [status, exportDownloadUrl, exportArtifactPath, exportType, downloading, lastDownloadedUrl]);

  const handleExport = async (type) => {
    try {
      setDownloading(type);
      setLastDownloadedUrl('');
      const hfRepo = type === 'lora' ? loraHfRepo : mergedHfRepo;
      const res = await api.export(type, hfRepo.trim());
      if (res?.success && res?.message) {
        toast.success(res.message);
      }
    } catch {
      // Error toast is handled by api.request.
    }
  };

  const handleExportGguf = async () => {
    if (!ggufMergedDir.trim() || !ggufHfRepo.trim()) {
      toast.error('Enter both the merged model directory and the HF repo ID.');
      return;
    }
    try {
      setGgufLoading(true);
      const res = await api.exportGguf(ggufMergedDir.trim(), ggufHfRepo.trim(), ggufQuant);
      if (res?.success) toast.success('GGUF export started. Check status panel for progress.');
    } catch {
      // api.request shows toast
    } finally {
      setGgufLoading(false);
    }
  };

  const isReady = status === 'done' || status === 'finished';
  const isExporting = status === 'exporting' || downloading !== null;
  const isPeftLora = modelType === 'peft_lora';
  const modelTypeLabel = isPeftLora ? 'PEFT LoRA' : (modelType === 'merged_base' ? 'Merged/Base' : 'Unknown');

  return (
    <div className="max-w-4xl mx-auto min-h-0 flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <StepFlow />
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold text-textMain tracking-tight">Export Model</h1>
        <p className="text-textMuted">Download your fine-tuned model artifacts.</p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.4fr)_360px] gap-6 mt-2 min-h-0">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-surface border border-border/50 p-8 rounded-3xl shadow-xl flex flex-col relative overflow-hidden group">
          <div className="absolute -top-10 -right-10 w-40 h-40 bg-primary/10 rounded-full blur-3xl group-hover:bg-primary/20 transition-all duration-700"></div>
          
          <div className="p-4 bg-primary/10 w-max rounded-2xl mb-6 relative z-10 border border-primary/20 shadow-inner">
             <Layers size={32} className="text-primary" />
          </div>
          
          <h3 className="text-xl font-bold text-textMain mb-2 relative z-10">LoRA Adapter</h3>
          <p className="text-sm text-textMuted mb-4 flex-1 relative z-10 leading-relaxed">The lightweight adapter weights generated during fine-tuning. Requires the original base model configuration to run.</p>
          
          <input
            type="text"
            value={loraHfRepo}
            onChange={(e) => setLoraHfRepo(e.target.value)}
            placeholder="username/my-lora-repo (optional HF push)"
            className="w-full mb-4 px-3 py-2 text-sm bg-background border border-border/50 text-textMain rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/50 relative z-10"
          />

          <button 
             onClick={() => handleExport('lora')}
             disabled={!isReady || downloading !== null || !isPeftLora}
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
          <p className="text-sm text-textMuted mb-4 flex-1 relative z-10 leading-relaxed">Full standalone model with LoRA weights merged directly into the base model. Ready for immediate deployment.</p>

          <input
            type="text"
            value={mergedHfRepo}
            onChange={(e) => setMergedHfRepo(e.target.value)}
            placeholder="username/my-merged-repo (optional HF push)"
            className="w-full mb-4 px-3 py-2 text-sm bg-background border border-border/50 text-textMain rounded-xl focus:outline-none focus:ring-2 focus:ring-accent/50 relative z-10"
          />
          
          <button 
             onClick={() => handleExport('merged')}
             disabled={!isReady || downloading !== null}
             className="w-full px-6 py-3.5 bg-accent hover:bg-[#0EA5E9] text-white font-bold rounded-xl transition-all shadow-[0_4px_14px_0_rgba(16,185,129,0.39)] hover:shadow-[0_6px_20px_rgba(16,185,129,0.23)] hover:-translate-y-0.5 disabled:opacity-50 disabled:shadow-none disabled:transform-none flex items-center justify-center gap-2 relative z-10"
          >
             {downloading === 'merged' ? <Loader2 size={18} className="animate-spin" /> : <Download size={18} />}
             {downloading === 'merged' ? 'Merging & Exporting...' : 'Export Full Model'}
          </button>
          </div>

          {/* ── GGUF Export Card (new) ── */}
          <div className="bg-surface border border-border/50 p-8 rounded-3xl shadow-xl flex flex-col relative overflow-hidden group md:col-span-2 xl:col-span-1">
          <div className="absolute -top-10 -right-10 w-40 h-40 bg-warning/10 rounded-full blur-3xl group-hover:bg-warning/20 transition-all duration-700"></div>

          <div className="p-4 bg-warning/10 w-max rounded-2xl mb-6 relative z-10 border border-warning/20 shadow-inner">
             <Upload size={32} className="text-warning" />
          </div>

          <h3 className="text-xl font-bold text-textMain mb-2 relative z-10">GGUF Export → HF</h3>
          <p className="text-sm text-textMuted mb-4 flex-1 relative z-10 leading-relaxed">
            Convert the merged model to GGUF format (for llama.cpp / Ollama) and push it directly to Hugging Face.
            First export as Merged above, then run GGUF export.
          </p>

          <div className="flex flex-col gap-2 mb-4 relative z-10">
            <input
              type="text"
              value={ggufMergedDir}
              onChange={(e) => setGgufMergedDir(e.target.value)}
              placeholder="./outputs/_tmp_export_merged"
              className="w-full px-3 py-2 text-sm bg-background border border-border/50 text-textMain rounded-xl focus:outline-none focus:ring-2 focus:ring-warning/50"
            />
            <input
              type="text"
              value={ggufHfRepo}
              onChange={(e) => setGgufHfRepo(e.target.value)}
              placeholder="username/my-model-gguf (HF repo)"
              className="w-full px-3 py-2 text-sm bg-background border border-border/50 text-textMain rounded-xl focus:outline-none focus:ring-2 focus:ring-warning/50"
            />
            <select
              value={ggufQuant}
              onChange={(e) => setGgufQuant(e.target.value)}
              className="w-full px-3 py-2 text-sm bg-background border border-border/50 text-textMain rounded-xl focus:outline-none appearance-none"
            >
              <option value="f16">f16 (full precision, largest)</option>
              <option value="q8_0">q8_0 (8-bit quantised)</option>
              <option value="q4_k_m">q4_k_m (4-bit, recommended)</option>
            </select>
          </div>

          {ggufPushStatus && (
            <div className={`text-xs mb-3 font-semibold relative z-10 ${ggufPushStatus === 'pushed' ? 'text-success' : ggufPushStatus === 'error' ? 'text-error' : 'text-warning'}`}>
              {ggufPushMessage}
              {ggufPushedRepo && (
                <a href={ggufPushedRepo} target="_blank" rel="noreferrer" className="ml-2 underline text-primary">View on HF</a>
              )}
            </div>
          )}

          <button
            onClick={handleExportGguf}
            disabled={!isReady || ggufLoading || ggufPushStatus === 'converting' || ggufPushStatus === 'pushing'}
            className="w-full px-6 py-3.5 bg-warning/90 hover:bg-warning text-white font-bold rounded-xl transition-all disabled:opacity-50 flex items-center justify-center gap-2 relative z-10"
          >
            {ggufLoading || ggufPushStatus === 'converting' || ggufPushStatus === 'pushing'
              ? <Loader2 size={18} className="animate-spin" />
              : <Upload size={18} />}
            {ggufLoading || ggufPushStatus === 'converting' ? 'Converting...'
              : ggufPushStatus === 'pushing' ? 'Uploading to HF...'
              : 'Convert GGUF & Push to HF'}
          </button>
          </div>
        </div>

        <div className="bg-surface border border-border/50 rounded-2xl p-4 shadow-sm flex flex-col gap-2 h-max">
          <div className="flex items-center gap-2 text-sm font-semibold text-textMain">
            {isExporting ? <Loader2 size={16} className="animate-spin text-primary" /> : (isReady ? <CheckCircle2 size={16} className="text-success" /> : <AlertCircle size={16} className="text-warning" />)}
            Export Status
          </div>
          <div className="text-sm text-textMuted">
            Model loaded: <span className="text-textMain font-semibold">{model || 'No model loaded'}</span>
          </div>
          <div className="text-sm text-textMuted">
            Model type: <span className={`font-semibold ${isPeftLora ? 'text-success' : 'text-warning'}`}>{modelTypeLabel}</span>
          </div>
          <div className="text-sm text-textMuted">
            Current state: <span className="text-textMain font-semibold">{status}</span>
          </div>
          {!isPeftLora && (
            <div className="text-xs text-warning">
              LoRA Adapter export is disabled for merged/base models. Use Export Full Model.
            </div>
          )}
          {isExporting && (
            <>
              <div className="text-sm text-textMuted">
                Progress: <span className="text-textMain font-semibold">{Math.max(0, Math.min(100, Number(loadingProgress || 0)))}%</span>
              </div>
              <div className="w-full h-2 rounded-full bg-background border border-border/50 overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${Math.max(0, Math.min(100, Number(loadingProgress || 0)))}%` }}
                />
              </div>
              {loadingMessage && <div className="text-xs text-textMuted">{loadingMessage}</div>}
            </>
          )}
          {lastExport?.message && (
            <div className="text-sm text-success">{lastExport.message}</div>
          )}
          {/* HF push status */}
          {hfPushStatus && (
            <div className={`text-sm font-semibold ${hfPushStatus === 'pushed' ? 'text-success' : hfPushStatus === 'error' ? 'text-error' : 'text-warning'}`}>
              HF: {hfPushMessage}
              {hfPushedRepo && (
                <a href={hfPushedRepo} target="_blank" rel="noreferrer" className="ml-2 underline text-primary text-xs">View repo</a>
              )}
            </div>
          )}
          {(lastExport?.path || exportArtifactPath) && (
            <div className="text-sm text-textMuted flex items-start gap-2">
              <FolderOpen size={14} className="mt-0.5 text-primary shrink-0" />
              <span className="break-all">Saved at: <span className="text-textMain">{lastExport?.path || exportArtifactPath}</span></span>
            </div>
          )}
          {(lastExport?.downloadUrl || exportDownloadUrl) && (
            <a
              href={lastExport?.downloadUrl || `http://localhost:8000${exportDownloadUrl}`}
              target="_blank"
              rel="noreferrer"
              className="text-sm text-primary hover:underline inline-flex items-center gap-1"
            >
              <LinkIcon size={13} /> Open export download link
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
