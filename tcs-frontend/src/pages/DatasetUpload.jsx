import React, { useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGlobalState } from '../context/GlobalState';
import StepFlow from '../components/ui/StepFlow';
import { api } from '../api';
import {
  UploadCloud,
  FileJson,
  ArrowRight,
  Edit2,
  Loader2,
  Database,
  FileText,
} from 'lucide-react';
import clsx from 'clsx';
import toast from 'react-hot-toast';

const compactText = (value, limit = 220) => {
  const text = String(value ?? '').trim();
  if (!text) return '—';
  return text.length > limit ? `${text.slice(0, limit).trim()}...` : text;
};

export default function DatasetUpload() {
  const navigate = useNavigate();
  const { dataset, setDataset, status, setStatus } = useGlobalState();
  const [isDragActive, setIsDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef(null);


  const previewItems = useMemo(() => (dataset?.preview || []).slice(0, 5), [dataset]);
  const hasDataset = previewItems.length > 0;

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setIsDragActive(true);
    if (e.type === 'dragleave') setIsDragActive(false);
  };

  const handleFileSelected = (selected) => {
    if (!selected) return;
    if (selected.type !== 'application/json' && !selected.name.toLowerCase().endsWith('.json')) {
      toast.error('Please upload a valid JSON file.');
      return;
    }
    setFile(selected);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    const dropped = e.dataTransfer?.files?.[0];
    handleFileSelected(dropped);
  };

  const handleUpload = async () => {    if (!file || isLoading) return;
    setIsLoading(true);
    try {
      const res = await api.uploadDataset(file);
      if (res?.success) {
        setDataset({
          name: file.name,
          size: res.size,
          preview: Array.isArray(res.preview) ? res.preview.slice(0, 5) : [],
        });
        if (status === 'idle' || status === 'model_ready') setStatus('dataset_ready');
        toast.success('Dataset processed successfully.');
      }
    } catch {
      // Toast is handled by api.request.
    } finally {
      setIsLoading(false);
    }
  };

  const handleEditDataset = async () => {
    try {
      await api.clearDataset();
      setDataset(null);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      setStatus('model_ready');
      toast.success('Dataset cleared. Upload a new one.');
    } catch {
      // API layer toast handles failure.
    }
  };



  return (
    <div className="max-w-[1600px] mx-auto min-h-0 flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-4 duration-500 px-2">
      <StepFlow />

      <div className="flex items-center justify-between gap-3 shrink-0">
        <div className="min-w-0">
          <h1 className="text-2xl md:text-3xl font-extrabold text-textMain tracking-tight">Dataset Upload</h1>
          <p className="text-sm text-textMuted mt-1">Upload JSON and verify the first 5 rows before training.</p>
        </div>
        {hasDataset && (
          <div className="shrink-0 flex items-center gap-2">
            <button
              onClick={handleEditDataset}
              className="px-4 py-2.5 rounded-xl bg-surfaceHover border border-border text-textMain text-sm font-semibold hover:bg-background transition-all flex items-center gap-2"
            >
              <Edit2 size={14} /> Edit Dataset
            </button>
            <button
              onClick={() => navigate('/training')}
              className="px-4 py-2.5 rounded-xl bg-success text-white text-sm font-semibold hover:bg-[#059669] transition-all flex items-center gap-2"
            >
              Move to Training <ArrowRight size={15} />
            </button>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[390px_minmax(0,1fr)] gap-4 flex-1 min-h-0">
        <div className="bg-surface border border-border/50 rounded-3xl p-5 shadow-xl flex flex-col gap-4 min-h-0">
          <button
            type="button"
            className={clsx(
              'w-full border-2 border-dashed rounded-2xl p-6 flex flex-col items-center justify-center gap-3 transition-all min-h-[240px]',
              isDragActive
                ? 'border-primary bg-primary/10'
                : 'border-border/60 bg-background/50 hover:border-textMuted/60'
            )}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => {
              if (fileInputRef.current) fileInputRef.current.value = '';
              fileInputRef.current?.click();
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".json,application/json"
              onChange={(e) => handleFileSelected(e.target.files?.[0])}
            />
            <div className="p-4 rounded-2xl bg-primary/10 border border-primary/20 text-primary">
              <UploadCloud size={30} />
            </div>
            <div className="text-center">
              <p className="text-sm font-semibold text-textMain">Drop JSON here or click to browse</p>
              <p className="text-xs text-textMuted mt-1">Expected keys: instruction, reasoning, output</p>
            </div>
            {file && (
              <div className="mt-2 w-full flex items-center gap-2 bg-surface border border-border/50 px-3 py-2 rounded-xl text-sm">
                <FileJson size={15} className="text-primary shrink-0" />
                <span className="truncate">{file.name}</span>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setFile(null);
                  }}
                  className="ml-auto text-textMuted hover:text-error"
                >
                  &times;
                </button>
              </div>
            )}
          </button>

          <div className="rounded-2xl bg-background/70 border border-border/50 px-4 py-3">
            <div className="text-[10px] uppercase tracking-[0.22em] font-bold text-textMuted">Dataset Rows</div>
            <div className="mt-1 text-xl font-bold text-textMain">{dataset?.size || 0}</div>
          </div>

          <div className="mt-auto flex items-center gap-3">
            <button
              onClick={handleUpload}
              disabled={!file || isLoading || status === 'training'}
              className="flex-1 px-4 py-3 rounded-xl bg-primary hover:bg-primaryHover disabled:bg-surfaceHover disabled:text-textMuted text-white font-semibold text-sm transition-all flex items-center justify-center gap-2"
            >
              {isLoading ? <Loader2 size={16} className="animate-spin" /> : <UploadCloud size={16} />}
              {isLoading ? 'Processing...' : 'Upload Dataset'}
            </button>
            {hasDataset && (
              <button
                onClick={() => navigate('/training')}
                className="px-4 py-3 rounded-xl border border-success/25 text-success bg-success/10 hover:bg-success/15 transition-all text-sm font-semibold flex items-center gap-2"
              >
                <ArrowRight size={14} />
              </button>
            )}
          </div>


        </div>

        <div className="bg-surface border border-border/50 rounded-3xl p-5 shadow-xl flex flex-col min-h-0">
          <div className="flex items-center justify-between gap-3 mb-4 shrink-0">
            <div>
              <h2 className="text-xs uppercase tracking-[0.24em] font-bold text-textMuted flex items-center gap-2">
                <CheckCircle2 size={13} className="text-success" /> Top 5 Preview
              </h2>
              <p className="text-xs text-textMuted mt-1">Compact view for the first five rows.</p>
            </div>
            <span className="text-xs text-textMuted flex items-center gap-2">
              <FileText size={13} /> {dataset?.name || 'No dataset loaded'}
            </span>
          </div>

          {hasDataset ? (
            <div className="flex-1 min-h-0 overflow-auto rounded-2xl border border-border/50 bg-background/50">
              <table className="w-full table-fixed border-collapse text-left">
                <thead className="sticky top-0 z-10 bg-surface">
                  <tr className="border-b border-border/50">
                    <th className="w-14 px-3 py-2 text-[10px] uppercase tracking-[0.2em] text-textMuted">#</th>
                    <th className="px-3 py-2 text-[10px] uppercase tracking-[0.2em] text-textMuted">Instruction</th>
                    <th className="px-3 py-2 text-[10px] uppercase tracking-[0.2em] text-textMuted">Reasoning</th>
                    <th className="px-3 py-2 text-[10px] uppercase tracking-[0.2em] text-textMuted">Output</th>
                  </tr>
                </thead>
                <tbody>
                  {previewItems.map((item, idx) => (
                    <tr key={`${idx}-${item.instruction || ''}`} className="border-b border-border/30 align-top">
                      <td className="px-3 py-3 text-xs text-textMuted font-semibold">{idx + 1}</td>
                      <td className="px-3 py-3 text-xs text-textMain leading-5 break-words">{compactText(item.instruction)}</td>
                      <td className="px-3 py-3 text-xs text-textMain leading-5 break-words">{compactText(item.reasoning)}</td>
                      <td className="px-3 py-3 text-xs text-textMain leading-5 break-words">{compactText(item.output)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex-1 min-h-0 rounded-2xl border border-dashed border-border/50 bg-background/30 flex items-center justify-center text-sm text-textMuted">
              Upload a dataset to render the top 5 preview rows.
            </div>
          )}

          <div className="mt-3 pt-3 border-t border-border/30 text-xs text-textMuted shrink-0 flex items-center justify-between">
            <span className="flex items-center gap-2"><Database size={13} /> Rendered rows: {previewItems.length}</span>
            <span>Total loaded: {dataset?.size || 0}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
