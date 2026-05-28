import React, { useMemo, useRef, useState } from 'react';
import StepFlow from '../components/ui/StepFlow';
import { api } from '../api';
import { useGlobalState } from '../context/GlobalState';
import { BarChart2, ChevronDown, ChevronUp, CheckCircle2, Play, Settings2, Minus, Plus, UploadCloud, Loader2 } from 'lucide-react';
import { Link } from 'react-router-dom';
import toast from 'react-hot-toast';

export default function Evaluation() {
  const { setEvalResults, evalResults } = useGlobalState();
  const [isLoading, setIsLoading] = useState(false);
  const [expanded, setExpanded] = useState(true);
  const [numSamples, setNumSamples] = useState(10);
  // Eval dataset upload state (moved from DatasetUpload)
  const [evalFile, setEvalFile] = useState(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalResult, setEvalResult] = useState(null);
  const evalFileInputRef = useRef(null);

  const metrics = evalResults;
  const hasRun = !!metrics;
  const sanitizedSampleCount = Math.max(1, Math.min(100, Number(numSamples) || 10));

  const handleEvalUpload = async () => {
    if (!evalFile || evalLoading) return;
    setEvalLoading(true);
    try {
      const res = await api.uploadEvalDataset(evalFile);
      if (res?.success) {
        setEvalResult({ size: res.size, preview: res.preview });
        toast.success(`Eval dataset loaded: ${res.size} entries.`);
      }
    } catch {
      // API layer toast handles failure.
    } finally {
      setEvalLoading(false);
    }
  };

  const handleRunEvaluation = async () => {
    if (sanitizedSampleCount !== numSamples) {
      setNumSamples(sanitizedSampleCount);
    }
    setIsLoading(true);
    try {
      const res = await api.evaluate(sanitizedSampleCount);
      setEvalResults(res);
      setExpanded(true);
      toast.success('Evaluation completed.');
    } catch {
      // API layer shows toast.
    } finally {
      setIsLoading(false);
    }
  };

  const rouge = metrics?.rouge_l || 0;
  const bertScoreF1 = (metrics?.bert_score_f1 || 0) / 100;
  const samples = useMemo(() => metrics?.samples || [], [metrics]);
  const reasoningPresentCount = Number(metrics?.reasoning_present_count || 0);
  const reasoningRetryCount = Number(metrics?.reasoning_retry_count || 0);
  const resolveReasoning = (sample) => {
    const candidates = [
      sample?.model_reasoning,
      sample?.reasoning,
      sample?.modelReasoning,
      sample?.cot,
    ];
    const value = candidates.find((v) => typeof v === 'string' && v.trim());
    return value?.trim() || '';
  };

  return (
    <div className="max-w-[1650px] mx-auto min-h-0 overflow-x-hidden flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-4 duration-500 px-2">
      <StepFlow />

      <div className="flex items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-textMain tracking-tight">Model Evaluation</h1>
          <p className="text-sm text-textMuted mt-1">Results are persisted so you can revisit this page any time.</p>
        </div>
        {hasRun && (
          <Link to="/export" className="px-4 py-2.5 bg-surfaceHover border border-border hover:bg-primary/20 hover:text-primary hover:border-primary/50 text-textMain rounded-xl transition-all font-medium text-sm">
            Proceed to Export
          </Link>
        )}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[360px_minmax(0,1fr)] gap-4 flex-1 min-h-0">
        <div className="bg-surface border border-border/50 p-5 rounded-3xl shadow-xl flex flex-col gap-4">
          <div className="p-3 bg-primary/10 text-primary rounded-2xl border border-primary/20 w-max">
            <Settings2 size={28} />
          </div>
          <h2 className="text-xl font-bold text-textMain">Run Evaluation</h2>
          <p className="text-sm text-textMuted">
            ROUGE-L compares sequence overlap quality. BERTScore measures semantic similarity.
          </p>

          {/* ── Eval Dataset Upload ── */}
          <div className="flex flex-col gap-2 p-3 bg-background/60 border border-border/40 rounded-2xl">
            <div className="text-[10px] uppercase tracking-[0.22em] font-bold text-textMuted flex items-center gap-2">
              <UploadCloud size={11} className="text-warning" /> Evaluation Dataset
            </div>
            <p className="text-[11px] text-textMuted leading-relaxed">
              Upload a JSON file for evaluation. This replaces <span className="font-semibold text-textMain">only</span> the test split.
            </p>
            <div
              className="w-full border border-dashed border-warning/40 rounded-xl px-3 py-3 flex flex-col items-center gap-2 cursor-pointer hover:border-warning/70 transition-all bg-background/40"
              onClick={() => evalFileInputRef.current?.click()}
            >
              <input
                ref={evalFileInputRef}
                type="file"
                className="hidden"
                accept=".json,application/json"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) setEvalFile(f);
                }}
              />
              <UploadCloud size={18} className="text-warning" />
              <span className="text-xs text-textMuted">
                {evalFile ? evalFile.name : 'Click to select eval JSON'}
              </span>
            </div>
            {evalResult && (
              <div className="text-xs text-success font-semibold">
                ✓ Eval dataset loaded: {evalResult.size} entries
              </div>
            )}
            <button
              onClick={handleEvalUpload}
              disabled={!evalFile || evalLoading}
              className="w-full px-4 py-2 rounded-xl bg-warning/20 hover:bg-warning/30 border border-warning/30 text-warning font-semibold text-sm transition-all flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {evalLoading ? <Loader2 size={14} className="animate-spin" /> : <UploadCloud size={14} />}
              {evalLoading ? 'Loading eval...' : 'Upload Eval Dataset'}
            </button>
          </div>

          <div className="flex flex-col gap-1 w-full bg-background border border-border/50 rounded-xl px-4 py-3">
            <label className="text-[10px] font-bold text-textMuted uppercase tracking-[0.2em]">Evaluation Samples</label>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setNumSamples((v) => Math.max(1, Number(v || 10) - 1))}
                className="h-8 w-8 rounded-lg border border-border/60 bg-surfaceHover text-textMain hover:bg-background transition-colors flex items-center justify-center"
              >
                <Minus size={13} />
              </button>
              <input
                type="number"
                min="1"
                max="100"
                value={numSamples}
                onChange={(e) => setNumSamples(parseInt(e.target.value, 10) || 10)}
                onBlur={() => setNumSamples(sanitizedSampleCount)}
                className="w-full bg-transparent text-textMain font-bold focus:outline-none"
              />
              <button
                type="button"
                onClick={() => setNumSamples((v) => Math.min(100, Number(v || 10) + 1))}
                className="h-8 w-8 rounded-lg border border-border/60 bg-surfaceHover text-textMain hover:bg-background transition-colors flex items-center justify-center"
              >
                <Plus size={13} />
              </button>
            </div>
          </div>
          <button
            onClick={handleRunEvaluation}
            disabled={isLoading}
            className="w-full px-5 py-3 bg-primary hover:bg-primaryHover disabled:bg-surfaceHover disabled:text-textMuted text-white font-semibold rounded-xl transition-all flex items-center justify-center gap-2"
          >
            {isLoading ? <BarChart2 size={17} className="animate-pulse" /> : <Play size={17} fill="currentColor" />}
            {isLoading ? 'Evaluating...' : 'Run Evaluation'}
          </button>
          {hasRun && (
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-success/10 border border-success/20 text-success rounded-full text-xs font-bold w-max">
              <CheckCircle2 size={13} /> Persisted
            </div>
          )}
        </div>

        <div className="bg-surface border border-border/50 rounded-3xl shadow-xl p-5 flex flex-col min-h-0">
          {!hasRun ? (
            <div className="flex-1 flex items-center justify-center text-sm text-textMuted">
              Run evaluation to see metrics and outputs.
            </div>
          ) : (
            <div className="flex flex-col flex-1 min-h-0">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div className="bg-background border border-border/50 p-4 rounded-2xl">
                  <div className="text-[10px] font-bold uppercase tracking-[0.22em] text-textMuted mb-2">ROUGE-L</div>
                  <div className="text-3xl font-extrabold text-primary">{(rouge * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-background border border-border/50 p-4 rounded-2xl">
                  <div className="text-[10px] font-bold uppercase tracking-[0.22em] text-textMuted mb-2">BERTScore F1</div>
                  <div className="text-3xl font-extrabold text-accent">{(bertScoreF1 * 100).toFixed(1)}%</div>
                </div>
              </div>
              <div className="mb-3 text-xs text-textMuted">
                Backend reasoning received for {reasoningPresentCount} / {samples.length} samples.
              </div>
              {reasoningRetryCount > 0 && (
                <div className="mb-3 text-xs text-textMuted">
                  Format-enforced retry used for {reasoningRetryCount} sample{reasoningRetryCount === 1 ? '' : 's'}.
                </div>
              )}

              <div className="border border-border/40 rounded-2xl overflow-hidden flex flex-col flex-1 min-h-0">
                <button
                  onClick={() => setExpanded((v) => !v)}
                  className="w-full flex items-center justify-between p-4 hover:bg-surfaceHover transition-colors"
                >
                  <h3 className="text-sm font-bold text-textMain">Sample Outputs (with Reasoning)</h3>
                  {expanded ? <ChevronUp size={18} className="text-textMuted" /> : <ChevronDown size={18} className="text-textMuted" />}
                </button>
                {expanded && (
                  <div className="border-t border-border/30 p-4 pb-24 overflow-y-auto flex-1 min-h-0">
                    <div className="flex flex-col gap-4">
                      {samples.map((sample, idx) => (
                        <div key={idx} className="bg-background border border-border/40 rounded-2xl p-4">
                          <div className="mb-3">
                            <span className="text-[10px] font-bold text-textMuted uppercase tracking-wider px-2 py-1 rounded inline-block mb-1 border border-border">Question</span>
                            <p className="text-sm font-semibold text-textMain leading-relaxed">{sample.question}</p>
                          </div>
                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                            <div>
                              <span className="text-[10px] font-bold text-success uppercase bg-success/10 px-2 py-1 rounded inline-block mb-2 border border-success/20">Expected</span>
                              <p className="text-sm text-textMain leading-relaxed">{sample.expected}</p>
                            </div>
                            <div>
                              <span className="text-[10px] font-bold border border-primary/20 text-primary uppercase bg-primary/10 px-2 py-1 rounded inline-block mb-2">Predicted</span>
                              <p className="text-sm text-textMain leading-relaxed">{sample.model_output}</p>
                            </div>
                          </div>
                          <div className="mt-3 pt-3 border-t border-border/20">
                            <span className="text-[10px] font-bold border border-warning/20 text-warning uppercase bg-warning/10 px-2 py-1 rounded inline-block mb-2">Model Reasoning</span>
                            <p className="text-sm text-textMuted leading-relaxed italic">
                              {resolveReasoning(sample) || 'No explicit reasoning was generated for this sample.'}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
