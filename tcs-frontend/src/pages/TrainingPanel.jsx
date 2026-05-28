import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGlobalState } from '../context/GlobalState';
import StepFlow from '../components/ui/StepFlow';
import { api } from '../api';
import * as Slider from '@radix-ui/react-slider';
import {
  Play,
  RotateCcw,
  Loader2,
  Activity,
  CheckCircle2,
  ArrowRight,
  SlidersHorizontal,
  Sparkles,
  Minus,
  Plus,
} from 'lucide-react';
import {
  LineChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import toast from 'react-hot-toast';

const SliderConfig = ({ label, min, max, step, value, onChange, disabled, tooltip, formatValue = (v) => String(v), precision = 0 }) => {
  const numericValue = Number(value);
  const [draftValue, setDraftValue] = useState(String(numericValue));

  useEffect(() => {
    setDraftValue(String(numericValue));
  }, [numericValue]);

  const clampAndNormalize = (nextValue) => {
    const clamped = Math.max(min, Math.min(max, nextValue));
    const snapped = Math.round(clamped / step) * step;
    return Number(snapped.toFixed(precision));
  };

  const commitTypedValue = () => {
    const parsed = Number(draftValue);
    if (Number.isNaN(parsed)) {
      setDraftValue(String(numericValue));
      return;
    }
    const normalized = clampAndNormalize(parsed);
    onChange(normalized);
    setDraftValue(String(normalized));
  };

  const bumpValue = (delta) => {
    const normalized = clampAndNormalize(numericValue + delta);
    onChange(normalized);
    setDraftValue(String(normalized));
  };

  return (
  <div className="flex flex-col gap-1.5 min-w-0">
    <div className="flex items-center justify-between gap-3">
      <label className="text-xs font-semibold text-textMain">{label}</label>
      <span className="text-[10px] font-mono font-bold bg-background px-2 py-0.5 rounded-lg text-primary border border-border/50">
        {formatValue(numericValue)}
      </span>
    </div>
    <div className="flex items-center gap-2">
      <button
        type="button"
        onClick={() => bumpValue(-step)}
        disabled={disabled || numericValue <= min}
        className="h-9 w-9 shrink-0 rounded-xl border border-border/60 bg-background text-textMain hover:bg-surfaceHover disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center transition-colors"
      >
        <Minus size={14} />
      </button>
      <div className="flex-1 px-1">
        <Slider.Root
          value={[numericValue]}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          onValueChange={(vals) => onChange(vals[0])}
          className="relative flex h-9 w-full touch-none select-none items-center"
        >
          <Slider.Track className="relative h-2.5 grow overflow-hidden rounded-full bg-background border border-border/60">
            <Slider.Range className="absolute h-full rounded-full bg-gradient-to-r from-primary/80 to-accent/80" />
          </Slider.Track>
          <Slider.Thumb className="block h-5 w-5 rounded-full border border-primary/60 bg-white shadow-lg outline-none ring-offset-background transition hover:scale-110 focus-visible:ring-2 focus-visible:ring-primary/50 disabled:pointer-events-none disabled:opacity-50" />
        </Slider.Root>
      </div>
      <button
        type="button"
        onClick={() => bumpValue(step)}
        disabled={disabled || numericValue >= max}
        className="h-9 w-9 shrink-0 rounded-xl border border-border/60 bg-background text-textMain hover:bg-surfaceHover disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center transition-colors"
      >
        <Plus size={14} />
      </button>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={draftValue}
        onChange={(e) => setDraftValue(e.target.value)}
        onBlur={commitTypedValue}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            commitTypedValue();
          }
        }}
        disabled={disabled}
        className="w-24 bg-background border border-border/60 text-textMain text-sm rounded-xl px-3 py-2 text-right font-semibold focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
      />
    </div>
    <p className="text-[11px] leading-relaxed text-textMuted break-words">{tooltip}</p>
  </div>
  );
};

const TrainingTooltip = ({ active, payload, label }) => {
  if (!active || !payload || payload.length === 0) return null;
  const lossEntry = payload.find((p) => p.dataKey === 'loss');
  const accEntry = payload.find((p) => p.dataKey === 'accuracy');
  const current = Number(lossEntry?.value || 0);
  const prev = Number(lossEntry?.payload?.prevLoss ?? current);
  const delta = current - prev;
  const improving = delta <= 0;

  return (
    <div className="rounded-xl border border-border/60 bg-surface/95 backdrop-blur px-3 py-2 shadow-lg">
      <p className="text-[10px] uppercase tracking-[0.2em] font-bold text-textMuted">Step {label}</p>
      <p className="text-sm font-extrabold text-primary mt-1">Loss: {current.toFixed(4)}</p>
      <p className={`text-[11px] mt-1 font-semibold ${improving ? 'text-success' : 'text-warning'}`}>
        {delta >= 0 ? '+' : ''}{delta.toFixed(4)} vs prev
      </p>
      {accEntry?.value != null && (
        <p className="text-sm font-extrabold text-accent mt-1">Accuracy: {(Number(accEntry.value) * 100).toFixed(1)}%</p>
      )}
    </div>
  );
};

export default function TrainingPanel() {
  const navigate = useNavigate();
  const {
    model,
    dataset,
    status,
    setStatus,
    trainConfig: config,
    setTrainConfig: setConfig,
    currentEpoch,
    currentStep,
  } = useGlobalState();
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState([]);
  const [accuracyData, setAccuracyData] = useState([]);

  const [chartMode, setChartMode] = useState('smooth');
  const [windowSize, setWindowSize] = useState(120);
  const [errorMsg, setErrorMsg] = useState(null);
  const [selectedPreset, setSelectedPreset] = useState('balanced');
  const minVisibleSteps = 5;
  const areConfigsEqual = (a, b) => (
    Number(a.lora_rank) === Number(b.lora_rank)
    && Number(a.lora_alpha) === Number(b.lora_alpha)
    && Number(a.lora_dropout) === Number(b.lora_dropout)
    && Number(a.learning_rate) === Number(b.learning_rate)
    && Number(a.epochs) === Number(b.epochs)
    && Number(a.batch_size) === Number(b.batch_size)
  );
  const presets = [
    {
      key: 'conservative',
      label: 'Conservative',
      category: 'Safe',
      values: { lora_rank: 8, lora_alpha: 16, lora_dropout: 0.08, learning_rate: 0.0001, epochs: 3, batch_size: 2 },
    },
    {
      key: 'balanced',
      label: 'Balanced',
      category: 'Recommended',
      values: { lora_rank: 32, lora_alpha: 64, lora_dropout: 0.05, learning_rate: 0.00015, epochs: 5, batch_size: 2 },
    },
    {
      key: 'aggressive',
      label: 'Aggressive',
      category: 'High Capacity',
      values: { lora_rank: 64, lora_alpha: 128, lora_dropout: 0.03, learning_rate: 0.00012, epochs: 7, batch_size: 4 },
    },
  ];
  const customPreset = { key: 'custom', label: 'Custom', category: 'Custom' };
  const presetGroups = useMemo(() => ({
    Safe: presets.filter((p) => p.category === 'Safe'),
    Recommended: presets.filter((p) => p.category === 'Recommended'),
    'High Capacity': presets.filter((p) => p.category === 'High Capacity'),
  }), [presets]);

  const resetToDefault = () => {
    setConfig({
      lora_rank: 16,
      lora_alpha: 32,
      lora_dropout: 0.05,
      learning_rate: 0.0002,
      epochs: 3,
      batch_size: 2,
    });
  };

  const handleStartTraining = async () => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const res = await api.train({ ...config });
      if (res?.success) {
        setStatus('training');
        setData([]);
        toast.success('Training started.');
      }
    } catch {
      setStatus('idle');
      setErrorMsg('Training failed to start.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (!model || !dataset) return;

    let intervalId;
    const poll = async () => {
      try {
        const d = await api.progress();
        if (d?.status === 'error' && d?.error_message) {
          setErrorMsg(d.error_message);
        }
        const losses = d.all_losses ?? d.training_loss ?? [];
        const accuracies = d.all_accuracies ?? d.training_accuracy ?? [];
        setData(
          losses.map((loss, idx) => ({
            step: idx + 1,
            loss,
            prevLoss: idx > 0 ? losses[idx - 1] : loss,
          })),
        );
        setAccuracyData(accuracies);
      } catch {
        // Shared API toast handles backend connectivity failures.
      }
    };

    // Always fetch once so tab switches don't wipe the visible curve.
    poll();

    if (status === 'training' || status === 'finished' || status === 'done') {
      intervalId = setInterval(poll, 1800);
    }

    if (status === 'error') {
      clearInterval(intervalId);
    }

    return () => clearInterval(intervalId);
  }, [status, model, dataset]);

  const isFormDisabled = status === 'training' || status === 'loading_model' || !model || !dataset;
  const maxEpochs = Number(config?.epochs || 3);
  const progressPercent = maxEpochs > 0
    ? Math.min(100, Math.round(((Number(currentEpoch) || 0) / maxEpochs) * 100))
    : 0;
  const latestLoss = data.length > 0 ? data[data.length - 1].loss : 0;
  const bestLoss = data.length > 0 ? Math.min(...data.map((d) => Number(d.loss || 0))) : 0;
  const latestAccuracy = accuracyData.length > 0 ? accuracyData[accuracyData.length - 1] : null;
  const maxVisibleSteps = Math.max(minVisibleSteps, data.length || minVisibleSteps);
  const effectiveWindowSize = Math.min(windowSize, maxVisibleSteps);
  const visibleData = useMemo(() => {
    const base = data.length <= effectiveWindowSize ? data : data.slice(data.length - effectiveWindowSize);
    // Merge accuracy data into chart data points
    // Accuracy is logged less frequently (per eval step), so we interpolate
    if (accuracyData.length === 0) return base;
    return base.map((point) => {
      // Find the closest accuracy measurement for this step
      // accuracy measurements are sparse, so we use the latest one at or before this step's index
      const stepIdx = point.step - 1;
      const totalSteps = data.length;
      const accIdx = Math.min(
        accuracyData.length - 1,
        Math.floor((stepIdx / Math.max(1, totalSteps - 1)) * accuracyData.length)
      );
      return {
        ...point,
        accuracy: accIdx >= 0 ? accuracyData[accIdx] : null,
      };
    });
  }, [data, effectiveWindowSize, accuracyData]);
  const isReady = !!model && !!dataset;

  useEffect(() => {
    if (windowSize > maxVisibleSteps) {
      setWindowSize(maxVisibleSteps);
    }
  }, [windowSize, maxVisibleSteps]);

  useEffect(() => {
    const matchedPreset = presets.find((preset) => areConfigsEqual(config, preset.values));
    setSelectedPreset(matchedPreset ? matchedPreset.key : customPreset.key);
  }, [config]);

  const metrics = useMemo(() => [
    { label: 'Current Loss', value: latestLoss ? latestLoss.toFixed(4) : '0.0000' },
    { label: 'Accuracy', value: latestAccuracy != null ? `${(latestAccuracy * 100).toFixed(1)}%` : '—', accent: true },
    { label: 'Epoch', value: `${currentEpoch || 0} / ${maxEpochs}` },
    { label: 'Total Steps', value: `${currentStep || 0}` },
  ], [latestLoss, latestAccuracy, currentEpoch, maxEpochs, currentStep]);

  return (
    <div className="max-w-[1650px] mx-auto h-full min-h-0 overflow-x-hidden flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-4 duration-500 px-2">
      <StepFlow />

      <div className="flex items-center justify-between gap-3 shrink-0">
        <div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-textMain tracking-tight">Training Configuration</h1>
          <p className="text-sm text-textMuted mt-1">Tune LoRA hyperparameters and monitor loss in real-time.</p>
        </div>
        <div className="hidden md:flex items-center gap-2 rounded-xl border border-border/50 bg-surface px-3 py-2 text-xs text-textMuted">
          <Sparkles size={13} className="text-primary" /> LoRA fine-tuning
        </div>
      </div>

      {!isReady && (
        <div className="rounded-2xl border border-warning/20 bg-warning/5 px-4 py-3 text-warning flex items-start gap-3 shadow-sm shrink-0">
          <Activity size={17} className="mt-0.5 shrink-0" />
          <div className="text-sm">
            Load a model and upload a dataset before starting training.
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-[390px_minmax(0,1fr)] gap-4 flex-1 min-h-0">
        <div className="bg-surface border border-border/50 rounded-3xl p-5 shadow-xl flex flex-col min-h-0 overflow-hidden">
          <div className="flex items-center justify-between gap-3 mb-4">
            <div>
              <h3 className="text-xs font-bold text-textMuted tracking-[0.22em] uppercase flex items-center gap-2">
                <SlidersHorizontal size={13} className="text-primary" /> Hyperparameters
              </h3>
            </div>
            <button
              onClick={resetToDefault}
              disabled={isFormDisabled}
              className="text-textMuted hover:text-textMain disabled:opacity-50 transition-colors flex items-center gap-1 text-[10px] font-bold uppercase tracking-[0.24em] bg-surfaceHover px-3 py-2 rounded-xl border border-border/50"
            >
              <RotateCcw size={10} /> Reset
            </button>
          </div>

          <div className="flex-1 min-h-0 flex flex-col gap-4 overflow-y-auto overflow-x-hidden pr-2 pb-2">
            <div className="flex flex-col gap-1.5 min-w-0">
              <label className="text-xs font-semibold text-textMain">Preset</label>
              <select
                value={selectedPreset}
                disabled={isFormDisabled}
                onChange={(e) => {
                  const key = e.target.value;
                  setSelectedPreset(key);
                  const preset = presets.find((p) => p.key === key);
                  if (preset) setConfig({ ...config, ...preset.values });
                }}
                className="w-full bg-background border border-border/50 text-textMain text-sm rounded-2xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-primary/50 appearance-none disabled:opacity-50"
              >
                {Object.entries(presetGroups).map(([group, items]) => (
                  <optgroup key={group} label={group}>
                    {items.map((preset) => (
                      <option key={preset.key} value={preset.key}>{preset.label}</option>
                    ))}
                  </optgroup>
                ))}
                <optgroup label={customPreset.category}>
                  <option value={customPreset.key}>{customPreset.label}</option>
                </optgroup>
              </select>
            </div>

            <SliderConfig
              label="LoRA Rank (r)"
              min={4}
              max={128}
              step={4}
              value={config.lora_rank}
              onChange={(v) => setConfig({ ...config, lora_rank: v })}
              disabled={isFormDisabled}
              precision={0}
              tooltip="Defines adapter capacity. Higher rank gives the model more freedom to learn complex patterns, but increases VRAM usage, training time, and overfitting risk on small datasets."
            />
            <SliderConfig
              label="LoRA Alpha"
              min={8}
              max={256}
              step={8}
              value={config.lora_alpha}
              onChange={(v) => setConfig({ ...config, lora_alpha: v })}
              disabled={isFormDisabled}
              precision={0}
              tooltip="Controls how strongly LoRA updates influence the base model. Higher alpha can improve adaptation speed, but can also over-amplify noise and hurt generalization if set too high."
            />
            <SliderConfig
              label="LoRA Dropout"
              min={0}
              max={0.5}
              step={0.01}
              value={config.lora_dropout}
              onChange={(v) => setConfig({ ...config, lora_dropout: v })}
              disabled={isFormDisabled}
              precision={2}
              formatValue={(v) => Number(v).toFixed(2)}
              tooltip="Regularizes adapter updates to reduce overfitting. Higher dropout can improve robustness on small datasets, but too much will slow convergence and may underfit."
            />

            <div className="h-px bg-border/40" />

            <div className="flex flex-col gap-1.5 min-w-0">
              <label className="text-xs font-semibold text-textMain">Learning Rate</label>
              <input
                type="number"
                step="0.00001"
                value={config.learning_rate}
                onChange={(e) => setConfig({ ...config, learning_rate: Number(e.target.value) })}
                disabled={isFormDisabled}
                className="w-full bg-background border border-border/50 text-textMain text-sm rounded-2xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
              />
              <p className="text-[11px] leading-relaxed text-textMuted">
                Step size of parameter updates. Too high can cause unstable loss spikes; too low can make training very slow or get stuck in weak minima.
              </p>
            </div>

            <SliderConfig
              label="Epochs"
              min={1}
              max={25}
              step={1}
              value={config.epochs}
              onChange={(v) => setConfig({ ...config, epochs: v })}
              disabled={isFormDisabled}
              precision={0}
              tooltip="How many complete passes over the training data are performed. More epochs can improve fit, but beyond a point they increase overfitting risk and training time."
            />

            <div className="flex flex-col gap-1.5 min-w-0">
              <label className="text-xs font-semibold text-textMain">Batch Size</label>
              <select
                value={config.batch_size}
                onChange={(e) => setConfig({ ...config, batch_size: Number(e.target.value) })}
                disabled={isFormDisabled}
                className="w-full bg-background border border-border/50 text-textMain text-sm rounded-2xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-primary/50 appearance-none disabled:opacity-50"
              >
                <option value={1}>1 (Micro)</option>
                <option value={2}>2 (Standard)</option>
                <option value={4}>4 (Large)</option>
                <option value={8}>8 (High VRAM)</option>
              </select>
              <p className="text-[11px] leading-relaxed text-textMuted">
                Number of samples processed per optimizer step. Larger batches improve throughput and gradient stability but require more VRAM and can reduce generalization in some cases.
              </p>
            </div>
          </div>

          <div className="mt-3 pt-4 border-t border-border/30 shrink-0 bg-surface flex flex-col gap-2">
            <button
              onClick={handleStartTraining}
              disabled={isFormDisabled}
              className="w-full py-3 bg-success hover:bg-[#059669] text-white font-semibold rounded-xl transition-all disabled:opacity-50 disabled:bg-surfaceHover disabled:text-textMuted flex items-center justify-center gap-2"
            >
              {(status === 'training' || isLoading)
                ? <Loader2 size={16} className="animate-spin" />
                : <Play size={16} className="fill-current" />}
              {(status === 'training' || isLoading) ? 'Training...' : 'Start Training'}
            </button>
            {status === 'finished' && (
              <button
                onClick={() => navigate('/evaluation')}
                className="w-full mt-2 py-2.5 bg-success/10 hover:bg-success/15 text-success font-semibold border border-success/20 rounded-xl transition-all flex items-center justify-center gap-2 text-sm"
              >
                Move to Evaluation <ArrowRight size={15} />
              </button>
            )}
          </div>
        </div>

        <div className="flex flex-col gap-4 min-h-0">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 shrink-0">
            {metrics.map((metric) => (
              <div key={metric.label} className="bg-surface border border-border/50 p-4 rounded-2xl shadow-sm">
                <span className="text-[10px] font-bold text-textMuted uppercase tracking-[0.24em]">{metric.label}</span>
                <div className={`mt-2 text-2xl font-extrabold ${metric.accent ? 'text-accent' : 'text-textMain'}`}>{metric.value}</div>
              </div>
            ))}
          </div>

          <div className="bg-surface border border-border/50 rounded-3xl p-5 shadow-xl flex flex-col min-h-0 flex-1">
            <div className="flex items-center justify-between gap-3 mb-3 shrink-0">
              <div>
                <h3 className="text-sm font-bold text-textMain">Live Loss & Accuracy</h3>
                <p className="text-xs text-textMuted mt-1">Loss (blue, left axis) · Accuracy (teal, right axis)</p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setChartMode((m) => (m === 'smooth' ? 'linear' : 'smooth'))}
                  className="px-3 py-1.5 rounded-lg border border-border/50 bg-background text-[10px] font-bold uppercase tracking-[0.2em] text-textMuted hover:text-textMain"
                >
                  {chartMode === 'smooth' ? 'Smooth' : 'Raw'}
                </button>
              {status === 'training' && (
                <span className="flex items-center gap-2 text-[10px] font-bold text-warning bg-warning/10 px-3 py-1 rounded-full border border-warning/20 uppercase tracking-[0.2em]">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-warning opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-warning" />
                  </span>
                  Training
                </span>
              )}
              {status === 'finished' && (
                <span className="flex items-center gap-1.5 text-[10px] font-bold text-success bg-success/10 px-3 py-1 rounded-full border border-success/20 uppercase tracking-[0.2em]">
                  <CheckCircle2 size={11} /> Complete
                </span>
              )}
              {status === 'error' && (
                <span className="text-[11px] font-semibold text-error">{errorMsg || 'Training failed'}</span>
              )}
              </div>
            </div>

            <div className="w-full bg-background border border-border/50 rounded-full h-2 mb-3 shrink-0 overflow-hidden">
              <div
                className="bg-gradient-to-r from-success via-primary to-accent h-full transition-all duration-700 ease-out"
                style={{ width: `${progressPercent}%` }}
              />
            </div>

            <div className="flex-1 min-h-0 w-full bg-background/50 rounded-2xl p-3 border border-border/30">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={visibleData} margin={{ top: 8, right: 40, bottom: 4, left: -12 }}>
                  <defs>
                    <linearGradient id="lossFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#3B82F6" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#3B82F6" stopOpacity={0.02} />
                    </linearGradient>
                    <linearGradient id="accFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#06B6D4" stopOpacity={0.2} />
                      <stop offset="100%" stopColor="#06B6D4" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2A2E39" vertical={false} />
                  <XAxis dataKey="step" stroke="#94A3B8" fontSize={10} tickLine={false} axisLine={false} />
                  <YAxis yAxisId="loss" stroke="#94A3B8" fontSize={10} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                  <YAxis yAxisId="accuracy" orientation="right" stroke="#06B6D4" fontSize={10} tickLine={false} axisLine={false} domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <RechartsTooltip
                    content={<TrainingTooltip />}
                    cursor={{ stroke: '#334155', strokeWidth: 1, strokeDasharray: '5 5' }}
                  />
                  {bestLoss > 0 && (
                    <ReferenceLine
                      yAxisId="loss"
                      y={bestLoss}
                      stroke="#10B981"
                      strokeDasharray="4 4"
                      ifOverflow="extendDomain"
                      label={{ value: `best ${bestLoss.toFixed(4)}`, fill: '#10B981', fontSize: 10 }}
                    />
                  )}
                  <Area
                    yAxisId="loss"
                    type={chartMode === 'smooth' ? 'monotone' : 'linear'}
                    dataKey="loss"
                    stroke="none"
                    fill="url(#lossFill)"
                    isAnimationActive={false}
                  />
                  <Line
                    yAxisId="loss"
                    type={chartMode === 'smooth' ? 'monotone' : 'linear'}
                    dataKey="loss"
                    stroke="#3B82F6"
                    strokeWidth={3}
                    dot={false}
                    activeDot={{ r: 4, fill: '#10B981', stroke: '#1E212B', strokeWidth: 2 }}
                    isAnimationActive={false}
                  />
                  {accuracyData.length > 0 && (
                    <>
                      <Area
                        yAxisId="accuracy"
                        type={chartMode === 'smooth' ? 'monotone' : 'linear'}
                        dataKey="accuracy"
                        stroke="none"
                        fill="url(#accFill)"
                        isAnimationActive={false}
                        connectNulls
                      />
                      <Line
                        yAxisId="accuracy"
                        type={chartMode === 'smooth' ? 'monotone' : 'linear'}
                        dataKey="accuracy"
                        stroke="#06B6D4"
                        strokeWidth={2}
                        strokeDasharray="6 3"
                        dot={false}
                        activeDot={{ r: 4, fill: '#06B6D4', stroke: '#1E212B', strokeWidth: 2 }}
                        isAnimationActive={false}
                        connectNulls
                      />
                    </>
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-3 border border-border/30 bg-background/60 rounded-xl px-3 py-2 shrink-0">
              <div className="flex items-center justify-between gap-3 mb-1">
                <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-textMuted">Visible Steps</span>
                <span className="text-[11px] font-semibold text-textMain">
                  Last {Math.min(effectiveWindowSize, data.length)} / {data.length || 0}
                </span>
              </div>
              <input
                type="range"
                min={minVisibleSteps}
                max={maxVisibleSteps}
                step={1}
                value={effectiveWindowSize}
                onChange={(e) => setWindowSize(Number(e.target.value))}
                disabled={data.length <= minVisibleSteps}
                className="w-full h-1.5 bg-background border border-border/50 rounded-lg appearance-none cursor-pointer accent-primary disabled:opacity-40 disabled:cursor-not-allowed"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
