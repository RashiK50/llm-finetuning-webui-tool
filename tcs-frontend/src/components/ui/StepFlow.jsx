import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import clsx from 'clsx';
import { Server, Database, Activity, BarChart2, MessageSquareCode, Download } from 'lucide-react';
import { useGlobalState } from '../../context/GlobalState';

const steps = [
  { id: 'model', name: 'Model', path: '/model-setup', icon: Server, requires: [] },
  { id: 'dataset', name: 'Dataset', path: '/dataset', icon: Database, requires: ['model_loaded'] },
  { id: 'train', name: 'Train', path: '/training', icon: Activity, requires: ['dataset_ready'] },
  { id: 'evaluate', name: 'Evaluate', path: '/evaluation', icon: BarChart2, requires: ['training_finished'] },
  { id: 'export', name: 'Export', path: '/export', icon: Download, requires: ['training_finished'] },
  { id: 'chat', name: 'Chat', path: '/chat', icon: MessageSquareCode, requires: ['training_finished'] },
];

const StepFlow = () => {
  const location = useLocation();
  const currentIndex = steps.findIndex(s => location.pathname.startsWith(s.path));
  const { model, dataset, status } = useGlobalState();

  const isModelLoaded = !!model;
  const isDatasetReady = !!dataset;
  const isTrainingFinished = status === 'finished' || status === 'done';

  const isStepEnabled = (step) => {
    if (step.requires.includes('model_loaded') && !isModelLoaded) return false;
    if (step.requires.includes('dataset_ready') && !isDatasetReady) return false;
    if (step.requires.includes('training_finished') && !isTrainingFinished) return false;
    return true;
  };

  if (location.pathname === '/') return null;

  return (
    <div className="w-full mb-2 relative px-2 py-2 shrink-0 overflow-x-auto overflow-y-visible min-h-[72px]">
      <div className="min-w-[680px] flex items-center justify-between relative">
        <div className="absolute left-5 right-5 top-7 h-[2px] bg-border/70 -z-10" />
        {steps.map((step, idx) => {
        const isActive = idx === currentIndex;
        const isPast = idx < currentIndex;
        const enabled = isStepEnabled(step);

        return (
          <Link key={step.id} to={enabled ? step.path : '#'} onClick={(e) => { if (!enabled) e.preventDefault(); }} className={clsx(
            "flex flex-col items-center gap-2 bg-background px-1 relative group outline-none focus:outline-none",
            enabled ? "cursor-pointer" : "cursor-not-allowed opacity-50"
          )}>
            <div className={clsx(
              "w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-300 z-10",
              enabled && "group-hover:scale-110",
              isActive ? "bg-primary text-white shadow-[0_0_20px_rgba(59,130,246,0.4)] border border-primary" : 
              isPast ? "bg-primary/20 text-primary border border-primary/30 group-hover:bg-primary/30" : 
              "bg-surface text-textMuted border border-border group-hover:border-textMuted/50 group-hover:text-textMain/80"
            )}>
              <step.icon size={16} className={isActive ? "animate-pulse" : (enabled ? "transition-transform group-hover:scale-110" : "")} />
            </div>
            <span className={clsx(
              "text-[10px] font-bold uppercase tracking-[0.2em] transition-colors duration-300", 
              isActive ? "text-primary" : isPast ? "text-textMain group-hover:text-primary" : "text-textMuted group-hover:text-textMain/80"
            )}>
              {step.name}
            </span>
          </Link>
        );
      })}
      </div>
    </div>
  );
};

export default StepFlow;
