import React from 'react';
import { useGlobalState } from '../../context/GlobalState';
import { Server, Database as DatabaseIcon, Cpu, Zap, RotateCw } from 'lucide-react';
import clsx from 'clsx';

const StatusBadge = ({ label, value, active, icon: Icon, colorClass }) => (
  <div className="flex items-center gap-2 bg-surfaceHover px-4 py-2 rounded-xl border border-border border-opacity-40 shadow-sm transition-all duration-300">
    <Icon size={16} className={active ? colorClass : "text-textMuted"} />
    <span className="text-xs font-medium text-textMuted">{label}:</span>
    <span className={clsx("text-xs font-semibold truncate max-w-[150px]", active ? "text-textMain" : "text-textMuted")}>
      {value}
    </span>
  </div>
);

const Topbar = () => {
  const { status, model, dataset, gpu } = useGlobalState();

  const getStatusColor = () => {
    switch(status) {
      case 'idle': return 'text-textMuted';
      case 'loading_model':
      case 'dataset_ready': return 'text-accent';
      case 'training': return 'text-warning';
      case 'evaluating': return 'text-primary';
      case 'error': return 'text-error';
      case 'done': return 'text-success';
      default: return 'text-primary';
    }
  };

  const isAnimated = status === 'training' || status === 'evaluating' || status === 'loading_model';

  return (
    <div className="h-16 flex-shrink-0 bg-surface/80 backdrop-blur-xl border-b border-border border-opacity-50 flex items-center justify-between px-8 z-10 sticky top-0 shadow-sm">
      <div className="flex items-center gap-4">
        <StatusBadge 
          label="Model" 
          value={model || 'Not Loaded'} 
          active={!!model}
          icon={Server}
          colorClass="text-accent"
        />
        <StatusBadge 
          label="Dataset" 
          value={dataset ? dataset.size : 'Not Ready'} 
          active={!!dataset}
          icon={DatabaseIcon}
          colorClass="text-primary"
        />
        <div className="flex items-center gap-2 bg-surfaceHover/80 px-4 py-2 rounded-xl border border-border border-opacity-40 shadow-sm">
          {isAnimated ? (
             <RotateCw size={16} className={clsx("animate-spin", getStatusColor())} />
          ) : (
             <Zap size={16} className={getStatusColor()} />
          )}
          <span className="text-xs font-medium text-textMuted">Status:</span>
          <span className={clsx("text-xs font-bold uppercase tracking-wider", getStatusColor())}>
            {status.replace('_', ' ')}
          </span>
        </div>
      </div>
      <div className="flex items-center">
         <StatusBadge 
          label="GPU" 
          value={gpu} 
          active={true}
          icon={Cpu}
          colorClass={status === 'training' ? "text-warning animate-pulse" : "text-primary"}
        />
      </div>
    </div>
  );
};

export default Topbar;
