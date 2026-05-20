import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Settings, Database, Activity, BarChart2, Download, MessageSquareCode } from 'lucide-react';
import clsx from 'clsx';
import { useGlobalState } from '../../context/GlobalState';

const navItems = [
  { name: 'Dashboard', path: '/', icon: LayoutDashboard, requires: [] },
  { name: 'Model Setup', path: '/model-setup', icon: Settings, requires: [] },
  { name: 'Dataset', path: '/dataset', icon: Database, requires: ['model_loaded'] },
  { name: 'Training', path: '/training', icon: Activity, requires: ['dataset_ready'] },
  { name: 'Evaluation', path: '/evaluation', icon: BarChart2, requires: ['training_finished'] },
  { name: 'Export', path: '/export', icon: Download, requires: ['training_finished'] },
  { name: 'Chat Playground', path: '/chat', icon: MessageSquareCode, requires: ['training_finished'] },
];

const Sidebar = () => {
  const { model, dataset, status } = useGlobalState();
  const isModelLoaded = !!model;
  const isDatasetReady = !!dataset;
  const isTrainingFinished = status === 'finished' || status === 'done';

  const isStepEnabled = (item) => {
    if (item.requires.includes('model_loaded') && !isModelLoaded) return false;
    if (item.requires.includes('dataset_ready') && !isDatasetReady) return false;
    if (item.requires.includes('training_finished') && !isTrainingFinished) return false;
    return true;
  };

  return (
    <div className="w-64 flex-shrink-0 bg-surface border-r border-border border-opacity-50 hidden md:flex flex-col shadow-xl z-20">
      <div className="h-16 flex items-center px-6 border-b border-border border-opacity-50 shrink-0">
        <div className="flex items-center gap-3 text-primary">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Activity size={20} className="text-primary animate-pulse" />
          </div>
          <span className="text-lg font-bold text-textMain tracking-tight">TCS FineTuner</span>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto py-6 px-4 flex flex-col gap-1.5">
        <div className="text-xs font-semibold text-textMuted uppercase tracking-wider mb-2 px-2">Pipeline</div>
        {navItems.map((item) => {
          const enabled = isStepEnabled(item);
          return (
            <NavLink
              key={item.path}
              to={enabled ? item.path : '#'}
              onClick={(e) => { if (!enabled) e.preventDefault(); }}
              className={({ isActive }) =>
                clsx(
                  "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 text-sm font-medium outline-none",
                  !enabled && "opacity-50 cursor-not-allowed",
                  isActive && enabled
                    ? "bg-primary bg-opacity-10 text-primary border border-primary/20 shadow-[0_0_10px_rgba(59,130,246,0.1)]" 
                    : (!enabled ? "text-textMuted hover:text-textMuted border border-transparent" : "text-textMuted hover:bg-surfaceHover hover:text-textMain border border-transparent")
                )
              }
            >
              <item.icon size={18} />
              {item.name}
            </NavLink>
          );
        })}
      </div>
    </div>
  );
};

export default Sidebar;
