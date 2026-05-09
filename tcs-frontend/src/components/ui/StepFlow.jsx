import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import clsx from 'clsx';
import { Server, Database, Activity, BarChart2, MessageSquareCode, Download } from 'lucide-react';

const steps = [
  { id: 'model', name: 'Model', path: '/model-setup', icon: Server },
  { id: 'dataset', name: 'Dataset', path: '/dataset', icon: Database },
  { id: 'train', name: 'Train', path: '/training', icon: Activity },
  { id: 'evaluate', name: 'Evaluate', path: '/evaluation', icon: BarChart2 },
  { id: 'export', name: 'Export', path: '/export', icon: Download },
  { id: 'chat', name: 'Chat', path: '/chat', icon: MessageSquareCode },
];

const StepFlow = () => {
  const location = useLocation();
  const currentIndex = steps.findIndex(s => location.pathname.startsWith(s.path));

  if (location.pathname === '/') return null;

  return (
    <div className="w-full flex items-center justify-between mb-8 relative px-4">
      <div className="absolute left-8 right-8 top-1/2 -translate-y-1/2 h-[2px] bg-border -z-10" />
      {steps.map((step, idx) => {
        const isActive = idx === currentIndex;
        const isPast = idx < currentIndex;
        const widthPercent = (idx / (steps.length - 1)) * 100;

        return (
          <Link key={step.id} to={step.path} className="flex flex-col items-center gap-3 bg-background px-2 relative group cursor-pointer outline-none focus:outline-none">
            <div className={clsx(
              "w-12 h-12 rounded-2xl flex items-center justify-center transition-all duration-300 z-10 group-hover:scale-110",
              isActive ? "bg-primary text-white shadow-[0_0_20px_rgba(59,130,246,0.4)] border border-primary" : 
              isPast ? "bg-primary/20 text-primary border border-primary/30 group-hover:bg-primary/30" : 
              "bg-surface text-textMuted border border-border group-hover:border-textMuted/50 group-hover:text-textMain/80"
            )}>
              <step.icon size={20} className={isActive ? "animate-pulse" : "transition-transform group-hover:scale-110"} />
            </div>
            <span className={clsx(
              "text-xs font-bold uppercase tracking-wider transition-colors duration-300", 
              isActive ? "text-primary" : isPast ? "text-textMain group-hover:text-primary" : "text-textMuted group-hover:text-textMain/80"
            )}>
              {step.name}
            </span>
          </Link>
        );
      })}
    </div>
  );
};

export default StepFlow;
