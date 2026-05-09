import React, { useEffect, useState } from 'react';
import StepFlow from '../components/ui/StepFlow';
import { api } from '../api';
import { useGlobalState } from '../context/GlobalState';
import { BarChart2, ChevronDown, ChevronUp, CheckCircle2, Play } from 'lucide-react';
import clsx from 'clsx';
import { Link } from 'react-router-dom';

export default function Evaluation() {
  const { status } = useGlobalState();
  const [metrics, setMetrics] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const fetchEval = async () => {
      const res = await api.evaluate();
      setMetrics(res);
      setIsLoading(false);
    };
    fetchEval();
  }, []);

  if (isLoading || status === 'training') {
    return (
      <div className="max-w-4xl mx-auto flex flex-col items-center justify-center gap-6 mt-32 animate-in fade-in">
         <div className="relative">
           <div className="w-24 h-24 rounded-full border-4 border-surfaceHover"></div>
           <div className="w-24 h-24 rounded-full border-4 border-primary border-t-transparent animate-spin absolute top-0 left-0"></div>
           <BarChart2 className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-primary" size={32} />
         </div>
         <h2 className="text-2xl font-bold text-textMain animate-pulse">Running Evaluation Pipeline...</h2>
         <p className="text-textMuted">Generating samples and calculating ROUGE & BERT Scores.</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <StepFlow />
      
      <div className="flex justify-between items-end">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-bold text-textMain tracking-tight mb-2">Model Evaluation</h1>
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-success/10 border border-success/20 text-success rounded-full text-xs font-bold w-max">
            <CheckCircle2 size={14} /> Output Verified
          </div>
        </div>
        <Link to="/export" className="px-6 py-2.5 bg-surfaceHover border border-border hover:bg-primary/20 hover:text-primary hover:border-primary/50 text-textMain rounded-xl transition-all font-medium">
          Proceed to Export
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-surface border border-border/50 p-8 rounded-3xl shadow-xl flex flex-col items-center justify-center relative overflow-hidden group">
          <div className="absolute -left-10 -bottom-10 w-32 h-32 bg-primary/10 rounded-full blur-2xl group-hover:bg-primary/20 transition-colors"></div>
          <span className="text-sm font-bold text-textMuted uppercase tracking-wider mb-4 relative z-10">ROUGE-L</span>
          <div className="relative">
             <svg className="w-32 h-32 transform -rotate-90">
               <circle cx="64" cy="64" r="56" stroke="currentColor" strokeWidth="12" fill="transparent" className="text-surfaceHover" />
               <circle cx="64" cy="64" r="56" stroke="currentColor" strokeWidth="12" fill="transparent" strokeDasharray="351.8" strokeDashoffset={351.8 - (351.8 * metrics?.rouge)} className="text-primary transition-all duration-1000 ease-out" />
             </svg>
             <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-3xl font-extrabold text-textMain">{(metrics?.rouge * 100).toFixed(1)}%</span>
          </div>
        </div>

        <div className="bg-surface border border-border/50 p-8 rounded-3xl shadow-xl flex flex-col items-center justify-center relative overflow-hidden group">
          <div className="absolute -left-10 -bottom-10 w-32 h-32 bg-accent/10 rounded-full blur-2xl group-hover:bg-accent/20 transition-colors"></div>
          <span className="text-sm font-bold text-textMuted uppercase tracking-wider mb-4 relative z-10">BERTScore</span>
          <div className="relative">
             <svg className="w-32 h-32 transform -rotate-90">
               <circle cx="64" cy="64" r="56" stroke="currentColor" strokeWidth="12" fill="transparent" className="text-surfaceHover" />
               <circle cx="64" cy="64" r="56" stroke="currentColor" strokeWidth="12" fill="transparent" strokeDasharray="351.8" strokeDashoffset={351.8 - (351.8 * metrics?.bert)} className="text-accent transition-all duration-1000 ease-out" />
             </svg>
             <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-3xl font-extrabold text-textMain">{(metrics?.bert * 100).toFixed(1)}%</span>
          </div>
        </div>

        <div className="bg-surface border border-border/50 p-8 rounded-3xl shadow-xl flex flex-col items-center justify-center relative overflow-hidden group">
          <div className="absolute -left-10 -bottom-10 w-32 h-32 bg-warning/10 rounded-full blur-2xl group-hover:bg-warning/20 transition-colors"></div>
          <span className="text-sm font-bold text-textMuted uppercase tracking-wider mb-4 relative z-10">Token F1</span>
          <div className="relative">
             <svg className="w-32 h-32 transform -rotate-90">
               <circle cx="64" cy="64" r="56" stroke="currentColor" strokeWidth="12" fill="transparent" className="text-surfaceHover" />
               <circle cx="64" cy="64" r="56" stroke="currentColor" strokeWidth="12" fill="transparent" strokeDasharray="351.8" strokeDashoffset={351.8 - (351.8 * metrics?.f1)} className="text-warning transition-all duration-1000 ease-out" />
             </svg>
             <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-3xl font-extrabold text-textMain">{(metrics?.f1 * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      <div className="bg-surface border border-border/50 rounded-3xl shadow-xl overflow-hidden mt-4">
        <button 
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center justify-between p-6 hover:bg-surfaceHover transition-colors focus:outline-none"
        >
          <div className="flex items-center gap-3">
             <div className="p-2 bg-primary/10 text-primary rounded-lg">
                <Play size={18} />
             </div>
             <h3 className="text-lg font-bold text-textMain">Sample Outputs Generation</h3>
          </div>
          {expanded ? <ChevronUp size={24} className="text-textMuted" /> : <ChevronDown size={24} className="text-textMuted" />}
        </button>

        {expanded && metrics?.samples && (
          <div className="p-6 pt-0 border-t border-border/30 animate-in fade-in slide-in-from-top-4">
            <div className="flex flex-col gap-6 mt-6">
              {metrics.samples.map((sample, idx) => (
                <div key={idx} className="bg-background border border-border/50 rounded-2xl p-5 shadow-inner">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <span className="text-[10px] font-bold text-success uppercase bg-success/10 px-2 py-1 rounded inline-block mb-2 border border-success/20">Target (Actual)</span>
                      <p className="text-sm text-textMain leading-relaxed p-3 bg-surface rounded-xl border border-border/30">{sample.target}</p>
                    </div>
                    <div>
                      <span className="text-[10px] font-bold border border-primary/20 text-primary uppercase bg-primary/10 px-2 py-1 rounded inline-block mb-2">Predicted (Model)</span>
                      <p className="text-sm text-textMain leading-relaxed p-3 bg-surface rounded-xl border border-border/30">{sample.predicted}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
