import React, { useEffect, useState } from 'react';
import { useGlobalState } from '../context/GlobalState';
import StepFlow from '../components/ui/StepFlow';
import { api } from '../api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, CheckCircle2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function LiveProgress() {
  const { status, setStatus } = useGlobalState();
  const [data, setData] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [currentEpoch, setCurrentEpoch] = useState(1);
  const navigate = useNavigate();

  useEffect(() => {
    if (status !== 'training' && status !== 'evaluating' && status !== 'done') return;
    
    // If we're just visiting and it's already done
    if (status === 'done') {
      setData([{ step: 0, loss: 2.5 }, { step: 50, loss: 1.2 }, { step: 100, loss: 0.12 }]);
      setCurrentStep(100);
      return;
    }

    let loss = 2.5; 
    const interval = setInterval(async () => {
      loss = Math.max(0.1, loss - 0.2 - (Math.random() * 0.1));
      
      setData(prev => {
        const newData = [...prev, { step: prev.length * 10, loss }];
        if (newData.length > 50) return newData.slice(newData.length - 50);
        return newData;
      });
      setCurrentStep(prev => prev + 10);
      
      if (loss <= 0.15) {
        setStatus('evaluating');
        clearInterval(interval);
        setTimeout(() => setStatus('done'), 2000);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [status, setStatus]);

  const progressPercent = Math.min(100, Math.max(0, (2.5 - (data[data.length - 1]?.loss || 2.5)) / 2.35 * 100));

  return (
    <div className="max-w-5xl mx-auto flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <StepFlow />
      <div className="flex justify-between items-end">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-bold text-textMain tracking-tight">Live Progress</h1>
          <p className="text-textMuted">Monitor training metrics and loss curve in real-time.</p>
        </div>
        {status === 'done' && (
          <button onClick={() => navigate('/evaluation')} className="px-6 py-2.5 bg-primary hover:bg-primaryHover text-white font-medium rounded-xl transition-all shadow-[0_4px_14px_0_rgba(59,130,246,0.39)] animate-in fade-in zoom-in">
            Go to Evaluation
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-surface border border-border/50 p-6 rounded-3xl shadow-sm flex flex-col justify-center relative overflow-hidden group">
           <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity"><Activity size={80}/></div>
           <span className="text-sm font-bold text-textMuted uppercase tracking-wider mb-2">Current Loss</span>
           <span className="text-4xl font-extrabold text-primary">{data.length > 0 ? data[data.length - 1].loss.toFixed(4) : "2.500"}</span>
        </div>
        <div className="bg-surface border border-border/50 p-6 rounded-3xl shadow-sm flex flex-col justify-center relative">
           <span className="text-sm font-bold text-textMuted uppercase tracking-wider mb-2">Epoch</span>
           <span className="text-4xl font-extrabold text-textMain">{currentEpoch} <span className="text-lg text-textMuted">/ 3</span></span>
        </div>
        <div className="bg-surface border border-border/50 p-6 rounded-3xl shadow-sm flex flex-col justify-center relative">
           <span className="text-sm font-bold text-textMuted uppercase tracking-wider mb-2">Total Steps</span>
           <span className="text-4xl font-extrabold text-textMain">{currentStep}</span>
        </div>
      </div>

      <div className="bg-surface border border-border/50 rounded-3xl p-8 shadow-xl relative mt-2">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-lg font-bold text-textMain">Training Loss Curve</h3>
          {status === 'training' && (
            <span className="flex items-center gap-2 text-sm font-bold text-warning bg-warning/10 px-4 py-1.5 rounded-full border border-warning/20 shadow-sm">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-warning opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-warning"></span>
              </span>
              Training...
            </span>
          )}
          {status === 'done' && (
             <span className="flex items-center gap-2 text-sm font-bold text-success bg-success/10 px-4 py-1.5 rounded-full border border-success/20 shadow-sm">
               <CheckCircle2 size={16} /> Training Complete
             </span>
          )}
        </div>

        <div className="w-full bg-background border border-border/50 rounded-full h-3 mb-8 overflow-hidden shadow-inner">
          <div className="bg-gradient-to-r from-warning via-primary to-accent h-full transition-all duration-1000 ease-out" style={{ width: `${progressPercent}%` }}></div>
        </div>

        <div className="h-80 w-full bg-background/50 rounded-2xl p-4 border border-border/30">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2A2E39" vertical={false} />
              <XAxis dataKey="step" stroke="#94A3B8" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke="#94A3B8" fontSize={12} tickLine={false} axisLine={false} domain={['auto', 'auto']} dx={-10} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1E212B', borderColor: '#334155', borderRadius: '16px', color: '#F8FAFC', boxShadow: '0 10px 25px rgba(0,0,0,0.3)' }}
                itemStyle={{ color: '#3B82F6', fontWeight: 'bold' }}
                cursor={{ stroke: '#334155', strokeWidth: 2, strokeDasharray: '5 5' }}
              />
              <Line type="monotone" dataKey="loss" stroke="#3B82F6" strokeWidth={4} dot={false} activeDot={{ r: 8, fill: '#10B981', stroke: '#1E212B', strokeWidth: 3 }} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
