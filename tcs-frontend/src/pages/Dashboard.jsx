import React from 'react';
import { useGlobalState } from '../context/GlobalState';
import { Server, Database, Activity, Cpu } from 'lucide-react';
import { Link } from 'react-router-dom';

const QuickActionCard = ({ title, desc, icon: Icon, to, active }) => (
  <Link to={to} className="group relative bg-surface hover:bg-surfaceHover border border-border/50 rounded-3xl p-6 transition-all duration-300 shadow-lg hover:shadow-[0_8px_30px_rgb(0,0,0,0.12)] hover:-translate-y-1 block overflow-hidden">
    <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-full blur-3xl -mr-10 -mt-10 transition-all duration-500 group-hover:bg-primary/20 pointer-events-none"></div>
    <div className="flex items-center gap-4 mb-4">
      <div className={`p-3 rounded-2xl ${active ? 'bg-primary text-white shadow-[0_0_15px_rgba(59,130,246,0.5)]' : 'bg-background text-primary border border-border/50'}`}>
        <Icon size={24} />
      </div>
      <h3 className="text-xl font-bold text-textMain tracking-tight">{title}</h3>
    </div>
    <p className="text-textMuted text-sm font-medium">{desc}</p>
  </Link>
);

export default function Dashboard() {
  const { model, dataset, status } = useGlobalState();

  return (
    <div className="max-w-6xl mx-auto flex flex-col gap-10 animate-in fade-in slide-in-from-bottom-4 duration-500 pb-10">
      <div className="flex flex-col gap-3 mt-4">
        <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-primary to-accent inline-block tracking-tight">TCS FineTuner Studio</h1>
        <p className="text-textMuted text-lg font-medium max-w-2xl">End-to-End LLM orchestration platform. Train, evaluate, and orchestrate large language models efficiently.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <QuickActionCard 
          title="Setup Model" 
          desc={model ? `Active: ${model}` : "Select a pre-trained base model for fine-tuning."} 
          icon={Server} 
          to="/model-setup" 
          active={!!model} 
        />
        <QuickActionCard 
          title="Prepare Dataset" 
          desc={dataset ? `Loaded: ${dataset.size}` : "Upload instruction pairs (JSON) for training."} 
          icon={Database} 
          to="/dataset" 
          active={!!dataset} 
        />
        <QuickActionCard 
          title="Training Config" 
          desc={status === 'training' ? "Training in progress..." : "Configure hyperparameters and start LoRA SFT."} 
          icon={Activity} 
          to="/training" 
          active={status === 'training'} 
        />
      </div>

      <div className="bg-surface border border-border/50 rounded-3xl p-8 shadow-2xl relative overflow-hidden flex flex-col gap-6">
        <div className="flex items-center gap-3">
          <Cpu className="text-accent" size={24} />
          <h2 className="text-2xl font-bold text-textMain tracking-tight">System Resource Overview</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-background rounded-2xl p-6 border border-border/30">
            <h4 className="text-sm font-semibold text-textMuted mb-2 uppercase tracking-wide">GPU Memory</h4>
            <div className="text-3xl font-extrabold text-textMain">12<span className="text-lg text-textMuted font-medium"> / 80 GB</span></div>
            <div className="w-full bg-surfaceHover h-2 rounded-full mt-4 overflow-hidden">
               <div className="bg-gradient-to-r from-primary to-accent h-full w-[15%] rounded-full"></div>
            </div>
          </div>
          <div className="bg-background rounded-2xl p-6 border border-border/30">
            <h4 className="text-sm font-semibold text-textMuted mb-2 uppercase tracking-wide">System RAM</h4>
            <div className="text-3xl font-extrabold text-textMain">42<span className="text-lg text-textMuted font-medium"> / 256 GB</span></div>
            <div className="w-full bg-surfaceHover h-2 rounded-full mt-4 overflow-hidden">
               <div className="bg-gradient-to-r from-primary to-accent h-full w-[16%] rounded-full"></div>
            </div>
          </div>
          <div className="bg-background rounded-2xl p-6 border border-border/30 flex justify-center flex-col">
            <h4 className="text-sm font-semibold text-textMuted mb-2 uppercase tracking-wide">Orchestration</h4>
            <div className="flex items-center gap-2">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-success"></span>
              </span>
              <span className="text-sm font-medium text-success">Cluster Online</span>
            </div>
            <span className="text-xs text-textMuted mt-2">Connected to primary compute node.</span>
          </div>
        </div>
      </div>
    </div>
  );
}
