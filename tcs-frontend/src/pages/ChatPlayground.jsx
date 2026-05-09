import React, { useState, useRef, useEffect } from 'react';
import StepFlow from '../components/ui/StepFlow';
import { api } from '../api';
import { Send, Loader2, Bot, User, AlertCircle } from 'lucide-react';
import { useGlobalState } from '../context/GlobalState';
import clsx from 'clsx';

export default function ChatPlayground() {
  const { status } = useGlobalState();
  const [messages, setMessages] = useState([
    { role: 'model', content: 'Hello! I am your fine-tuned model. How can I help you today?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const endRef = useRef(null);

  const scrollToBottom = () => endRef.current?.scrollIntoView({ behavior: "smooth" });

  useEffect(() => { scrollToBottom() }, [messages, isLoading]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    const res = await api.chat(userMessage);
    
    setMessages(prev => [...prev, { role: 'model', content: res.response }]);
    setIsLoading(false);
  };

  const isReady = status === 'done';

  return (
    <div className="max-w-4xl mx-auto flex flex-col h-[calc(100vh-8rem)] animate-in fade-in slide-in-from-bottom-4 duration-500 pb-6">
      <StepFlow />
      
      <div className="flex flex-col gap-2 mb-6 shrink-0">
        <h1 className="text-3xl font-bold text-textMain tracking-tight">Chat Playground</h1>
        <p className="text-textMuted">Test your fine-tuned model interactively.</p>
      </div>

      {!isReady && (
        <div className="flex items-start gap-4 bg-warning/5 border border-warning/20 text-warning p-5 rounded-2xl shadow-sm mb-6 shrink-0">
          <AlertCircle size={20} className="shrink-0 mt-0.5" />
          <div className="flex flex-col gap-1">
            <span className="font-bold text-sm tracking-wide uppercase">Model Not Available</span>
            <span className="text-sm font-medium opacity-90">Please finish training the model before using the chat playground. It requires the 'done' status.</span>
          </div>
        </div>
      )}

      <div className="flex-1 bg-surface border border-border/50 rounded-3xl shadow-xl flex flex-col overflow-hidden relative">
        {/* Chat History */}
        <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
          {messages.map((msg, idx) => (
            <div key={idx} className={clsx("flex gap-4 max-w-[85%]", msg.role === 'user' ? "self-end flex-row-reverse" : "self-start")}>
               <div className={clsx(
                 "shrink-0 w-10 h-10 rounded-full flex items-center justify-center shadow-md",
                 msg.role === 'user' ? "bg-primary text-white" : "bg-surfaceHover text-primary border border-border/80"
               )}>
                 {msg.role === 'user' ? <User size={18} /> : <Bot size={20} />}
               </div>
               <div className={clsx(
                 "p-4 rounded-2xl text-sm leading-relaxed shadow-sm",
                 msg.role === 'user' 
                   ? "bg-primary text-white rounded-tr-none shadow-[0_4px_15px_rgba(59,130,246,0.3)]" 
                   : "bg-surfaceHover border border-border/50 text-textMain rounded-tl-none shadow-sm"
               )}>
                 {msg.content}
               </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-4 max-w-[85%] self-start animate-in fade-in zoom-in">
               <div className="shrink-0 w-10 h-10 rounded-full bg-surfaceHover border border-border text-primary flex items-center justify-center shadow-sm">
                 <Bot size={20} />
               </div>
               <div className="p-4 rounded-2xl text-sm bg-surfaceHover border border-border/50 text-textMuted rounded-tl-none flex items-center gap-2">
                 <Loader2 size={16} className="animate-spin text-primary" />
                 <span className="animate-pulse">Model is computing response...</span>
               </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 border-t border-border/50 bg-background/50 backdrop-blur shrink-0">
          <form onSubmit={handleSend} className="relative flex items-center">
             <input 
               type="text" 
               value={input}
               onChange={(e) => setInput(e.target.value)}
               disabled={!isReady || isLoading}
               placeholder={isReady ? "Message your model..." : "Waiting for model to be trained..."}
               className="w-full bg-background border border-border/50 text-textMain rounded-full pl-6 pr-14 py-4 focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50 transition-all shadow-inner"
             />
             <button 
               type="submit"
               disabled={!isReady || isLoading || !input.trim()}
               className="absolute right-2 w-10 h-10 rounded-full flex items-center justify-center bg-primary hover:bg-primaryHover text-white disabled:bg-surfaceHover disabled:text-textMuted transition-all focus:ring-2 focus:ring-primary/50 focus:ring-offset-2 focus:ring-offset-background"
             >
                <Send size={16} className={clsx(input.trim() && isReady && !isLoading ? "-ml-0.5 mt-0.5" : "")} />
             </button>
          </form>
        </div>
      </div>
    </div>
  );
}
