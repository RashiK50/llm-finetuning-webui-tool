import React, { useEffect, useMemo, useRef, useState } from 'react';
import StepFlow from '../components/ui/StepFlow';
import { api } from '../api';
import { Send, Loader2, Bot, User, AlertCircle, Sparkles } from 'lucide-react';
import { useGlobalState } from '../context/GlobalState';
import clsx from 'clsx';
import toast from 'react-hot-toast';

export default function ChatPlayground() {
  const { status, model, chatMessages, setChatMessages } = useGlobalState();
  const messages = chatMessages;
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const isReady = useMemo(() => {
    if (!model) return false;
    return !['idle', 'loading_model', 'error'].includes(status);
  }, [model, status]);

  const handleSend = async (e) => {
    e.preventDefault();
    const prompt = input.trim();
    if (!prompt || isLoading || !isReady) return;

    const history = messages.map((msg) => ({ role: msg.role, content: msg.content }));
    setInput('');
    setChatMessages((prev) => [...prev, { role: 'user', content: prompt }]);
    setIsLoading(true);

    try {
      const res = await api.chat(prompt, history, 220, 0.35);
      const reply = String(res?.response || '').trim();
      setChatMessages((prev) => [...prev, { role: 'assistant', content: reply || 'No response generated.' }]);
    } catch {
      setChatMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'I could not generate a response right now. Please retry.' },
      ]);
      toast.error('Chat request failed.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-[1500px] mx-auto h-full min-h-0 flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-4 duration-500 px-2">
      <StepFlow />

      <div className="flex items-center justify-between gap-3 shrink-0">
        <div className="min-w-0">
          <h1 className="text-2xl md:text-3xl font-extrabold text-textMain tracking-tight">Chat Playground</h1>
          <p className="text-sm text-textMuted mt-1">Conversation uses the latest context window and backend generation settings.</p>
        </div>
        <div className="hidden md:flex items-center gap-2 rounded-xl border border-border/50 bg-surface px-3 py-2 text-xs text-textMuted">
          <Sparkles size={13} className="text-primary" /> Context-aware replies
        </div>
      </div>

      {!isReady && (
        <div className="flex items-start gap-3 bg-warning/5 border border-warning/20 text-warning p-4 rounded-2xl shadow-sm shrink-0">
          <AlertCircle size={18} className="shrink-0 mt-0.5" />
          <div className="text-sm">Load and prepare the model before using chat.</div>
        </div>
      )}

      <div className="flex-1 min-h-0 bg-surface border border-border/50 rounded-3xl shadow-xl flex flex-col overflow-hidden">
        <div className="flex-1 min-h-0 overflow-y-auto p-5 md:p-6 flex flex-col gap-4 bg-[radial-gradient(circle_at_top,_rgba(59,130,246,0.05),_transparent_40%)]">
          {messages.map((msg, idx) => (
            <div
              key={`${idx}-${msg.role}`}
              className={clsx('flex gap-3 max-w-[88%]', msg.role === 'user' ? 'self-end flex-row-reverse' : 'self-start')}
            >
              <div
                className={clsx(
                  'shrink-0 w-9 h-9 rounded-xl flex items-center justify-center shadow-sm',
                  msg.role === 'user' ? 'bg-primary text-white' : 'bg-surfaceHover text-primary border border-border/80',
                )}
              >
                {msg.role === 'user' ? <User size={16} /> : <Bot size={17} />}
              </div>
              <div
                className={clsx(
                  'px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm whitespace-pre-wrap break-words',
                  msg.role === 'user'
                    ? 'bg-primary text-white rounded-tr-none'
                    : 'bg-background border border-border/50 text-textMain rounded-tl-none',
                )}
              >
                {msg.content}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-3 max-w-[88%] self-start">
              <div className="shrink-0 w-9 h-9 rounded-xl bg-surfaceHover border border-border text-primary flex items-center justify-center">
                <Bot size={17} />
              </div>
              <div className="px-4 py-3 rounded-2xl text-sm bg-surfaceHover border border-border/50 text-textMuted rounded-tl-none flex items-center gap-2">
                <Loader2 size={15} className="animate-spin text-primary" />
                Thinking...
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        <div className="p-4 border-t border-border/50 bg-background/70 shrink-0">
          <form onSubmit={handleSend} className="relative flex items-center">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={!isReady || isLoading}
              placeholder={isReady ? 'Ask your model...' : 'Waiting for model readiness...'}
              className="w-full bg-background border border-border/50 text-textMain rounded-full pl-5 pr-14 py-3.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!isReady || isLoading || !input.trim()}
              className="absolute right-2 w-9 h-9 rounded-full flex items-center justify-center bg-primary hover:bg-primaryHover text-white disabled:bg-surfaceHover disabled:text-textMuted transition-all"
            >
              <Send size={15} className={clsx(input.trim() && isReady && !isLoading ? '-ml-0.5 mt-0.5' : '')} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
