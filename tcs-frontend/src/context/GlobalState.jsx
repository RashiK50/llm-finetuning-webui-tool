import React, { createContext, useContext, useState, useEffect } from 'react';
import { api } from '../api';

const GlobalStateContext = createContext();

export const GlobalStateProvider = ({ children }) => {
  const [status, setStatus] = useState('idle');
  const [model, setModel] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [gpu, setGpu] = useState('NVIDIA A100 80GB (Idle)');

  useEffect(() => {
    const fetchStatus = async () => {
      const res = await api.status();
      // Keep existing status if it has progressed beyond 'idle'
      setStatus(prev => prev !== 'idle' ? prev : res.status);
    };
    fetchStatus();
    
    const interval = setInterval(() => {
      if (status === 'training' || status === 'evaluating') {
        setGpu('NVIDIA A100 80GB (98% / 350W)');
      } else {
        setGpu('NVIDIA A100 80GB (Idle)');
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [status]);

  return (
    <GlobalStateContext.Provider value={{
      status, setStatus,
      model, setModel,
      dataset, setDataset,
      gpu, setGpu
    }}>
      {children}
    </GlobalStateContext.Provider>
  );
};

export const useGlobalState = () => useContext(GlobalStateContext);
