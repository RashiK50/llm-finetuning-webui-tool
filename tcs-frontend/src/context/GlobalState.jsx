import React, { createContext, useContext, useState, useEffect } from 'react';
import { api, notifyBackendDown, clearBackendDownToast } from '../api';

const GlobalStateContext = createContext();

export const GlobalStateProvider = ({ children }) => {
  const [status, setStatus] = useState('idle');
  const [model, setModel] = useState(null);
  const [modelType, setModelType] = useState('none');
  const [dataset, setDataset] = useState(null);
  const [gpu, setGpu] = useState('NVIDIA A100 80GB (Idle)');
  const [gpuMem, setGpuMem] = useState(0);
  const [gpuTotalMem, setGpuTotalMem] = useState(0);
  const [gpuName, setGpuName] = useState('CPU Mode');
  const [ram, setRam] = useState(0); // Placeholder, update if backend provides
  const [ramTotal, setRamTotal] = useState(0);
  const [datasetName, setDatasetName] = useState('');
  const [backendOnline, setBackendOnline] = useState(true);
  
  // Training state
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [exportDownloadUrl, setExportDownloadUrl] = useState('');
  const [exportArtifactPath, setExportArtifactPath] = useState('');
  const [exportType, setExportType] = useState('');
  const [exportError, setExportError] = useState('');
  // HF push state (new)
  const [hfPushStatus, setHfPushStatus]   = useState('');
  const [hfPushMessage, setHfPushMessage] = useState('');
  const [hfPushedRepo, setHfPushedRepo]   = useState('');
  const [ggufPushStatus, setGgufPushStatus]   = useState('');
  const [ggufPushMessage, setGgufPushMessage] = useState('');
  const [ggufPushedRepo, setGgufPushedRepo]   = useState('');

  // Frontend Persistent State
  const [modelConfig, setModelConfig] = useState({ modelName: "mistralai/Mistral-7B-v0.3", customPath: "" });
  const [trainConfig, setTrainConfig] = useState({ lora_rank: 16, lora_alpha: 32, lora_dropout: 0.05, learning_rate: 0.0002, epochs: 3, batch_size: 2 });
  const [evalResults, setEvalResults] = useState(null);
  const [chatMessages, setChatMessages] = useState([
    { role: 'assistant', content: 'Model is online. Ask me something from your fine-tuning domain.' }
  ]);

  useEffect(() => {
    const syncStatus = (res) => {
      setBackendOnline(true);
      clearBackendDownToast();
      setStatus(res.status || 'idle');
      setModel(res.model_name || null);
      setModelType(res.model_type || 'none');
      setGpuMem(res.gpu_memory_used_gb || 0);
      setGpuTotalMem(res.gpu_memory_total_gb || 0);
      setGpuName(res.gpu_name || 'CPU Mode');
      setRam(res.ram_used_gb || 0);
      setRamTotal(res.ram_total_gb || 0);
      setGpu(res.gpu_available ? 'GPU Active' : 'CPU Mode');
      setDatasetName(res.dataset_name || '');
      setCurrentEpoch(res.current_epoch || 0);
      setCurrentStep(res.current_step || 0);
      setLoadingProgress(res.loading_progress || 0);
      setLoadingMessage(res.loading_message || '');
      setExportDownloadUrl(res.export_download_url || '');
      setExportArtifactPath(res.export_artifact_path || '');
      setExportType(res.export_type || '');
      setExportError(res.export_error || '');
      // HF push state (new)
      setHfPushStatus(res.hf_push_status   || '');
      setHfPushMessage(res.hf_push_message || '');
      setHfPushedRepo(res.hf_pushed_repo   || '');
      setGgufPushStatus(res.gguf_push_status   || '');
      setGgufPushMessage(res.gguf_push_message || '');
      setGgufPushedRepo(res.gguf_pushed_repo   || '');
      if (res.eval_results !== undefined && res.eval_results !== null) {
        setEvalResults(res.eval_results);
      }

      if (res.dataset_ready) {
        setDataset((prev) => ({
          ...(prev || {}),
          name: res.dataset_name || prev?.name || 'Dataset',
          size: res.dataset_size || 0,
          preview: Array.isArray(res.dataset_preview) && res.dataset_preview.length > 0
            ? res.dataset_preview
            : (prev?.preview || []),
        }));
      } else {
        setDataset(null);
      }

      if (!res.model_loaded || (res.status || '') === 'idle') {
        setEvalResults(null);
        setChatMessages([{ role: 'assistant', content: 'Model is online. Ask me something from your fine-tuning domain.' }]);
      }
    };

    api.status().then(syncStatus).catch(() => {
      setBackendOnline(false);
      notifyBackendDown();
    });

    const eventSource = new EventSource('http://localhost:8000/api/status-stream');

    eventSource.onmessage = (event) => {
      try {
        syncStatus(JSON.parse(event.data));
      } catch (e) {
        console.error('Error parsing SSE data', e);
      }
    };

    eventSource.onerror = () => {
      setBackendOnline(false);
      setLoadingMessage('');
      notifyBackendDown();
    };

    return () => eventSource.close();
  }, []);

  return (
    <GlobalStateContext.Provider value={{
      status, setStatus,
      model, setModel,
      modelType,
      dataset, setDataset,
      gpu, setGpu,
      gpuMem, gpuTotalMem, gpuName, ram, ramTotal, datasetName,
      backendOnline,
      currentEpoch, currentStep,
      loadingProgress, loadingMessage,
      exportDownloadUrl, exportArtifactPath, exportType, exportError,
      hfPushStatus, hfPushMessage, hfPushedRepo,
      ggufPushStatus, ggufPushMessage, ggufPushedRepo,
      modelConfig, setModelConfig,
      trainConfig, setTrainConfig,
      evalResults, setEvalResults,
      chatMessages, setChatMessages
    }}>
      {children}
    </GlobalStateContext.Provider>
  );
};

export const useGlobalState = () => useContext(GlobalStateContext);
