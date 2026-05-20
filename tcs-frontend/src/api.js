import toast from 'react-hot-toast';

const BASE = "http://localhost:8000/api";
const BACKEND_TOAST_ID = "backend-offline";

const notifyBackendDown = (message = "Backend is not connected. Start the API server and refresh.") => {
  toast.error(message, { id: BACKEND_TOAST_ID, duration: Infinity });
};

const clearBackendDownToast = () => {
  toast.dismiss(BACKEND_TOAST_ID);
};

const request = async (path, options = {}, { showToast = true } = {}) => {
  try {
    const res = await fetch(`${BASE}${path}`, options);
    if (!res.ok) {
      let message = `Request failed (${res.status})`;
      try {
        const data = await res.json();
        message = data.detail || data.message || message;
      } catch {
        // Ignore JSON parse errors and keep the fallback message.
      }
      if (showToast) toast.error(message);
      throw new Error(message);
    }
    if (res.status === 204) return null;
    return await res.json();
  } catch (error) {
    if (error instanceof TypeError) {
      notifyBackendDown();
      throw new Error("Backend unavailable");
    }
    throw error;
  }
};

export const api = {
  status: async () => request("/status"),
  loadModel: async (model_name) => request("/load-model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name })
  }),
  unloadModel: async () => request("/unload-model", { method: "POST" }),
  cancelOperation: async (operation = "loading_model") => request("/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ operation })
  }),
  uploadDataset: async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    return request("/upload-dataset", {
      method: "POST",
      body: formData
    });
  },
  train: async (config) => request("/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config)
  }),
  progress: async () => request("/progress", {}, { showToast: false }),
  evaluate: async (num_samples = 10) => request("/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ num_samples })
  }),
  export: async (type = "lora", hfRepoId = "") => {
    const params = new URLSearchParams({ type });
    if (hfRepoId) params.append("hf_repo_id", hfRepoId);
    return request(`/export?${params.toString()}`);
  },
  exportGguf: async (mergedModelDir, hfRepoId, quantisation = "f16") =>
    request("/export-gguf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        merged_model_dir: mergedModelDir,
        hf_repo_id: hfRepoId,
        quantisation,
      }),
    }),
  loadFinetunedModel: async (baseModelName, loraRepoId) =>
    request("/load-finetuned-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        base_model_name: baseModelName,
        lora_repo_id: loraRepoId,
      }),
    }),
  uploadEvalDataset: async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    return request("/upload-eval-dataset", { method: "POST", body: formData });
  },
  clearDataset: async () => request("/clear-dataset", { method: "POST" }),
  chat: async (message, history = [], max_tokens = 200, temperature = 0.7) => request("/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, max_tokens, temperature })
  }),
};

export { notifyBackendDown, clearBackendDownToast };
