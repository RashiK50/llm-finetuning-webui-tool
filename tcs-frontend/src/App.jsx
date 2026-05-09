import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import ModelSetup from './pages/ModelSetup';
import DatasetUpload from './pages/DatasetUpload';
import TrainingPanel from './pages/TrainingPanel';
import Evaluation from './pages/Evaluation';
import ExportPanel from './pages/ExportPanel';
import ChatPlayground from './pages/ChatPlayground';
import LiveProgress from './pages/LiveProgress';

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="model-setup" element={<ModelSetup />} />
        <Route path="dataset" element={<DatasetUpload />} />
        <Route path="training" element={<TrainingPanel />} />
        <Route path="live-progress" element={<LiveProgress />} />
        <Route path="evaluation" element={<Evaluation />} />
        <Route path="export" element={<ExportPanel />} />
        <Route path="chat" element={<ChatPlayground />} />
      </Route>
    </Routes>
  );
};

export default App;
