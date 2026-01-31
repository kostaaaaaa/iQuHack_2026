import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Activity, ShieldCheck, Zap } from 'lucide-react';
import './App.css';

// Placeholder API calls - replace with real fetches later
const fetchClassicalVaR = async (params) => {
  const response = await fetch('http://localhost:8000/api/classical-var', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  return response.json();
};

const fetchQuantumVaR = async (params) => {
  const response = await fetch('http://localhost:8000/api/quantum-var', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  return response.json();
};

function App() {
  const [params, setParams] = useState({
    confidence_level: 0.95,
    time_horizon: 1,
    shots: 1000
  });

  const [classicalResult, setClassicalResult] = useState(null);
  const [quantumResult, setQuantumResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleRun = async () => {
    setLoading(true);
    try {
      const [cRes, qRes] = await Promise.all([
        fetchClassicalVaR(params),
        fetchQuantumVaR(params)
      ]);
      setClassicalResult(cRes);
      setQuantumResult(qRes);
    } catch (error) {
      console.error("API Error", error);
    } finally {
      setLoading(false);
    }
  };

  // Mock data for visualization
  const data = [
    { x: -0.2, y: 0.01 }, { x: -0.15, y: 0.05 }, { x: -0.1, y: 0.15 },
    { x: -0.05, y: 0.35 }, { x: 0, y: 0.6 }, { x: 0.05, y: 0.35 },
    { x: 0.1, y: 0.15 }, { x: 0.15, y: 0.05 }, { x: 0.2, y: 0.01 },
  ];

  return (
    <div className="container">
      <header className="header">
        <h1 className="title text-gradient">
          <Zap size={32} style={{ marginBottom: -6, marginRight: 8 }} />
          Quantum VaR Est.
        </h1>
        <p className="subtitle">Risk Analysis: Classical Monte Carlo vs. Quantum Amplitude Estimation</p>
      </header>

      <div className="main-grid">
        <aside className="sidebar glass-panel">
          <h3>Parameters</h3>
          <div className="input-group">
            <label>Confidence Level</label>
            <input
              type="number"
              step="0.01"
              max="1"
              className="input-field"
              value={params.confidence_level}
              onChange={(e) => setParams({ ...params, confidence_level: parseFloat(e.target.value) })}
            />
          </div>
          <div className="input-group">
            <label>Time Horizon (Years)</label>
            <input
              type="number"
              className="input-field"
              value={params.time_horizon}
              onChange={(e) => setParams({ ...params, time_horizon: parseInt(e.target.value) })}
            />
          </div>
          <div className="input-group">
            <label>Shots / Samples</label>
            <input
              type="number"
              className="input-field"
              value={params.shots}
              onChange={(e) => setParams({ ...params, shots: parseInt(e.target.value) })}
            />
          </div>
          <button className="btn" onClick={handleRun} disabled={loading}>
            {loading ? 'Running...' : 'Calculate VaR'}
          </button>
        </aside>

        <section className="results-area">
          <div className="cards-row">
            <div className="glass-panel result-card">
              <div className="card-header">
                <Activity size={20} color="var(--color-primary)" />
                <h4>Classical Monte Carlo</h4>
              </div>
              <div className="metric">
                {classicalResult ? (classicalResult.var_value * 100).toFixed(2) + '%' : '--'}
              </div>
              <div className="meta">
                {classicalResult ? `${classicalResult.samples} samples` : 'Waiting for run'}
              </div>
            </div>

            <div className="glass-panel result-card" style={{ borderColor: 'var(--color-secondary)' }}>
              <div className="card-header">
                <ShieldCheck size={20} color="var(--color-secondary)" />
                <h4>Quantum AE</h4>
              </div>
              <div className="metric" style={{ color: 'var(--color-secondary)' }}>
                {quantumResult ? (quantumResult.var_value * 100).toFixed(2) + '%' : '--'}
              </div>
              <div className="meta">
                {quantumResult ? `Depth: ${quantumResult.depth}, Qubits: ${quantumResult.qubits}` : 'Waiting for run'}
              </div>
            </div>
          </div>

          <div className="glass-panel chart-panel">
            <h3>Distribution Visualization</h3>
            <div style={{ width: '100%', height: 300 }}>
              <ResponsiveContainer>
                <LineChart data={data}>
                  <XAxis dataKey="x" stroke="#888" />
                  <YAxis stroke="#888" />
                  <CartesianGrid strokeDasharray="3 3" stroke="#ddd" />
                  <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #ccc' }} />
                  <Line type="monotone" dataKey="y" stroke="#007a5e" strokeWidth={3} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
