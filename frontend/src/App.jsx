import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Area, ComposedChart } from 'recharts';
import { Activity, ShieldCheck, Zap } from 'lucide-react';
import './App.css';

const RISK_METRICS = [
  { key: 'VaR', label: 'VaR', fullName: 'Value at Risk' },
  { key: 'CVaR', label: 'CVaR', fullName: 'Conditional VaR' },
  { key: 'RVaR', label: 'RVaR', fullName: 'Range VaR' },
];

// Placeholder API calls - replace with real fetches later
const fetchClassicalResult = async (params) => {
  const response = await fetch('http://localhost:8000/api/classical-var', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  return response.json();
};

const fetchQuantumResult = async (params) => {
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
    time_horizon: 10,
    shots: 1000,
  });

  const [selectedMetrics, setSelectedMetrics] = useState([]);
  const [results, setResults] = useState({});
  const [loading, setLoading] = useState(false);

  const toggleMetric = (key) => {
    setSelectedMetrics(prev =>
      prev.includes(key)
        ? prev.filter(m => m !== key)
        : [...prev, key]
    );
  };

  const handleRun = async () => {
    if (selectedMetrics.length === 0) return;

    setLoading(true);
    try {
      const newResults = {};
      for (const metric of selectedMetrics) {
        const [classical, quantum] = await Promise.all([
          fetchClassicalResult({ ...params, risk_metric: metric }),
          fetchQuantumResult({ ...params, risk_metric: metric })
        ]);
        newResults[metric] = { classical, quantum };
      }
      setResults(newResults);
    } catch (error) {
      console.error("API Error", error);
    } finally {
      setLoading(false);
    }
  };

  // Mock data for visualization
  const data = [
    { x: -0.2, y: 0.01, tail: 0.01 },
    { x: -0.15, y: 0.05, tail: 0.05 },
    { x: -0.1, y: 0.15, tail: 0.15 },
    { x: -0.05, y: 0.35, tail: null },
    { x: 0, y: 0.6, tail: null },
    { x: 0.05, y: 0.35, tail: null },
    { x: 0.1, y: 0.15, tail: null },
    { x: 0.15, y: 0.05, tail: null },
    { x: 0.2, y: 0.01, tail: null },
  ];

  return (
    <div className="container">
      <header className="header">
        <h1 className="title text-gradient">
          <Zap size={32} style={{ marginBottom: -6, marginRight: 8 }} />
          Quantum Risk Estimation
        </h1>
        <p className="subtitle">Classical Monte Carlo vs. Quantum Amplitude Estimation</p>
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
            <label>Time Horizon (Days)</label>
            <input
              type="number"
              min="1"
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

          <div className="input-group">
            <label>Risk Metrics</label>
            <div className="checkbox-group">
              {RISK_METRICS.map(({ key, label, fullName }) => (
                <label key={key} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={selectedMetrics.includes(key)}
                    onChange={() => toggleMetric(key)}
                  />
                  <span className="checkbox-text">{label}</span>
                  <span className="checkbox-hint">{fullName}</span>
                </label>
              ))}
            </div>
          </div>

          <button
            className="btn"
            onClick={handleRun}
            disabled={loading || selectedMetrics.length === 0}
          >
            {loading ? 'Running...' : `Calculate (${selectedMetrics.length})`}
          </button>
        </aside>

        <section className="results-area">
          {selectedMetrics.length === 0 ? (
            <div className="glass-panel empty-state">
              <p>Select at least one risk metric to calculate.</p>
            </div>
          ) : (
            <>
              {/* Column Headers */}
              <div className="results-columns-header">
                <div className="column-header">
                  <Activity size={20} color="var(--color-primary)" />
                  <h3>Classical MC</h3>
                </div>
                <div className="column-header quantum">
                  <ShieldCheck size={20} color="var(--color-secondary)" />
                  <h3>Quantum AE</h3>
                </div>
              </div>

              {selectedMetrics.map(metric => {
                const metricInfo = RISK_METRICS.find(m => m.key === metric);
                const result = results[metric];

                return (
                  <div key={metric} className="metric-section">
                    <div className="metric-label">{metricInfo.fullName} ({metric})</div>
                    <div className="metric-row">
                      <div className="glass-panel result-card">
                        <div className="metric">
                          {result?.classical ? (result.classical.var_value * 100).toFixed(2) + '%' : '--'}
                        </div>
                        <div className="meta">
                          {result?.classical ? `${result.classical.samples} samples` : 'Waiting'}
                        </div>
                      </div>

                      <div className="glass-panel result-card quantum">
                        <div className="metric" style={{ color: 'var(--color-secondary)' }}>
                          {result?.quantum ? (result.quantum.var_value * 100).toFixed(2) + '%' : '--'}
                        </div>
                        <div className="meta">
                          {result?.quantum ? `D:${result.quantum.depth} Q:${result.quantum.qubits}` : 'Waiting'}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </>
          )}

          <div className="glass-panel chart-panel">
            <h3>Distribution Visualization</h3>
            <div style={{ width: '100%', height: 250 }}>
              <ResponsiveContainer>
                <ComposedChart data={data}>
                  <defs>
                    <pattern id="tailPattern" patternUnits="userSpaceOnUse" width="4" height="4">
                      <path d="M-1,1 l2,-2 M0,4 l4,-4 M3,5 l2,-2" stroke="#e74c3c" strokeWidth="1" />
                    </pattern>
                  </defs>
                  <XAxis dataKey="x" stroke="#888" />
                  <YAxis stroke="#888" />
                  <CartesianGrid strokeDasharray="3 3" stroke="#ddd" />
                  <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #ccc' }} />
                  <Area type="monotone" dataKey="tail" fill="url(#tailPattern)" stroke="none" />
                  <ReferenceLine x={-0.1} stroke="#e74c3c" strokeWidth={2} strokeDasharray="5 5" label={{ value: 'α=5%', position: 'top', fill: '#e74c3c', fontSize: 12 }} />
                  <Line type="monotone" dataKey="y" stroke="#007a5e" strokeWidth={3} dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
