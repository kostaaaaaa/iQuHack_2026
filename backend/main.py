from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import VaRRequest, ClassicalResult, QuantumResult
from analysts import get_analyst, AnalysisParams, ANALYSTS

app = FastAPI(title="VaR Estimation API", description="API for Classical and Quantum Risk Estimation")

# CORS configuration to allow frontend integration
origins = [
    "http://localhost:5173",  # Vite default
    "http://localhost:5174",  # Vite alternate port
    "http://localhost:3000",  # React default
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Risk Estimation API"}

@app.get("/api/metrics")
async def list_metrics():
    """List available risk metrics."""
    return {
        "metrics": [
            {"key": analyst.metric_name, "name": analyst.full_name}
            for analyst in ANALYSTS.values()
        ]
    }

@app.post("/api/classical-var", response_model=ClassicalResult)
async def calculate_classical_var(request: VaRRequest):
    """
    Classical Monte Carlo risk estimation.
    Supports VaR, CVaR, RVaR, and EVaR based on risk_metric parameter.
    """
    metric = getattr(request, 'risk_metric', 'VaR') or 'VaR'
    
    try:
        analyst = get_analyst(metric)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    params = AnalysisParams(
        confidence_level=request.confidence_level,
        time_horizon=request.time_horizon or 1,
        shots=request.shots or 1000
    )
    
    result = analyst.classical_analysis(params)
    
    return ClassicalResult(
        var_value=result.value,
        confidence_level=result.confidence_level,
        samples=result.samples,
        execution_time=result.execution_time
    )

@app.post("/api/quantum-var", response_model=QuantumResult)
async def calculate_quantum_var(request: VaRRequest):
    """
    Quantum Amplitude Estimation risk estimation.
    Supports VaR, CVaR, RVaR, and EVaR based on risk_metric parameter.
    """
    metric = getattr(request, 'risk_metric', 'VaR') or 'VaR'
    
    try:
        analyst = get_analyst(metric)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    params = AnalysisParams(
        confidence_level=request.confidence_level,
        time_horizon=request.time_horizon or 1,
        shots=request.shots or 1000
    )
    
    result = analyst.quantum_analysis(params)
    
    return QuantumResult(
        var_value=result.value,
        confidence_level=result.confidence_level,
        depth=result.depth,
        qubits=result.qubits,
        estimated_error=result.estimated_error
    )
