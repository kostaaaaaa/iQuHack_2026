from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import VaRRequest, ClassicalResult, QuantumResult

app = FastAPI(title="VaR Estimation API", description="API for Classical and Quantum VaR Estimation")

# CORS configuration to allow frontend integration
origins = [
    "http://localhost:5173",  # Vite default
    "http://localhost:3000",  # React default
    "http://127.0.0.1:5173",
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
    return {"message": "Welcome to the VaR Estimation API"}

@app.post("/api/classical-var", response_model=ClassicalResult)
async def calculate_classical_var(request: VaRRequest):
    """
    Placeholder endpoint for Classical Monte Carlo VaR estimation.
    """
    # TODO: Implement actual logic from notebooks
    return ClassicalResult(
        var_value=0.15,
        confidence_level=request.confidence_level,
        samples=request.shots or 1000,
        execution_time=0.05
    )

@app.post("/api/quantum-var", response_model=QuantumResult)
async def calculate_quantum_var(request: VaRRequest):
    """
    Placeholder endpoint for Quantum Amplitude Estimation VaR.
    """
    # TODO: Implement actual Classiq SDK logic
    return QuantumResult(
        var_value=0.148,
        confidence_level=request.confidence_level,
        depth=50,
        qubits=10,
        estimated_error=0.002
    )
