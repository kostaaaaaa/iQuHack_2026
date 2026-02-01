from pydantic import BaseModel, Field
from typing import Optional

class VaRRequest(BaseModel):
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Confidence level for VaR (e.g., 0.95)")
    time_horizon: Optional[int] = Field(1, description="Time horizon in years")
    shots: Optional[int] = Field(1000, description="Number of samples or shots")
    risk_metric: Optional[str] = Field("VaR", description="Risk metric: VaR, CVaR, RVaR, or EVaR")

class ClassicalResult(BaseModel):
    var_value: float
    confidence_level: float
    samples: int
    execution_time: float

class QuantumResult(BaseModel):
    var_value: float
    confidence_level: float
    depth: int
    qubits: int
    estimated_error: float
