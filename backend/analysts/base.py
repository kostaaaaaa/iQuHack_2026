# analysts/base.py
"""
Abstract Analyst base class for risk metric analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class AnalysisParams:
    """Common parameters for risk analysis."""
    confidence_level: float
    time_horizon: int = 1
    shots: int = 1000
    # Distribution parameters (optional)
    mu: Optional[float] = None
    sigma: Optional[float] = None
    x: Optional[np.ndarray] = None
    p: Optional[np.ndarray] = None


@dataclass
class ClassicalResult:
    """Result from classical Monte Carlo analysis."""
    value: float
    confidence_level: float
    samples: int
    execution_time: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantumResult:
    """Result from quantum amplitude estimation analysis."""
    value: float
    confidence_level: float
    depth: int
    qubits: int
    estimated_error: float
    oracle_calls: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Analyst(ABC):
    """
    Abstract base class for risk metric analysts.
    
    Each concrete analyst (VaR, CVaR, RVaR, EVaR) must implement
    methods for both classical and quantum analysis.
    """
    
    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Return the name of the risk metric (e.g., 'VaR', 'CVaR')."""
        pass
    
    @property
    @abstractmethod
    def full_name(self) -> str:
        """Return the full descriptive name of the metric."""
        pass
    
    @abstractmethod
    def classical_analysis(self, params: AnalysisParams) -> ClassicalResult:
        """
        Perform classical Monte Carlo analysis.
        
        Args:
            params: Analysis parameters including confidence level, shots, etc.
            
        Returns:
            ClassicalResult with the computed risk value and metadata.
        """
        pass
    
    @abstractmethod
    def quantum_analysis(self, params: AnalysisParams) -> QuantumResult:
        """
        Perform quantum amplitude estimation analysis.
        
        Args:
            params: Analysis parameters including confidence level, shots, etc.
            
        Returns:
            QuantumResult with the computed risk value and quantum circuit metadata.
        """
        pass
