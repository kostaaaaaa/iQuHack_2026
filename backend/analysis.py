# analysis.py
"""
Analysis utilities for VaR / CVaR / RVaR / EVaR experiments (classical + quantum IQAE).

Design goals:
- You can log many runs (different alpha, epsilon, num_qubits, distributions, etc.)
- You can compare classical vs quantum VaR search results
- You can plot distribution/CDF with risk markers
- You can analyze error (absolute/relative), confidence intervals, and scaling trends
- You can export a compact "menu" of plots to choose from for reports/presentations

Usage sketch (in your main script):
-------------------------------
from analysis import RiskAnalysis, RunRecord

an = RiskAnalysis()

# After you compute x, p and obtain `risk` dict from value_at_risk(...)
# and optionally IQAE metadata per step:
rec = RunRecord(
    tag="iqae_run_1",
    method="quantum",
    alpha=ALPHA,
    beta=risk["beta"],
    num_qubits=num_qubits,
    mu=mu,
    sigma=sigma,
    distribution="lognormal",
    x=x,
    p=p,
    risk=risk,
    var_index=risk.get("VaR_index"),
    classical_risk=classical_risk_if_you_have_it,
    quantum_steps=steps_list_if_you_collect_it,  # optional
    notes="anything"
)
an.add_run(rec)

an.plot_distribution_with_markers(tag="iqae_run_1", save=False)
an.plot_cdf_with_var(tag="iqae_run_1", save=False)
an.plot_risk_summary(tag="iqae_run_1", save=False)
an.plot_error_report(tag="iqae_run_1", save=False)

# For multiple runs:
an.plot_scaling_queries_vs_precision(save=False)
an.plot_error_vs_alpha(save=False)
an.print_report()
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import json
import math

import numpy as np
import matplotlib.pyplot as plt


Number = Union[int, float]


# ----------------------------
# Helpers
# ----------------------------
def _ensure_np(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _safe_float(v, default=np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _mkdir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _nan_if_missing(d: Dict[str, Any], key: str) -> float:
    return _safe_float(d.get(key, np.nan))


def _format_pct(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{100*x:.2f}%"


def _maybe_savefig(save: bool, outdir: str, fname: str, dpi: int = 200) -> None:
    if save:
        _mkdir(outdir)
        path = os.path.join(outdir, fname)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")


# ----------------------------
# Data containers
# ----------------------------
@dataclass
class QuantumStep:
    """
    Optional: store per-step IQAE / estimation data if you want convergence plots.

    Typical fields to record per step:
    - index tested, threshold v, measured alpha estimate, confidence interval, oracle calls, shots, etc.
    """
    index: Optional[int] = None
    v: Optional[float] = None
    alpha_hat: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    epsilon: Optional[float] = None
    alpha_conf: Optional[float] = None
    shots: Optional[int] = None
    oracle_calls: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    """
    One experiment run (one distribution + alpha/beta + method).
    You can store both the result risk dict and optional comparisons.
    """
    tag: str
    method: str  # "classical" or "quantum" (or any label you want)

    # key parameters
    alpha: float
    beta: Optional[float] = None
    num_qubits: Optional[int] = None
    mu: Optional[float] = None
    sigma: Optional[float] = None
    distribution: str = "unknown"

    # the discretized distribution
    x: Optional[np.ndarray] = None
    p: Optional[np.ndarray] = None

    # outputs
    risk: Dict[str, Any] = field(default_factory=dict)          # VaR/CVaR/RVaR/EVaR etc.
    var_index: Optional[int] = None

    # comparisons
    classical_risk: Optional[Dict[str, Any]] = None             # if you also computed classical baseline
    quantum_steps: Optional[List[QuantumStep]] = None           # optional convergence info

    # optional: keep raw config blobs
    iqae_config: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# ----------------------------
# Main analysis object
# ----------------------------
class RiskAnalysis:
    def __init__(self, outdir: str = "figures", style: Optional[Dict[str, Any]] = None):
        self.outdir = outdir
        self.runs: Dict[str, RunRecord] = {}
        self.style = style or {}

    # ---------- run management ----------
    def add_run(self, rec: RunRecord) -> None:
        if rec.x is not None:
            rec.x = _ensure_np(rec.x)
        if rec.p is not None:
            p = _ensure_np(rec.p)
            s = float(np.sum(p))
            rec.p = p / s if s != 0 else p
        self.runs[rec.tag] = rec

    def get(self, tag: str) -> RunRecord:
        if tag not in self.runs:
            raise KeyError(f"No run with tag '{tag}'. Available: {list(self.runs.keys())}")
        return self.runs[tag]

    def list_tags(self) -> List[str]:
        return list(self.runs.keys())

    # ---------- basic computed metrics ----------
    def _cdf(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.cumsum(p)

    def _risk_value(self, rec: RunRecord, key: str) -> float:
        return _nan_if_missing(rec.risk, key)

    def _baseline_value(self, rec: RunRecord, key: str) -> float:
        if rec.classical_risk is None:
            return np.nan
        return _nan_if_missing(rec.classical_risk, key)

    def compute_error_table(self, tag: str) -> Dict[str, float]:
        """
        Compare run risk values to the classical baseline (if provided).
        Returns absolute/relative error for VaR/CVaR/RVaR/EVaR and alpha-vs-measured if available.
        """
        rec = self.get(tag)
        out: Dict[str, float] = {}

        for metric in ["VaR", "CVaR", "RVaR", "EVaR"]:
            v = self._risk_value(rec, metric)
            b = self._baseline_value(rec, metric)
            out[f"{metric}_abs_err"] = abs(v - b) if np.isfinite(v) and np.isfinite(b) else np.nan
            out[f"{metric}_rel_err"] = abs(v - b) / abs(b) if np.isfinite(v) and np.isfinite(b) and b != 0 else np.nan

        # If you stored final alpha_hat somewhere:
        # (You can store it in rec.risk["alpha_hat"] if you like)
        alpha_hat = _nan_if_missing(rec.risk, "alpha_hat")
        if np.isfinite(alpha_hat):
            out["alpha_hat_abs_err"] = abs(alpha_hat - rec.alpha)
            out["alpha_hat_rel_err"] = abs(alpha_hat - rec.alpha) / rec.alpha if rec.alpha != 0 else np.nan

        return out

    # ---------- plotting: distribution / cdf ----------
    def plot_distribution_with_markers(
        self,
        tag: str,
        save: bool = False,
        fname: Optional[str] = None,
        show_markers: bool = True,
    ) -> None:
        rec = self.get(tag)
        if rec.x is None or rec.p is None:
            raise ValueError("RunRecord must include x and p to plot distribution.")

        x, p = rec.x, rec.p
        plt.figure()
        plt.plot(x, p, marker="o", linewidth=1)

        plt.xlabel("Asset Value")
        plt.ylabel("Probability")
        plt.title(f"Distribution ({rec.distribution}) | tag={rec.tag} | method={rec.method}")

        if show_markers:
            self._add_risk_markers(rec)

        plt.grid(True)
        plt.legend(loc="best")

        if fname is None:
            fname = f"dist_{tag}.png"
        _maybe_savefig(save, self.outdir, fname)
        plt.show()

    def plot_cdf_with_var(
        self,
        tag: str,
        save: bool = False,
        fname: Optional[str] = None,
        show_markers: bool = True,
    ) -> None:
        rec = self.get(tag)
        if rec.x is None or rec.p is None:
            raise ValueError("RunRecord must include x and p to plot CDF.")

        x, p = rec.x, rec.p
        cdf = self._cdf(x, p)

        plt.figure()
        plt.plot(x, cdf, marker="o", linewidth=1)
        plt.axhline(rec.alpha, linestyle="--", linewidth=1, label=f"alpha={rec.alpha}")

        if show_markers:
            var = _nan_if_missing(rec.risk, "VaR")
            if np.isfinite(var):
                plt.axvline(var, linestyle="--", linewidth=1, label=f"VaR={var:.6g}")

        plt.xlabel("Asset Value")
        plt.ylabel("CDF  P(X ≤ x)")
        plt.title(f"CDF | tag={rec.tag} | method={rec.method}")
        plt.grid(True)
        plt.legend(loc="best")

        if fname is None:
            fname = f"cdf_{tag}.png"
        _maybe_savefig(save, self.outdir, fname)
        plt.show()

    def _add_risk_markers(self, rec: RunRecord) -> None:
        # Add vertical lines for any risk metrics available
        markers = [
            ("VaR", "--"),
            ("CVaR", ":"),
            ("RVaR", "-."),
            ("EVaR", (0, (3, 1, 1, 1))),  # dash-dot-dot style
        ]
        for key, ls in markers:
            v = _nan_if_missing(rec.risk, key)
            if np.isfinite(v):
                plt.axvline(v, linestyle=ls, linewidth=1, label=f"{key}={v:.6g}")

    # ---------- plotting: risk summary ----------
    def plot_risk_summary(self, tag: str, save: bool = False, fname: Optional[str] = None) -> None:
        """
        Bar plot comparing the risk measures for this run,
        and optionally against classical baseline (if provided).
        """
        rec = self.get(tag)
        metrics = ["VaR", "CVaR", "RVaR", "EVaR"]

        vals = np.array([_nan_if_missing(rec.risk, m) for m in metrics], dtype=float)
        base = None
        if rec.classical_risk is not None:
            base = np.array([_nan_if_missing(rec.classical_risk, m) for m in metrics], dtype=float)

        xloc = np.arange(len(metrics))
        width = 0.35

        plt.figure()
        plt.bar(xloc - (width / 2 if base is not None else 0), vals, width=width, label=rec.method)

        if base is not None:
            plt.bar(xloc + width / 2, base, width=width, label="classical_baseline")

        plt.xticks(xloc, metrics)
        plt.ylabel("Value")
        plt.title(f"Risk Summary | tag={rec.tag}")
        plt.grid(True, axis="y")
        plt.legend(loc="best")

        if fname is None:
            fname = f"risk_summary_{tag}.png"
        _maybe_savefig(save, self.outdir, fname)
        plt.show()

    # ---------- plotting: quantum convergence (optional) ----------
    def plot_quantum_convergence(
        self,
        tag: str,
        save: bool = False,
        fname: Optional[str] = None,
    ) -> None:
        """
        Plots alpha_hat with confidence intervals over steps, if you logged QuantumStep entries.
        """
        rec = self.get(tag)
        if not rec.quantum_steps:
            raise ValueError("No quantum_steps recorded for this run. Populate RunRecord.quantum_steps.")

        steps = rec.quantum_steps
        t = np.arange(len(steps))
        alpha_hat = np.array([_safe_float(s.alpha_hat) for s in steps], dtype=float)
        lo = np.array([_safe_float(s.ci_low) for s in steps], dtype=float)
        hi = np.array([_safe_float(s.ci_high) for s in steps], dtype=float)

        plt.figure()
        plt.plot(t, alpha_hat, marker="o", linewidth=1, label="alpha_hat")
        if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
            # Error bars if present; only plot for finite entries
            yerr_low = alpha_hat - lo
            yerr_high = hi - alpha_hat
            yerr = np.vstack([yerr_low, yerr_high])
            plt.errorbar(t, alpha_hat, yerr=yerr, fmt="none", capsize=3)

        plt.axhline(rec.alpha, linestyle="--", linewidth=1, label=f"target alpha={rec.alpha}")
        plt.xlabel("Step")
        plt.ylabel("Estimated alpha")
        plt.title(f"Quantum Convergence | tag={rec.tag}")
        plt.grid(True)
        plt.legend(loc="best")

        if fname is None:
            fname = f"q_convergence_{tag}.png"
        _maybe_savefig(save, self.outdir, fname)
        plt.show()

    # ---------- plotting: error report ----------
    def plot_error_report(self, tag: str, save: bool = False, fname: Optional[str] = None) -> None:
        """
        If a classical baseline exists, plots absolute error of VaR/CVaR/RVaR/EVaR.
        """
        rec = self.get(tag)
        if rec.classical_risk is None:
            raise ValueError("No classical_risk baseline attached to this run.")

        metrics = ["VaR", "CVaR", "RVaR", "EVaR"]
        abs_err = np.array([abs(_nan_if_missing(rec.risk, m) - _nan_if_missing(rec.classical_risk, m)) for m in metrics])
        rel_err = np.array([
            (abs(_nan_if_missing(rec.risk, m) - _nan_if_missing(rec.classical_risk, m)) / abs(_nan_if_missing(rec.classical_risk, m)))
            if np.isfinite(_nan_if_missing(rec.classical_risk, m)) and _nan_if_missing(rec.classical_risk, m) != 0 else np.nan
            for m in metrics
        ])

        xloc = np.arange(len(metrics))
        plt.figure()
        plt.bar(xloc, abs_err)
        plt.xticks(xloc, metrics)
        plt.ylabel("Absolute Error")
        plt.title(f"Absolute Error vs Classical | tag={rec.tag}")
        plt.grid(True, axis="y")

        if fname is None:
            fname = f"abs_error_{tag}.png"
        _maybe_savefig(save, self.outdir, fname)
        plt.show()

        plt.figure()
        plt.bar(xloc, rel_err)
        plt.xticks(xloc, metrics)
        plt.ylabel("Relative Error")
        plt.title(f"Relative Error vs Classical | tag={rec.tag}")
        plt.grid(True, axis="y")

        if fname is None:
            fname = f"rel_error_{tag}.png"
        _maybe_savefig(save, self.outdir, fname.replace("abs_", "rel_"))
        plt.show()

    # ---------- multi-run comparisons ----------
    def plot_error_vs_alpha(self, save: bool = False, fname: str = "error_vs_alpha.png") -> None:
        """
        Scatter of |VaR_error| (and others) vs alpha across runs that have classical baseline.
        """
        tags = [t for t, r in self.runs.items() if r.classical_risk is not None]
        if not tags:
            raise ValueError("No runs with classical_risk baseline to compare.")

        alphas = np.array([self.runs[t].alpha for t in tags], dtype=float)

        plt.figure()
        for metric in ["VaR", "CVaR", "RVaR", "EVaR"]:
            errs = []
            for t in tags:
                r = self.runs[t]
                v = _nan_if_missing(r.risk, metric)
                b = _nan_if_missing(r.classical_risk, metric)
                errs.append(abs(v - b) if np.isfinite(v) and np.isfinite(b) else np.nan)
            errs = np.array(errs, dtype=float)
            plt.plot(alphas, errs, marker="o", linestyle="none", label=f"|{metric} err|")

        plt.xlabel("alpha")
        plt.ylabel("Absolute error")
        plt.title("Error vs alpha (runs with classical baseline)")
        plt.grid(True)
        plt.legend(loc="best")
        _maybe_savefig(save, self.outdir, fname)
        plt.show()

    def plot_scaling_queries_vs_precision(
        self,
        key_precision: str = "epsilon",
        key_queries: str = "oracle_calls",
        save: bool = False,
        fname: str = "scaling_queries_vs_precision.png",
    ) -> None:
        """
        Generic scaling plot: queries vs precision (or any knobs).
        - If you logged quantum_steps, we try to pull final step's oracle_calls and epsilon.
        - Otherwise we look for rec.risk[...] or rec.iqae_config[...] keys.

        Recommended: store into rec.risk:
            rec.risk["epsilon"] = epsilon_used
            rec.risk["oracle_calls"] = total_oracle_calls
        """
        xs, ys = [], []
        for rec in self.runs.values():
            # Only meaningful for quantum runs, but we don't enforce it.
            prec = np.nan
            queries = np.nan

            # Try quantum_steps last entry
            if rec.quantum_steps:
                last = rec.quantum_steps[-1]
                prec = _safe_float(getattr(last, key_precision, np.nan))
                queries = _safe_float(getattr(last, key_queries, np.nan))

            # fallback to risk dict / iqae_config
            if not np.isfinite(prec):
                prec = _nan_if_missing(rec.risk, key_precision)
            if not np.isfinite(queries):
                queries = _nan_if_missing(rec.risk, key_queries)

            if np.isfinite(prec) and np.isfinite(queries):
                xs.append(prec)
                ys.append(queries)

        if not xs:
            raise ValueError(
                "No runs contained usable scaling data. "
                "Populate RunRecord.quantum_steps (with epsilon/oracle_calls) or "
                "store rec.risk['epsilon'], rec.risk['oracle_calls']."
            )

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)

        plt.figure()
        plt.plot(xs, ys, marker="o", linestyle="none")
        plt.xlabel(key_precision)
        plt.ylabel(key_queries)
        plt.title(f"Scaling: {key_queries} vs {key_precision}")
        plt.grid(True)
        _maybe_savefig(save, self.outdir, fname)
        plt.show()

    # ---------- reporting / exporting ----------
    def print_report(self) -> None:
        """
        Prints a concise report of all runs: parameters + risk + (optional) errors.
        """
        print("\n================ RISK ANALYSIS REPORT ================\n")
        for tag, rec in self.runs.items():
            print(f"tag: {tag}")
            print(f"  method: {rec.method}")
            print(f"  alpha: {rec.alpha} ({_format_pct(rec.alpha)})  beta: {rec.beta}")
            print(f"  num_qubits: {rec.num_qubits}  dist: {rec.distribution}  mu: {rec.mu}  sigma: {rec.sigma}")
            print("  risk:")
            for k in ["VaR", "CVaR", "RVaR", "EVaR"]:
                v = _nan_if_missing(rec.risk, k)
                print(f"    {k:5s}: {v:.8g}" if np.isfinite(v) else f"    {k:5s}: nan")
            if rec.classical_risk is not None:
                errs = self.compute_error_table(tag)
                print("  errors vs classical:")
                for k in ["VaR_abs_err", "CVaR_abs_err", "RVaR_abs_err", "EVaR_abs_err"]:
                    v = errs.get(k, np.nan)
                    print(f"    {k:12s}: {v:.6g}" if np.isfinite(v) else f"    {k:12s}: nan")
            if rec.notes:
                print(f"  notes: {rec.notes}")
            print("------------------------------------------------------")

    def save_json(self, path: str) -> None:
        """
        Save the run metadata and risk outputs (not the full x,p arrays by default).
        """
        payload = {}
        for tag, rec in self.runs.items():
            d = asdict(rec)
            # avoid dumping potentially huge arrays unless you want them
            d["x"] = None
            d["p"] = None
            payload[tag] = d
        _mkdir(os.path.dirname(path) or ".")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    # ---------- convenience: "one-call" figure suite ----------
    def plot_all_for_run(self, tag: str, save: bool = False) -> None:
        """
        Makes a nice suite of figures for a single run.
        """
        self.plot_distribution_with_markers(tag, save=save, fname=f"dist_{tag}.png")
        self.plot_cdf_with_var(tag, save=save, fname=f"cdf_{tag}.png")
        self.plot_risk_summary(tag, save=save, fname=f"risk_summary_{tag}.png")
        rec = self.get(tag)
        if rec.classical_risk is not None:
            self.plot_error_report(tag, save=save)
        if rec.quantum_steps:
            self.plot_quantum_convergence(tag, save=save, fname=f"q_conv_{tag}.png")
