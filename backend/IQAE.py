"""
Risk metrics with Classiq IQAE:
- VaR (quantile) via quantum CDF + binary search
- CVaR via tail probability and tail expectation
- RVaR (Range VaR) via band expectation between VaR_alpha and VaR_beta
- EVaR via quantum MGF estimation E[e^{tL}] and classical minimization over t

Requires:
  pip install classiq
Optional for circuit/hardware metrics:
  pip install "classiq[analyzer_sdk]"

Docs references:
- VaR + IQAE pattern (distribution load + comparator payoff + IQAE.run(epsilon, alpha))
- Amplitude loading with assign_amplitude_table
- Analyzer hardware comparison table for depth/gate metrics
"""



import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- Classiq imports ---
from classiq import (
    qfunc,
    qperm,
    QBit,
    QNum,
    QArray,
    Const,
    inplace_prepare_state,
    Constraints,
    Preferences,
    show,
)
from classiq.applications.iqae.iqae import IQAE

# Amplitude-loading primitive: |i>|0> -> a(i)|i>|1> + sqrt(1-a(i)^2)|i>|0>
# (expects indicator initialized to |0>)
from classiq.open_library.functions.amplitude_loading import assign_amplitude_table



# ----------------------------
# Utility: classical baselines
# ----------------------------

def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    s = probs.sum()
    if s <= 0:
        raise ValueError("Probability array must sum to positive value.")
    return probs / s


def classical_var(losses: np.ndarray, probs: np.ndarray, alpha: float) -> float:
    """VaR_alpha defined by P(L <= VaR_alpha) >= alpha (alpha-quantile)."""
    idx = classical_var_index(losses, probs, alpha)
    return float(losses[idx])


def classical_var_index(losses: np.ndarray, probs: np.ndarray, alpha: float) -> int:
    losses = np.asarray(losses, dtype=float)
    probs = normalize_probs(probs)
    # assume losses already sorted ascending; if not, sort jointly
    order = np.argsort(losses)
    losses_s = losses[order]
    probs_s = probs[order]
    cdf = np.cumsum(probs_s)
    idx_s = int(np.searchsorted(cdf, alpha, side="left"))
    idx_s = min(max(idx_s, 0), len(losses_s) - 1)
    # map back to original index in sorted space (we use sorted indices everywhere later anyway)
    return int(idx_s)


def classical_cvar(losses: np.ndarray, probs: np.ndarray, alpha: float) -> float:
    """
    CVaR_alpha for losses, using upper tail:
      CVaR_alpha = E[L | L >= VaR_alpha]
    (alpha close to 1)
    """
    losses = np.asarray(losses, dtype=float)
    probs = normalize_probs(probs)
    order = np.argsort(losses)
    L = losses[order]
    p = probs[order]
    var_a = classical_var(L, p, alpha)
    tail = (L >= var_a)
    tail_prob = p[tail].sum()
    if tail_prob <= 0:
        return float("nan")
    return float((L[tail] * p[tail]).sum() / tail_prob)


def classical_rvar(losses: np.ndarray, probs: np.ndarray, alpha: float, beta: float) -> float:
    """
    Range VaR (RVaR) between quantiles alpha and beta:
      RVaR_{alpha,beta} = E[L | VaR_alpha <= L <= VaR_beta]
    (alpha < beta, typically both close to 1)
    """
    if not (0 < alpha < beta < 1):
        raise ValueError("Require 0 < alpha < beta < 1.")
    losses = np.asarray(losses, dtype=float)
    probs = normalize_probs(probs)
    order = np.argsort(losses)
    L = losses[order]
    p = probs[order]
    va = classical_var(L, p, alpha)
    vb = classical_var(L, p, beta)
    band = (L >= va) & (L <= vb)
    band_prob = p[band].sum()
    if band_prob <= 0:
        return float("nan")
    return float((L[band] * p[band]).sum() / band_prob)


def classical_evar(losses: np.ndarray, probs: np.ndarray, alpha: float, t_grid: np.ndarray) -> Tuple[float, float]:
    """
    EVaR_alpha(L) = inf_{t>0} (1/t) * ( log E[e^{tL}] - log(1-alpha) ).
    Returns (best_evar, best_t) evaluated on provided t_grid.
    """
    losses = np.asarray(losses, dtype=float)
    probs = normalize_probs(probs)
    order = np.argsort(losses)
    L = losses[order]
    p = probs[order]
    one_minus_alpha = 1.0 - alpha
    if one_minus_alpha <= 0:
        return float("inf"), float("nan")

    best = float("inf")
    best_t = float("nan")
    for t in t_grid:
        if t <= 0:
            continue
        mgf = float(np.sum(p * np.exp(t * L)))
        val = (math.log(mgf) - math.log(one_minus_alpha)) / t
        if val < best:
            best = val
            best_t = float(t)
    return best, best_t


# -----------------------------------------
# Quantum estimation wrappers (IQAE + Classiq)
# -----------------------------------------

@dataclass
class IQAEResult:
    estimation: float
    ci_low: float
    ci_high: float
    elapsed_s: float
    extras: Dict[str, Any]


# Global index used in qperm comparator payoffs (as in docs pattern)
GLOBAL_INDEX: int = 0


def _extract_iqae_metadata(iqae_res: Any) -> Dict[str, Any]:
    """
    IQAE result objects can evolve across versions.
    We try to extract anything useful without breaking if fields are absent.
    """
    extras: Dict[str, Any] = {}
    for k in [
        "num_iterations",
        "num_oracle_queries",
        "epsilon",
        "alpha",
        "confidence_interval",
        "shots",
    ]:
        if hasattr(iqae_res, k):
            extras[k] = getattr(iqae_res, k)
    return extras


def estimate_cdf_prob_quantum(
    probs: List[float],
    num_qubits: int,
    threshold_index: int,
    epsilon: float,
    alpha_conf: float,
    constraints: Optional[Constraints] = None,
    preferences: Optional[Preferences] = None,
    verbose_show_once: bool = False,
) -> IQAEResult:
    """
    Estimates P(asset < threshold_index) where asset is distributed as probs over {0..2^n-1}.
    This mirrors the VaR tutorial approach (distribution loading + comparator payoff + IQAE). 
    """
    global GLOBAL_INDEX
    GLOBAL_INDEX = int(threshold_index)

    # --- QMOD ---
    @qfunc(synthesize_separately=True)
    def state_preparation(asset: QArray[QBit], ind: QBit):
        load_distribution(asset=asset)
        payoff(asset=asset, ind=ind)

    @qfunc
    def load_distribution(asset: QNum):
        inplace_prepare_state(probs, bound=0, target=asset)

    @qperm
    def payoff(asset: Const[QNum], ind: QBit):
        # Mark states with asset < GLOBAL_INDEX
        ind ^= asset < GLOBAL_INDEX

    # --- IQAE wrapper ---
    iqae = IQAE(
        state_prep_op=state_preparation,
        problem_vars_size=num_qubits,
        constraints=constraints,
        preferences=preferences,
    )

    if verbose_show_once:
        qprog = iqae.get_qprog()
        show(qprog)

    t0 = time.time()
    iqae_res = iqae.run(epsilon=epsilon, alpha=alpha_conf)
    t1 = time.time()

    est = float(iqae_res.estimation)
    ci = list(iqae_res.confidence_interval)
    meta = _extract_iqae_metadata(iqae_res)

    return IQAEResult(estimation=est, ci_low=float(ci[0]), ci_high=float(ci[1]), elapsed_s=t1 - t0, extras=meta)


def estimate_expectation_quantum_via_amplitude_loading(
    probs: List[float],
    num_qubits: int,
    amplitudes: List[float],
    epsilon: float,
    alpha_conf: float,
    constraints: Optional[Constraints] = None,
    preferences: Optional[Preferences] = None,
    verbose_show_once: bool = False,
) -> IQAEResult:
    """
    Estimates E[f] where f(i) in [0,1], by encoding amplitude sqrt(f(i)) on indicator using
    assign_amplitude_table.

    State prepared is:
      sum_i sqrt(p_i) |i> ( sqrt(f_i)|1> + sqrt(1-f_i)|0> )
    so probability of |1> equals E[f]. (Same QMCI/IQAE framework.)
    """
    if len(amplitudes) != 2**num_qubits:
        raise ValueError(f"amplitudes length must be 2^n (= {2**num_qubits}).")

    # --- QMOD ---
    @qfunc(synthesize_separately=True)
    def state_preparation(asset: QArray[QBit], ind: QBit):
        load_distribution(asset=asset)
        load_payoff(asset=asset, ind=ind)

    @qfunc
    def load_distribution(asset: QNum):
        inplace_prepare_state(probs, bound=0, target=asset)

    @qfunc
    def load_payoff(asset: QNum, ind: QBit):
        # a(i) = sqrt(f_i) must be in [0,1]
        assign_amplitude_table(amplitudes=amplitudes, index=asset, indicator=ind)

    iqae = IQAE(
        state_prep_op=state_preparation,
        problem_vars_size=num_qubits,
        constraints=constraints,
        preferences=preferences,
    )

    if verbose_show_once:
        qprog = iqae.get_qprog()
        show(qprog)

    t0 = time.time()
    iqae_res = iqae.run(epsilon=epsilon, alpha=alpha_conf)
    t1 = time.time()

    est = float(iqae_res.estimation)
    ci = list(iqae_res.confidence_interval)
    meta = _extract_iqae_metadata(iqae_res)

    return IQAEResult(estimation=est, ci_low=float(ci[0]), ci_high=float(ci[1]), elapsed_s=t1 - t0, extras=meta)


# -----------------------------------------
# VaR via quantum CDF + binary search
# -----------------------------------------

@dataclass
class VaRQuantumReport:
    var_value: float
    var_index: int
    outer_iters: int
    last_cdf_est: IQAEResult


def quantum_var_via_binary_search(
    losses_sorted: np.ndarray,
    probs_sorted: np.ndarray,
    alpha: float,
    epsilon_iqae: float = 0.01,
    alpha_conf_iqae: float = 0.01,
    constraints: Optional[Constraints] = None,
    preferences: Optional[Preferences] = None,
    show_first_circuit: bool = True,
) -> VaRQuantumReport:
    """
    losses_sorted: length 2^n, ascending.
    probs_sorted: length 2^n, sums to 1.
    Finds smallest index k such that CDF(k) >= alpha using quantum CDF queries.
    """
    N = len(losses_sorted)
    n = int(round(math.log2(N)))
    if 2**n != N:
        raise ValueError("Need len(losses)=2^n for straightforward indexing.")

    probs_list = probs_sorted.tolist()

    lo, hi = 0, N - 1
    outer_iters = 0
    last = None

    # We query CDF at "mid+1" boundary in terms of 'asset < GLOBAL_INDEX'.
    # To ask P(index <= mid) we can ask P(index < mid+1).
    while lo < hi:
        outer_iters += 1
        mid = (lo + hi) // 2
        thresh = mid + 1

        last = estimate_cdf_prob_quantum(
            probs=probs_list,
            num_qubits=n,
            threshold_index=thresh,
            epsilon=epsilon_iqae,
            alpha_conf=alpha_conf_iqae,
            constraints=constraints,
            preferences=preferences,
            verbose_show_once=(show_first_circuit and outer_iters == 1),
        )

        cdf_est = last.estimation

        # binary search decision
        if cdf_est >= alpha:
            hi = mid
        else:
            lo = mid + 1

        # Safety stop (shouldn't happen)
        if outer_iters > (n + 5):
            break

    var_idx = int(lo)
    var_val = float(losses_sorted[var_idx])
    if last is None:
        # should not happen, but keep robust
        last = IQAEResult(estimation=float("nan"), ci_low=float("nan"), ci_high=float("nan"), elapsed_s=0.0, extras={})
    return VaRQuantumReport(var_value=var_val, var_index=var_idx, outer_iters=outer_iters, last_cdf_est=last)


# -----------------------------------------
# CVaR / RVaR / EVaR using IQAE expectations
# -----------------------------------------

@dataclass
class RiskQuantumResults:
    var: float
    cvar: float
    rvar: float
    evar: float
    details: Dict[str, Any]


def compute_risk_measures_quantum(
    losses: np.ndarray,
    probs: np.ndarray,
    alpha: float = 0.95,
    beta: float = 0.99,
    # IQAE knobs:
    epsilon_cdf: float = 0.01,
    epsilon_exp: float = 0.01,
    alpha_conf_iqae: float = 0.01,
    # EVaR search:
    t_grid: Optional[np.ndarray] = None,
    constraints: Optional[Constraints] = None,
    preferences: Optional[Preferences] = None,
    show_first_circuit: bool = True,
) -> RiskQuantumResults:
    """
    Computes VaR_alpha, CVaR_alpha, RVaR_{alpha,beta}, EVaR_alpha using Classiq IQAE.
    Returns quantum estimates + a rich details dict.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1).")
    if not (0 < beta < 1) or not (alpha < beta):
        raise ValueError("require 0 < alpha < beta < 1 for RVaR.")

    losses = np.asarray(losses, dtype=float)
    probs = normalize_probs(np.asarray(probs, dtype=float))

    # Sort by loss (required for quantiles)
    order = np.argsort(losses)
    L = losses[order]
    p = probs[order]

    N = len(L)
    n = int(round(math.log2(N)))
    if 2**n != N:
        raise ValueError("This script assumes len(losses)=2^n for direct indexing.")

    probs_list = p.tolist()

    # --- VaR(alpha) via quantum binary search ---
    var_rep = quantum_var_via_binary_search(
        losses_sorted=L,
        probs_sorted=p,
        alpha=alpha,
        epsilon_iqae=epsilon_cdf,
        alpha_conf_iqae=alpha_conf_iqae,
        constraints=constraints,
        preferences=preferences,
        show_first_circuit=show_first_circuit,
    )
    var_a = var_rep.var_value
    idx_a = var_rep.var_index

    # --- VaR(beta) for RVaR band ---
    varb_rep = quantum_var_via_binary_search(
        losses_sorted=L,
        probs_sorted=p,
        alpha=beta,
        epsilon_iqae=epsilon_cdf,
        alpha_conf_iqae=alpha_conf_iqae,
        constraints=constraints,
        preferences=preferences,
        show_first_circuit=False,
    )
    var_b = varb_rep.var_value
    idx_b = varb_rep.var_index

    # Helper: build f_i for a band or tail; then amplitudes are sqrt(f_i)
    L_max = float(np.max(L))
    if L_max <= 0:
        # If losses can be <=0, shift them. (EVaR supports any real, but our [0,1] scaling needs nonnegativity)
        shift = float(-np.min(L))
        L_shift = L + shift
        L_max = float(np.max(L_shift))
    else:
        shift = 0.0
        L_shift = L

    def f_tail_loss(i: int) -> float:
        # scaled loss in [0,1], but only in tail i>=idx_a
        if i < idx_a:
            return 0.0
        return float(L_shift[i] / L_max)

    def f_tail_indicator(i: int) -> float:
        return 1.0 if i >= idx_a else 0.0

    def f_band_loss(i: int) -> float:
        # scaled loss in [0,1] for band idx_a..idx_b (inclusive)
        if i < idx_a or i > idx_b:
            return 0.0
        return float(L_shift[i] / L_max)

    def f_band_indicator(i: int) -> float:
        return 1.0 if (idx_a <= i <= idx_b) else 0.0

    # Build amplitude tables a(i)=sqrt(f_i) (must be in [0,1])
    tail_loss_amplitudes = [math.sqrt(max(0.0, min(1.0, f_tail_loss(i)))) for i in range(N)]
    tail_ind_amplitudes  = [math.sqrt(max(0.0, min(1.0, f_tail_indicator(i)))) for i in range(N)]
    band_loss_amplitudes = [math.sqrt(max(0.0, min(1.0, f_band_loss(i)))) for i in range(N)]
    band_ind_amplitudes  = [math.sqrt(max(0.0, min(1.0, f_band_indicator(i)))) for i in range(N)]

    # --- Tail probability P(L >= VaR_alpha) = E[1_tail] ---
    tail_prob_res = estimate_expectation_quantum_via_amplitude_loading(
        probs=probs_list,
        num_qubits=n,
        amplitudes=tail_ind_amplitudes,
        epsilon=epsilon_exp,
        alpha_conf=alpha_conf_iqae,
        constraints=constraints,
        preferences=preferences,
        verbose_show_once=False,
    )
    tail_prob = tail_prob_res.estimation

    # --- Tail loss expectation E[L * 1_tail] ---
    tail_loss_res = estimate_expectation_quantum_via_amplitude_loading(
        probs=probs_list,
        num_qubits=n,
        amplitudes=tail_loss_amplitudes,
        epsilon=epsilon_exp,
        alpha_conf=alpha_conf_iqae,
        constraints=constraints,
        preferences=preferences,
        verbose_show_once=False,
    )
    tail_loss_scaled = tail_loss_res.estimation
    # unscale: E[L_shift * 1_tail] = L_max * E[f]
    tail_loss_expect_shift = L_max * tail_loss_scaled
    # unshift back: L = L_shift - shift; but only in tail:
    # E[L * 1_tail] = E[L_shift*1_tail] - shift*P(tail)
    tail_loss_expect = tail_loss_expect_shift - shift * tail_prob

    cvar = float("nan")
    if tail_prob > 0:
        cvar = float(tail_loss_expect / tail_prob)

    # --- RVaR_{alpha,beta} = E[L | band] ---
    band_prob_res = estimate_expectation_quantum_via_amplitude_loading(
        probs=probs_list,
        num_qubits=n,
        amplitudes=band_ind_amplitudes,
        epsilon=epsilon_exp,
        alpha_conf=alpha_conf_iqae,
        constraints=constraints,
        preferences=preferences,
        verbose_show_once=False,
    )
    band_prob = band_prob_res.estimation

    band_loss_res = estimate_expectation_quantum_via_amplitude_loading(
        probs=probs_list,
        num_qubits=n,
        amplitudes=band_loss_amplitudes,
        epsilon=epsilon_exp,
        alpha_conf=alpha_conf_iqae,
        constraints=constraints,
        preferences=preferences,
        verbose_show_once=False,
    )
    band_loss_scaled = band_loss_res.estimation
    band_loss_expect_shift = L_max * band_loss_scaled
    band_loss_expect = band_loss_expect_shift - shift * band_prob

    rvar = float("nan")
    if band_prob > 0:
        rvar = float(band_loss_expect / band_prob)

    # --- EVaR via MGF estimates: E[e^{tL}] ---
    if t_grid is None:
        # Reasonable default grid; tune as needed.
        t_grid = np.linspace(0.25, 6.0, 24)

    one_minus_alpha = 1.0 - alpha
    evar = float("inf")
    best_t = float("nan")

    # We need to estimate mgf(t) = E[exp(t L)].
    # But IQAE estimates E[f] with f in [0,1].
    # We scale: exp(t L) in [exp(t Lmin), exp(t Lmax)].
    # Use shifted losses L (can be negative), handle range:
    Lmin = float(np.min(L))
    Lmax = float(np.max(L))

    for t in t_grid:
        if t <= 0:
            continue
        # Range of exp(tL)
        emin = math.exp(t * Lmin)
        emax = math.exp(t * Lmax)
        if emax == emin:
            mgf_est = emin
        else:
            # Define f(i) = (exp(t L_i) - emin)/(emax-emin) in [0,1]
            def f_mgf(i: int) -> float:
                return (math.exp(t * float(L[i])) - emin) / (emax - emin)

            mgf_amplitudes = [math.sqrt(max(0.0, min(1.0, f_mgf(i)))) for i in range(N)]
            mgf_res = estimate_expectation_quantum_via_amplitude_loading(
                probs=probs_list,
                num_qubits=n,
                amplitudes=mgf_amplitudes,
                epsilon=epsilon_exp,
                alpha_conf=alpha_conf_iqae,
                constraints=constraints,
                preferences=preferences,
                verbose_show_once=False,
            )
            f_est = mgf_res.estimation
            mgf_est = emin + (emax - emin) * f_est

        if one_minus_alpha <= 0:
            continue
        candidate = (math.log(mgf_est) - math.log(one_minus_alpha)) / float(t)
        if candidate < evar:
            evar = float(candidate)
            best_t = float(t)

    details = {
        "sorted_losses": L,
        "sorted_probs": p,
        "alpha": alpha,
        "beta": beta,
        "VaR(alpha)": {"value": var_a, "index": idx_a, "outer_iters": var_rep.outer_iters, "last_cdf": var_rep.last_cdf_est},
        "VaR(beta)":  {"value": var_b, "index": idx_b, "outer_iters": varb_rep.outer_iters, "last_cdf": varb_rep.last_cdf_est},
        "tail_probability": tail_prob_res,
        "tail_loss_expectation_scaled_Ef": tail_loss_res,
        "band_probability": band_prob_res,
        "band_loss_expectation_scaled_Ef": band_loss_res,
        "EVaR_best_t": best_t,
        "EVaR_t_grid": t_grid,
        "iqae_knobs": {
            "epsilon_cdf": epsilon_cdf,
            "epsilon_exp": epsilon_exp,
            "alpha_conf_iqae": alpha_conf_iqae,
        },
        "scaling": {
            "shift_applied": shift,
            "L_max_used_for_scaling": L_max,
            "loss_min": Lmin,
            "loss_max": Lmax,
        },
    }

    return RiskQuantumResults(var=var_a, cvar=cvar, rvar=rvar, evar=evar, details=details)


# -----------------------------------------
# Circuit / hardware metrics (optional)
# -----------------------------------------

def try_print_hardware_metrics(qprog) -> None:
    """
    Uses Classiq Analyzer to produce depth/gate count comparisons across providers.
    This requires: pip install "classiq[analyzer_sdk]"
    """
    try:
        from classiq import Analyzer  # type: ignore
    except Exception as e:
        print("\n[Analyzer not available]")
        print("Install with: pip install 'classiq[analyzer_sdk]'")
        print(f"Reason: {e}")
        return

    analyzer = Analyzer(circuit=qprog)

    providers = ["IBM Quantum", "Azure Quantum", "Amazon Braket"]
    print("\n[Hardware comparison table]")
    table = analyzer.get_hardware_comparison_table(providers=providers)
    # table is typically a pandas DataFrame-like object
    print(table)
    # Optional plotting if in Jupyter:
    # analyzer.plot_hardware_comparison_table()


# -----------------------------------------
# Example main (replace with your own data)
# -----------------------------------------

def make_example_distribution(num_qubits: int = 7, mu: float = 0.7, sigma: float = 0.13) -> Tuple[np.ndarray, np.ndarray]:
    """
    Example: discretized lognormal-like loss distribution over 2^n points.
    For a real risk workflow, you'd feed losses from portfolio scenarios.
    """
    N = 2**num_qubits
    # A simple synthetic loss grid (nonnegative), not the exact lognormal PDF discretization
    # Replace this with your own scenario losses and probabilities.
    x = np.linspace(0.0, 3.0, N)
    # "lognormal-ish" shaped density (unnormalized)
    pdf = np.exp(-((np.log(x + 1e-6) - mu) ** 2) / (2 * sigma**2)) / (np.maximum(x, 1e-6) * sigma * math.sqrt(2 * math.pi))
    p = normalize_probs(pdf)
    # Treat x as losses
    return x, p


def main():
    # -----------------------------
    # User knobs
    # -----------------------------
    num_qubits = 7                   # distribution resolution: N=2^n
    alpha = 0.95                     # VaR/CVaR confidence level (quantile)
    beta = 0.99                      # RVaR upper quantile

    # IQAE target precision/confidence:
    epsilon_cdf = 0.02               # VaR binary search queries (CDF)
    epsilon_exp = 0.02               # expectation queries (CVaR/RVaR/EVaR)
    alpha_conf_iqae = 0.01           # failure probability for IQAE

    # Synthesis constraints/preferences (tune to your hardware goals)
    constraints = Constraints(max_width=28)  # like the VaR example uses a width cap
    preferences = Preferences(machine_precision=num_qubits)

    # -----------------------------
    # Data (replace with your own)
    # -----------------------------
    losses, probs = make_example_distribution(num_qubits=num_qubits)

    # Classical baselines
    order = np.argsort(losses)
    Ls = losses[order]
    ps = probs[order]

    var_c = classical_var(Ls, ps, alpha)
    cvar_c = classical_cvar(Ls, ps, alpha)
    rvar_c = classical_rvar(Ls, ps, alpha, beta)
    t_grid = np.linspace(0.25, 6.0, 24)
    evar_c, best_t_c = classical_evar(Ls, ps, alpha, t_grid)

    print("=== Classical baselines ===")
    print(f"VaR_{alpha:.2f}  = {var_c:.6f}")
    print(f"CVaR_{alpha:.2f} = {cvar_c:.6f}")
    print(f"RVaR_{alpha:.2f},{beta:.2f} = {rvar_c:.6f}")
    print(f"EVaR_{alpha:.2f} ≈ {evar_c:.6f} (grid best t={best_t_c:.3f})")

    # -----------------------------
    # Quantum estimates
    # -----------------------------
    print("\n=== Quantum (Classiq IQAE) ===")
    qr = compute_risk_measures_quantum(
        losses=losses,
        probs=probs,
        alpha=alpha,
        beta=beta,
        epsilon_cdf=epsilon_cdf,
        epsilon_exp=epsilon_exp,
        alpha_conf_iqae=alpha_conf_iqae,
        t_grid=t_grid,
        constraints=constraints,
        preferences=preferences,
        show_first_circuit=True,   # shows the first synthesized program in the analyzer UI
    )

    print(f"VaR_{alpha:.2f}  = {qr.var:.6f}   | abs err = {abs(qr.var - var_c):.6f}")
    print(f"CVaR_{alpha:.2f} = {qr.cvar:.6f}  | abs err = {abs(qr.cvar - cvar_c):.6f}")
    print(f"RVaR_{alpha:.2f},{beta:.2f} = {qr.rvar:.6f} | abs err = {abs(qr.rvar - rvar_c):.6f}")
    print(f"EVaR_{alpha:.2f} ≈ {qr.evar:.6f}  | abs err = {abs(qr.evar - evar_c):.6f}")

    # Print some “necessary analysis”
    print("\n=== Diagnostics ===")

    var_alpha_info = qr.details["VaR(alpha)"]
    print(f"VaR outer binary-search iterations: {var_alpha_info['outer_iters']}")
    print(f"VaR index (sorted): {var_alpha_info['index']}, VaR value: {var_alpha_info['value']:.6f}")
    last_cdf: IQAEResult = var_alpha_info["last_cdf"]
    print(f"Last CDF IQAE estimate: {last_cdf.estimation:.6f}  CI=[{last_cdf.ci_low:.6f}, {last_cdf.ci_high:.6f}]  elapsed={last_cdf.elapsed_s:.2f}s")
    if last_cdf.extras:
        print(f"Last CDF IQAE extras: {last_cdf.extras}")

    tail_prob: IQAEResult = qr.details["tail_probability"]
    print(f"Tail prob P(L >= VaR_alpha): {tail_prob.estimation:.6f}  CI=[{tail_prob.ci_low:.6f}, {tail_prob.ci_high:.6f}]  elapsed={tail_prob.elapsed_s:.2f}s")

    tail_loss: IQAEResult = qr.details["tail_loss_expectation_scaled_Ef"]
    print(f"Tail loss E[f]=E[(L/Lmax)*1_tail]: {tail_loss.estimation:.6f}  CI=[{tail_loss.ci_low:.6f}, {tail_loss.ci_high:.6f}]  elapsed={tail_loss.elapsed_s:.2f}s")

    print(f"EVaR best t (grid): {qr.details['EVaR_best_t']}")

    # Optional: show hardware comparison metrics for one representative circuit
    # Here we re-create a representative qprog: the first CDF IQAE model's qprog is not stored.
    # If you want metrics, you can build a qprog explicitly similarly to estimate_cdf_prob_quantum(...)
    # and then call try_print_hardware_metrics(qprog).
    print("\n[Tip] For circuit depth/gate metrics across providers, install analyzer_sdk and call try_print_hardware_metrics(qprog).")


if __name__ == "__main__":
    main()
