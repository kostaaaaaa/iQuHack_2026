# iQuHACK 2026 – State Street × Classiq Challenge  
**Quantum Value at Risk Estimation via Monte Carlo and Quantum Amplitude Estimation**

> *“Quantum algorithms won’t eliminate uncertainty; but they might finally help us measure it better.”*  
> — **State Street × Classiq Team**

---

## Project Overview

This project was developed for the **iQuHACK 2026 – State Street × Classiq Challenge**.  
The goal is to **compare classical and quantum approaches to financial risk estimation**, focusing on **Value at Risk (VaR)** and its extensions.

We implement and benchmark:
- **Classical Monte Carlo estimation**
- **Quantum Amplitude Estimation (QAE)**
- **Iterative Quantum Amplitude Estimation (IQAE)**

on controlled, toy probability distributions using the **Classiq SDK**.  
The emphasis is **not practical speed-up**, but **sampling-complexity scaling, assumptions, and intuition**.

---

## Objectives

- Study convergence of Monte Carlo VaR estimation  
- Demonstrate quadratic sampling advantage of QAE/IQAE  
- Understand assumptions behind quantum advantage  
- Analyze discretization, oracle design, and stopping criteria  

---

## Repository Structure

```
team-name/
│
├── classical/
│   ├── monte_carlo_var.ipynb
│   ├── distributions.py
│   └── utils.py
│
├── quantum/
│   ├── state_preparation.py
│   ├── threshold_oracle.py
│   ├── qae.py
│   ├── iqae.py
│   └── var_bisection.py
│
├── experiments/
│   ├── scaling_analysis.ipynb
│   ├── sensitivity_analysis.ipynb
│   └── plots/
│
├── writeup/
│   └── report.pdf
│
├── requirements.txt
└── README.md
```

---

## Value at Risk Definition

For a profit-and-loss random variable $X$ and confidence level $\alpha = 95\%$:
$\text{VaR}_\alpha = \inf \{ x \mid \mathbb{P}(X \le x) \ge \alpha \}$

Distributions studied:
- Gaussian  
- Lognormal  
- Student’s t  

---

## Classical Monte Carlo

- Draw $N$ samples  
- Estimate the $95\%$ empirical quantile  
- Compare to theoretical VaR  

**Scaling:**  
$\varepsilon = O(1/\sqrt{N})$

---

## Quantum Methodology

Implemented fully in **Classiq**:

1. Probability state preparation  
2. Threshold oracle for tail events  
3. QAE and Iterative QAE  
4. Classical bisection search for VaR  

**Scaling:**  
$\varepsilon = O(1/N)$

---

## Benchmarking & Sensitivity

We analyze:
- Accuracy vs number of probability queries  
- Discretization resolution  
- Confidence level (fixed at $95\%$)  
- IQAE stopping rules  

We clearly distinguish:
- Estimation error  
- Modeling/discretization error  

---

## Extensions

- CVaR (Expected Shortfall)  
- RVaR (Range Value at Risk)  
- Fat-tailed distributions  

---

## Environment

- Classiq SDK  
- No quantum hardware required  
- Fully reproducible simulations  

---

## Key Insight

This challenge is about **understanding sampling complexity**, not practical runtime speedups.

Quantum amplitude estimation provides a **quadratic reduction in sample complexity**, but only under specific modeling and oracle assumptions.

---

## Submission

All code included  
Reproducible notebooks  
README and writeup provided inside the folder
