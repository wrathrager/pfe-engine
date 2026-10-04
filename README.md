# Counterparty Risk & CVA Engine: Multi-Asset Derivatives Netting Framework

This repository contains a quantitative risk infrastructure engine designed to simulate market paths, price OTC derivatives, extract Counterparty Credit Risk (CCR) exposures, and compute uncollateralized Credit Valuation Adjustments (CVA) and ISDA Initial Margin proxies.

## 1. Simulation Results & Inferential Analysis

### Single-Asset Exposure Profile (FX Forward via GBM)
The FX Forward simulation demonstrates the risk profile of an uncollateralized derivative over a 1-year horizon. 
* **Diffusion Mechanics:** The Potential Future Exposure (PFE) expands dynamically over time, driven by the stochastic volatility of the underlying asset, reaching a maximum 95% worst-case exposure of INR 18,409,624.65. 
* **Margin Calibration:** The engine computes a sensitivity-based Initial Margin proxy of INR 6,316,163.38 at Day 0. This delta-based margin provides a capital buffer against the calculated Expected Positive Exposure (EPE) of INR 2,643,839.70.

### Multi-Swap Portfolio Netting (Hull-White 1-Factor)
The interest rate swap (IRS) portfolio simulation validates the capital efficiency achieved through legally enforceable ISDA Master Agreements.
* **Netting Benefit:** Aggregating an offsetting 5Y Payer Swap and a 3Y Receiver Swap under a close-out netting agreement reduces the Credit Valuation Adjustment (CVA) capital charge from a gross $5,764.96 to a net $2,570.96. This represents a 55.40% reduction in required capital reserves.
* **Lifecycle Amortization:** The exposure plots accurately map the portfolio's lifecycle, highlighted by the sharp structural drop in both Expected Exposure (EE) and PFE at the 3-year mark when the receiver swap matures, before fully amortizing to zero at year 5.

---

## 2. Mathematical Framework & Models Implemented

### Market Simulation Engines
**1. Geometric Brownian Motion (GBM)**
Used to simulate the underlying spot path for FX Forwards over a discretized time grid:
$$S_{t+\Delta t} = S_t \exp\left(\left(r_d - r_f - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} Z\right)$$
where $Z \sim \mathcal{N}(0,1)$, $r_d$ is the domestic rate, and $r_f$ is the foreign rate.

**2. Hull-White 1-Factor Model (HW1F)**
Used to model the stochastic evolution of the short-rate $r(t)$ for Interest Rate Swap valuation:
$$dr(t) = (\theta(t) - a r(t))dt + \sigma dW(t)$$
where $a$ is the mean-reversion speed and $\sigma$ is the short-rate volatility. The analytical Zero-Coupon Bond (ZCB) price under this model is:
$$P(t, T) = A(t, T) \exp(-B(t, T) r(t))$$
where $B(t, T) = \frac{1 - e^{-a(T-t)}}{a}$.

### Counterparty Exposure Metrics
For a given derivative with Mark-to-Market value $V(t)$, the firm only faces credit risk when the position is an asset (in-the-money).
* **Exposure ($E_t$):** $E(t) = \max(V(t), 0)$
* **Expected Exposure (EE):** The cross-sectional average across all Monte Carlo paths at time $t$:
  $$EE(t) = \mathbb{E}[\max(V(t), 0)]$$
* **Potential Future Exposure (PFE):** The $95^{\text{th}}$ percentile of the exposure distribution at time $t$, used for internal credit limit monitoring:
  $$\mathbb{P}(E(t) \le PFE_{95\%}(t)) = 0.95$$
* **Expected Positive Exposure (EPE):** The time-weighted average of $EE(t)$ over the contract life, integrated via the trapezoidal rule:
  $$EPE = \frac{1}{T} \int_0^T EE(t) dt$$

### ISDA Master Agreement Netting
Evaluates the exposure of a portfolio of $N$ trades under different legal collateral agreements:
* **Gross Exposure (No Netting):**
  $$E_{\text{gross}}(t) = \sum_{i=1}^N \max(V_i(t), 0)$$
* **Net Exposure (With ISDA Close-Out Netting):**
  $$E_{\text{net}}(t) = \max\left(\sum_{i=1}^N V_i(t), 0\right)$$

### Credit Valuation Adjustment (CVA)
Calculates the uncollateralized bilateral CVA, representing the market value of counterparty credit risk. It integrates the discounted Expected Exposure with the counterparty's marginal Probability of Default (PD):
$$CVA = (1 - R) \sum_{i=1}^m D(t_i) \cdot EE(t_i) \cdot (PD(t_i) - PD(t_{i-1}))$$
where $R$ is the Recovery Rate (set to 40%), $D(t_i)$ is the risk-free discount factor, and $PD(t) = 1 - e^{-\lambda t}$ based on a flat hazard rate $\lambda$.

### SIMM Initial Margin Proxy
Approximates the regulatory standard initial margin by isolating the first-order sensitivity (Delta, $\Delta$) of the instrument and scaling it by a standard risk weight ($RW$):
$$IM = RW \times \left( N \cdot e^{-r_f \cdot T} \right) \times S_0$$