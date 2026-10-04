import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

class MarketSimulator:
    """Simulates market paths using Geometric Brownian Motion (GBM)."""
    def __init__(self, spot, drift, vol, t_span, steps, paths):
        self.spot = spot
        self.drift = drift
        self.vol = vol
        self.t_span = t_span
        self.steps = steps
        self.paths = paths
        self.dt = t_span / steps
        self.time_grid = np.linspace(0, t_span, steps + 1)
        
    def generate_paths(self):
        """Returns a matrix of simulated asset prices (paths x time_steps)."""
        np.random.seed(42) # For reproducibility during interviews
        Z = np.random.standard_normal((self.paths, self.steps))
        # GBM exact solution
        growth_factor = np.exp((self.drift - 0.5 * self.vol**2) * self.dt + self.vol * np.sqrt(self.dt) * Z)
        
        paths = np.zeros((self.paths, self.steps + 1))
        paths[:, 0] = self.spot
        for t in range(1, self.steps + 1):
            paths[:, t] = paths[:, t-1] * growth_factor[:, t-1]
        return self.time_grid, paths

class FXForwardPricer:
    """Prices an FX Forward across simulated paths and time grids."""
    def __init__(self, notional, strike, r_dom, r_for, maturity):
        self.notional = notional
        self.strike = strike
        self.r_dom = r_dom
        self.r_for = r_for
        self.maturity = maturity
        
    def calculate_mtm(self, time_grid, spot_paths):
        """
        Calculates Mark-to-Market (MtM) value at each time step t.
        MtM = N * (S_t * exp(-r_f * (T-t)) - K * exp(-r_d * (T-t)))
        """
        paths, time_steps = spot_paths.shape
        mtm_matrix = np.zeros((paths, time_steps))
        
        for idx, t in enumerate(time_grid):
            if t > self.maturity:
                break
            time_to_maturity = self.maturity - t
            # Forward pricing formula
            asset_leg = spot_paths[:, idx] * np.exp(-self.r_for * time_to_maturity)
            cash_leg = self.strike * np.exp(-self.r_dom * time_to_maturity)
            mtm_matrix[:, idx] = self.notional * (asset_leg - cash_leg)
            
        return mtm_matrix

class ExposureCalculator:
    """Extracts EE, PFE, and computes SIMM-style Margin."""
    def __init__(self, mtm_matrix, time_grid):
        self.mtm = mtm_matrix
        self.time_grid = time_grid
        # Exposure is max(MtM, 0) - bank only has risk when the trade is an asset
        self.exposure = np.maximum(self.mtm, 0)
        
    def get_expected_exposure(self):
        return np.mean(self.exposure, axis=0)
        
    def get_pfe(self, percentile=95):
        return np.percentile(self.exposure, percentile, axis=0)
    
    def get_epe(self):
        """Expected Positive Exposure (time-averaged EE)."""
        ee = self.get_expected_exposure()
        return np.trapezoid(ee, self.time_grid) / self.time_grid[-1]
    
    def calculate_simm_im_proxy(self, spot, notional, r_for, maturity, risk_weight=0.08):
        """
        Approximates ISDA SIMM Initial Margin using Delta sensitivity.
        Delta = dV/dS = N * exp(-r_f * T)
        """
        delta_sensitivity = notional * np.exp(-r_for * maturity)
        # In SIMM, Margin = Risk_Weight * Sensitivity
        initial_margin = risk_weight * delta_sensitivity * spot
        return initial_margin

# --- Execution Block ---
if __name__ == "__main__":
    # 1. Market Parameters (e.g., USD/INR FX Rate)
    S0 = 83.00        # Spot FX rate
    r_dom = 0.07      # Domestic rate (INR)
    r_for = 0.05      # Foreign rate (USD)
    vol = 0.12        # 12% annual volatility
    drift = r_dom - r_for # Covered interest parity drift
    
    # Simulation Parameters
    T = 1.0           # 1-year maturity
    steps = 250       # Daily steps
    num_paths = 1000
    
    # Contract Parameters
    notional = 1_000_000 # 1 Million USD Notional
    strike = S0 * np.exp(drift * T) # At-the-money forward strike (approx 84.67)
    
    # 2. Run Modules
    simulator = MarketSimulator(S0, drift, vol, T, steps, num_paths)
    t_grid, spot_paths = simulator.generate_paths()
    
    pricer = FXForwardPricer(notional, strike, r_dom, r_for, T)
    mtm_paths = pricer.calculate_mtm(t_grid, spot_paths)
    
    risk_engine = ExposureCalculator(mtm_paths, t_grid)
    ee = risk_engine.get_expected_exposure()
    pfe = risk_engine.get_pfe(95)
    epe = risk_engine.get_epe()
    im_proxy = risk_engine.calculate_simm_im_proxy(S0, notional, r_for, T)
    
    # 3. Output Results
    print(f"Contract Strike Rate: {strike:.2f}")
    print(f"Expected Positive Exposure (EPE): INR {epe:,.2f}")
    print(f"SIMM Initial Margin Proxy (Day 0): INR {im_proxy:,.2f}")
    print(f"Max PFE (95%): INR {np.max(pfe):,.2f}")

    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(t_grid, ee, label='Expected Exposure (EE)', color='blue', linewidth=2)
    plt.plot(t_grid, pfe, label='Potential Future Exposure (PFE 95%)', color='red', linestyle='--', linewidth=2)
    plt.title('Counterparty Credit Risk Exposure Profiles (1Y FX Forward)')
    plt.xlabel('Time (Years)')
    plt.ylabel('Exposure (INR)')
    plt.legend()
    plt.grid(True)
    plt.show()