"""UPC Sostenible Battery Agent - Datathon 2026
Only this file + optional model.joblib / .pkl are submitted.
Positive kW = charge; negative kW = discharge.
No network, no dynamic code execution.
"""
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None


def make_features(timestamp, horizons, net_history, price_history, co2_history,
                  prices, emissions):
    """Causal feature builder (same for training and inference).
    Net history only contains times strictly before the forecast origin.
    """
    t = pd.Timestamp(timestamp)

    def past(history, lag):
        return history.get(t - pd.Timedelta(hours=lag), np.nan)

    # Net load lags
    base = [past(net_history, k) for k in (1, 2, 3, 24, 48, 168)]

    # Rolling stats (causal windows)
    for window in (24, 168):
        a = np.array([past(net_history, k) for k in range(1, window + 1)], dtype=float)
        good = a[np.isfinite(a)]
        base.extend([
            float(np.mean(good)) if len(good) else np.nan,
            float(np.std(good)) if len(good) else np.nan,
            len(good) / float(window)
        ])

    # Price & CO2 lags
    base += [past(price_history, k) for k in (1, 24)]
    base += [past(co2_history, k) for k in (1, 24)]

    # Historical price-CO2 correlation (never uses future)
    a = np.array([past(price_history, k) for k in range(1, 25)], dtype=float)
    b = np.array([past(co2_history, k) for k in range(1, 25)], dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() > 3 and np.std(a[ok]) > 1e-9 and np.std(b[ok]) > 1e-9:
        corr = float(np.corrcoef(a[ok], b[ok])[0, 1])
    else:
        corr = 0.0
    base.append(corr)

    rows = []
    for h in horizons:
        target = t + pd.Timedelta(hours=int(h))
        seasonal = net_history.get(target - pd.Timedelta(hours=24), np.nan)
        weekly = net_history.get(target - pd.Timedelta(hours=168), np.nan)
        rows.append(base + [
            float(h),
            target.hour, target.dayofweek, target.month,
            np.sin(2 * np.pi * target.hour / 24.0),
            np.cos(2 * np.pi * target.hour / 24.0),
            np.sin(2 * np.pi * target.dayofyear / 365.25),
            np.cos(2 * np.pi * target.dayofyear / 365.25),
            seasonal, weekly,
            prices[h], emissions[h]
        ])
    return np.asarray(rows, dtype=float)


class BatteryAgent:
    def __init__(self):
        # Physical parameters (official)
        self.eta = float(np.sqrt(0.90))          # one-way efficiency ≈ 0.9487
        self.capacity = 2000.0                   # kWh
        self.power = 500.0                       # kW inverter limit

        # Persistent memory across the 2016 evaluation hours
        self.historial = []
        self.net_history = {}
        self.price_history = {}
        self.co2_history = {}

        # Scaling factors (positive baselines from training set)
        self.cost_scale = 140.0
        self.co2_scale = 100.0
        self.terminal_factor = 0.85
        self.model_weight = 0.75

        # Export compensation price (official excess solar compensation)
        self.export_price = 0.04                 # €/kWh

        # State discretization (finer = better near limits, still fast)
        self.n_states = 101
        self.states = np.linspace(0.0, self.capacity, self.n_states)

        # Optional pre-trained model
        self.model = None
        model_path = Path(__file__).resolve().parent / "model.joblib"
        if model_path.is_file() and joblib is not None:
            try:
                artifact = joblib.load(model_path)
                self.model = artifact.get("model", artifact)
                self.cost_scale = artifact.get("cost_scale", self.cost_scale)
                self.co2_scale = artifact.get("co2_scale", self.co2_scale)
                self.terminal_factor = artifact.get("terminal_factor", self.terminal_factor)
                self.model_weight = artifact.get("model_weight", self.model_weight)
            except Exception:
                self.model = None

        # Pre-compute signed transitions and feasibility mask
        delta = self.states[None, :] - self.states[:, None]
        self.transitions = np.where(delta >= 0.0, delta / self.eta, delta * self.eta)
        self.feasible = np.abs(self.transitions) <= (self.power + 1e-9)

    def _update_history(self, obs, prices, emissions):
        """Store the most recent realized net load (strictly causal)."""
        t = pd.Timestamp(obs.get("timestamp", pd.Timestamp.now()))
        telemetry = obs.get("noves_dades_5min", {}) or {}
        last_consum = float(telemetry.get("ultim_consum_kw", 1000.0))
        last_solar = float(telemetry.get("ultima_solar_kw", 0.0))
        last_net = last_consum - last_solar
        if not np.isfinite(last_net):
            last_net = 1000.0

        # The observation at step 0 already contains the current telemetry.
        # We record it as the value that just finished (t-1h).
        self.net_history[t - pd.Timedelta(hours=1)] = last_net
        self.price_history[t] = float(prices[0])
        self.co2_history[t] = float(emissions[0])

        # Keep only a rolling window of history
        cutoff = t - pd.Timedelta(hours=200)
        for hist in (self.net_history, self.price_history, self.co2_history):
            for key in list(hist.keys()):
                if key < cutoff:
                    del hist[key]

        return last_net

    def forecast_net(self, obs, prices, emissions):
        """Produce a 24-h net-load forecast (positive = net demand)."""
        t = pd.Timestamp(obs.get("timestamp", pd.Timestamp.now()))
        last_net = self._update_history(obs, prices, emissions)

        horizons = list(range(len(prices)))
        # Pure seasonal fallback (very robust)
        seasonal = np.array([
            self.net_history.get(t + pd.Timedelta(hours=h - 24), last_net)
            for h in horizons
        ], dtype=float)

        if self.model is not None and self.model_weight > 0.0:
            try:
                x = make_features(t, horizons, self.net_history,
                                  self.price_history, self.co2_history,
                                  prices, emissions)
                pred = np.asarray(self.model.predict(x), dtype=float)
                forecast = (self.model_weight * pred
                            + (1.0 - self.model_weight) * seasonal)
            except Exception:
                forecast = seasonal
        else:
            forecast = seasonal

        return np.clip(np.nan_to_num(forecast, nan=last_net), -6000.0, 6000.0)

    def optimize(self, soc, net, prices, emissions):
        """Receding-horizon DP on energy states.
        Stage cost = carbon*CO2 + money*price  (weighted 70/30).
        Handles negative prices and free excess solar correctly.
        """
        carbon = 0.70 / max(self.co2_scale, 1e-9)
        money = 0.30 / max(self.cost_scale, 1e-9)
        marginal = carbon * emissions + money * prices

        # Terminal value of energy left in the battery
        salvage = self.terminal_factor * self.eta * max(0.0, float(np.median(marginal)))
        value = -salvage * self.states

        # Backward pass
        for h in range(len(prices) - 1, 0, -1):
            grid = net[h] + self.transitions          # net grid power after action
            # Import cost when grid > 0; export revenue when grid < 0
            stage = (marginal[h] * np.maximum(grid, 0.0)
                     - money * self.export_price * np.maximum(-grid, 0.0))
            value = np.min(
                np.where(self.feasible, stage + value[None, :], np.inf),
                axis=1
            )

        # Continuous SOC → discrete candidate actions (including exact idle)
        delta = self.states - soc
        actions = np.where(delta >= 0.0, delta / self.eta, delta * self.eta)
        feasible = np.abs(actions) <= (self.power + 1e-9)

        grid0 = net[0] + actions
        costs = (marginal[0] * np.maximum(grid0, 0.0)
                 - money * self.export_price * np.maximum(-grid0, 0.0)
                 + value)
        costs[~feasible] = np.inf

        best_idx = int(np.argmin(costs))
        best_action = float(actions[best_idx])

        # Prefer exact idle when it is essentially as good (numerical stability)
        idle_cost = (marginal[0] * max(net[0], 0.0)
                     - money * self.export_price * max(-net[0], 0.0)
                     + np.interp(soc, self.states, value))
        if idle_cost <= costs[best_idx] + 1e-9:
            return 0.0
        return best_action

    def act(self, obs: dict) -> float:
        """Official entry point. Always returns a finite float in [-500, +500]."""
        # Persist observation (as required by the starter kit)
        self.historial.append(obs)

        # ---- Extract current market signals ---------------------------------
        current_p = float(obs.get("preu_actual", 0.14))
        current_c = float(obs.get("co2_actual", 0.15))

        prices = np.asarray(obs.get("preus_properes_24h", [current_p]), dtype=float)[:24]
        co2 = np.asarray(obs.get("co2_properes_24h", [current_c]), dtype=float)[:24]

        n = min(len(prices), len(co2))
        if n == 0:
            prices = np.array([current_p])
            co2 = np.array([current_c])
        else:
            prices = prices[:n]
            co2 = co2[:n]

        # Force current hour to the realized values
        prices[0] = current_p
        co2[0] = current_c

        prices = np.nan_to_num(prices, nan=0.14, posinf=0.50, neginf=0.0)
        co2 = np.maximum(np.nan_to_num(co2, nan=0.15, posinf=0.5, neginf=0.0), 0.0)

        # ---- Battery state --------------------------------------------------
        soc = float(obs.get("soc_kwh", 1000.0))
        if not np.isfinite(soc):
            return 0.0
        soc = float(np.clip(soc, 0.0, self.capacity))

        # ---- Forecast net campus load & optimize ---------------------------
        net = self.forecast_net(obs, prices, co2)
        action = self.optimize(soc, net, prices, co2)

        # ---- Hard physical safety clamp (zero infractions) -----------------
        # Official recommendation: leave a tiny margin so the simulator
        # never clips the action.
        eps = 1e-4
        max_charge = max(0.0, min(self.power, (self.capacity - soc) / self.eta - eps))
        max_discharge = max(0.0, min(self.power, soc * self.eta - eps))

        if not np.isfinite(action):
            return 0.0
        return float(np.clip(action, -max_discharge, max_charge))
