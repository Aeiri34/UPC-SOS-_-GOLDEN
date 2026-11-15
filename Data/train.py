"""Run from project directory: python train.py
Finite reproducible grid search, purged temporal CV, then untouched holdout.
Expand CONFIGS / folds for a larger budget; no finite search proves global optimality.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from agent import BatteryAgent, make_features
from data_pipeline import load_data, segments, simulate, aggregate

ROOT = Path(__file__).resolve().parent
CONFIGS = ([dict(family='lightgbm', depth=d, learning_rate=lr) for d in (3, 5) for lr in (.04, .08)] +
           [dict(family='xgboost', depth=d, learning_rate=lr) for d in (3, 5) for lr in (.04, .08)] +
           [dict(family='hist', depth=d, learning_rate=.06) for d in (3, 5)])


def estimator(config):
    family, depth, lr = config['family'], config['depth'], config['learning_rate']
    if family == 'lightgbm':
        return LGBMRegressor(n_estimators=200, learning_rate=lr, max_depth=depth,
                             num_leaves=2**depth, min_child_samples=60, reg_lambda=10.,
                             colsample_bytree=.9, subsample=1., verbosity=-1, n_jobs=2, random_state=42)
    if family == 'xgboost':
        return XGBRegressor(n_estimators=200, learning_rate=lr, max_depth=depth,
                            min_child_weight=30, reg_lambda=10., colsample_bytree=.9,
                            tree_method='hist', n_jobs=2, random_state=42)
    return HistGradientBoostingRegressor(max_iter=150, max_depth=depth, learning_rate=lr,
                                        min_samples_leaf=60, l2_regularization=10., random_state=42)


def supervised(frame):
    net = frame.net_kw.dropna().to_dict()
    price = frame.precio_eur_per_kwh.dropna().to_dict()
    carbon = frame.co2_kg_per_kwh.dropna().to_dict()
    xx, yy, origins, targets = [], [], [], []
    horizons = [0, 1, 3, 6, 12, 18, 23]
    for i in range(168, len(frame)-23, 2):
        t = frame.index[i]
        p = frame.precio_eur_per_kwh.iloc[i:i+24].to_numpy()
        c = frame.co2_kg_per_kwh.iloc[i:i+24].to_numpy()
        x = make_features(t, horizons, net, price, carbon, p, c)
        for j, h in enumerate(horizons):
            y = frame.net_kw.iloc[i+h]
            if np.isfinite(y) and np.isfinite(p[h]) and np.isfinite(c[h]):
                xx.append(x[j]); yy.append(y); origins.append(t); targets.append(frame.index[i+h])
    return np.array(xx), np.array(yy), pd.DatetimeIndex(origins), pd.DatetimeIndex(targets)


def scales(frame):
    valid = frame.dropna(subset=['net_kw','precio_eur_per_kwh','co2_kg_per_kwh'])
    grid = valid.net_kw.clip(lower=0)
    return float((grid*valid.precio_eur_per_kwh).mean()), float((grid*valid.co2_kg_per_kwh).mean())


def configured(model, scale, weight, terminal, export=0.):
    agent = BatteryAgent()
    agent.model = model
    agent.cost_scale, agent.co2_scale = scale
    agent.model_weight, agent.terminal_factor = weight, terminal
    agent.export_price = export
    return agent


def main():
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    frame, _, audit = load_data(ROOT/'data')
    frame.to_csv(out/'hourly.csv'); audit.to_csv(out/'data_audit.csv', index=False)
    x, y, origin, target = supervised(frame)
    # Last two calendar months are reserved before any selection or fitting.
    cutoff = frame.index.max().normalize().replace(day=1)-pd.DateOffset(months=2)
    dev = target < cutoff
    unique = pd.DatetimeIndex(sorted(set(origin[dev])))
    splits = list(TimeSeriesSplit(n_splits=3, gap=24).split(unique))
    records, fold_cache = [], []
    for fold, (tr, va) in enumerate(splits):
        # Explicit target embargo prevents multi-horizon labels crossing fold boundary.
        train_end, val_start, val_end = unique[tr[-1]], unique[va[0]], unique[va[-1]]
        train_mask = (origin <= train_end) & (target <= train_end) & dev
        valid_mask = (origin >= val_start) & (origin <= val_end) & (target <= val_end) & dev
        scale = scales(frame.loc[:train_end])
        fold_cache.append((train_mask, frame.loc[val_start:val_end], scale))
        for idx, config in enumerate(CONFIGS):
            model = estimator(config).fit(x[train_mask], y[train_mask])
            pred = model.predict(x[valid_mask])
            mae = float(mean_absolute_error(y[valid_mask], pred))
            records.append(dict(candidate=idx, fold=fold, mae=mae, **config))
            print('CV', fold, idx, config['family'], round(mae, 2), flush=True)
        # Column -4 is target-aligned yesterday; fallback to last measured net.
        baseline = np.where(np.isfinite(x[valid_mask, -4]), x[valid_mask, -4], x[valid_mask, 0])
        baseline = np.nan_to_num(baseline, nan=1000.)
        records.append(dict(candidate=-1, fold=fold, mae=float(mean_absolute_error(y[valid_mask], baseline)), family='seasonal', depth=0, learning_rate=0.))
    scores = pd.DataFrame(records)
    scores.to_csv(out/'cv_models.csv', index=False)
    ranking = scores.groupby('candidate').mae.agg(['mean','std'])
    # Penalize inter-fold variability modestly; holdout never participates.
    ranking['robust_mae'] = ranking['mean']+.25*ranking['std']
    learned = ranking.loc[ranking.index >= 0]
    best = int(learned.robust_mae.idxmin())
    ranking.to_csv(out/'cv_ranking.csv')
    policy_records = []
    # End-to-end controller comparison on the last complete 168h block of each CV fold.
    for fold, (train_mask, validation, scale) in enumerate(fold_cache):
        model = estimator(CONFIGS[best]).fit(x[train_mask], y[train_mask])
        blocks = segments(validation, min_hours=48)
        if not blocks:
            continue
        block = blocks[-1].iloc[-168:]
        for weight in (0., 1.):
            for terminal in (.5, .8, 1.):
                result, _ = simulate(configured(model, scale, weight, terminal), block)
                policy_records.append(dict(fold=fold, model_weight=weight, terminal_factor=terminal, **aggregate([result])))
    policy = pd.DataFrame(policy_records)
    policy.to_csv(out/'cv_policies.csv', index=False)
    rank = policy.groupby(['model_weight','terminal_factor']).score.agg(['mean','std'])
    rank['robust_score'] = rank['mean']-.25*rank['std'].fillna(0.)
    weight, terminal = rank.robust_score.idxmax()
    model = estimator(CONFIGS[best]).fit(x[dev], y[dev])
    scale = scales(frame.loc[frame.index < cutoff])
    test = frame.loc[frame.index >= cutoff]
    summary, traces = {}, []
    for export in (0., .04):
        for label, w in [('selected', weight), ('seasonal_mpc', 0.), ('ml_mpc', 1.)]:
            block_results = []
            for block_id, block in enumerate(segments(test, min_hours=24)):
                result, trace = simulate(configured(model, scale, w, terminal, export), block, export)
                block_results.append(result)
                if label == 'selected' and export == 0.:
                    trace['block_id'] = block_id; traces.append(trace)
            summary[f'{label}_export_{export}'] = aggregate(block_results)
            print('HOLDOUT', label, export, summary[f'{label}_export_{export}'], flush=True)
    pd.concat(traces).to_csv(out/'holdout_trace.csv')
    # Preserve the exact model used for holdout reproduction separately.
    payload = dict(model=model, cost_scale=scale[0], co2_scale=scale[1],
                   terminal_factor=float(terminal), model_weight=float(weight))
    joblib.dump(payload, out/'validated_model.joblib')
    summary['cutoff'] = str(cutoff)
    summary['chosen_config'] = CONFIGS[best]
    summary['policy'] = dict(model_weight=float(weight), terminal_factor=float(terminal))
    summary['cv_forecast_mae'] = float(ranking.loc[best, 'mean'])
    summary['cv_seasonal_mae'] = float(ranking.loc[-1, 'mean'])
    summary['data'] = dict(clock_hours=len(frame), complete_hours=int(frame[['load_kw','pv_kw','precio_eur_per_kwh','co2_kg_per_kwh']].notna().all(axis=1).sum()),
                           start=str(frame.index.min()), end=str(frame.index.max()))
    # Deployable refit uses all released data; its future score is NOT measured here.
    final_model = estimator(CONFIGS[best]).fit(x, y)
    final_scale = scales(frame)
    payload.update(model=final_model, cost_scale=final_scale[0], co2_scale=final_scale[1])
    joblib.dump(payload, ROOT/'model.joblib')
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Saved final model, untouched holdout metrics and audit.', flush=True)


if __name__ == '__main__':
    main()
