"""Read all six original CSVs, audit quality, preserve actual clock-time gaps."""
from pathlib import Path
import numpy as np
import pandas as pd

NAMES = ('contador', 'inversor', 'meteo', 'entradas_salidas', 'precio_pvpc', 'co2_red')


def read_raw(folder, name):
    folder = Path(folder)
    path = folder/(name+'.csv')
    if not path.exists():
        path = folder/(name+'(1).csv')
    frame = pd.read_csv(path, low_memory=False)
    frame['Fecha'] = pd.to_datetime(frame['Fecha'], errors='coerce')
    return frame


def load_data(folder):
    sources, audit = {}, []
    for name in NAMES:
        raw = read_raw(folder, name)
        audit.append({'dataset': name, 'rows': len(raw),
                      'duplicate_times': int(raw.Fecha.duplicated().sum()),
                      'invalid_times': int(raw.Fecha.isna().sum()),
                      'missing_cells': int(raw.isna().sum().sum())})
        raw = raw.dropna(subset=['Fecha']).set_index('Fecha').sort_index()
        numeric = raw.select_dtypes(include='number').drop(columns=['Medida'], errors='ignore')
        sources[name] = numeric.groupby(level=0).mean().resample('1h').mean()
    meter = sources['contador']['Contador - POTENCIA ACTIVA EXPORTADA (kW)']
    # Match the challenge's preprocessing recipe, despite ambiguous net/gross wording.
    load = (-meter).clip(0, 5000).rename('load_kw')
    solar_cols = [f'Inversor {i} - POTENCIA ACTIVA (kW)' for i in (1, 2, 3)]
    solar = sources['inversor'][solar_cols].sum(axis=1, min_count=3).clip(lower=0).rename('pv_kw')
    hourly = pd.concat([load, solar, sources['precio_pvpc']['precio_eur_per_kwh'],
                        sources['co2_red']['co2_kg_per_kwh']], axis=1).sort_index()
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq='1h', name='Fecha'))
    hourly['net_kw'] = hourly.load_kw-hourly.pv_kw
    # No backfill, interpolation or invented ground truth through missing blocks.
    return hourly, sources, pd.DataFrame(audit)


def segments(frame, min_hours=24):
    valid = frame[['load_kw','pv_kw','precio_eur_per_kwh','co2_kg_per_kwh']].notna().all(axis=1)
    group = (~valid).cumsum()
    return [part for _, part in frame[valid].groupby(group[valid]) if len(part) >= min_hours]


def simulate(agent, frame, export_price=0.):
    """Faithful arithmetic of supplied environment, including its baseline convention.
    Caller must pass one uninterrupted block. Resets SOC to 50% per block.
    """
    if len(frame)>1 and not (frame.index.to_series().diff().iloc[1:] == pd.Timedelta('1h')).all():
        raise ValueError('Simulation requires consecutive, complete hours')
    soc, eta, violations, rows = 1000., np.sqrt(.9), 0, []
    for i, (t, r) in enumerate(frame.iterrows()):
        last = frame.iloc[max(0, i-1)]
        future = frame.iloc[i:i+24]
        obs = dict(timestamp=t, hour=t.hour, day_of_week=t.dayofweek, soc_kwh=soc,
                   soc_pct=soc/2000., soc_percent=soc/20., preu_actual=r.precio_eur_per_kwh,
                   co2_actual=r.co2_kg_per_kwh,
                   preus_properes_24h=future.precio_eur_per_kwh.tolist(),
                   co2_properes_24h=future.co2_kg_per_kwh.tolist(),
                   noves_dades_5min=dict(ultim_consum_kw=last.load_kw, ultima_solar_kw=last.pv_kw))
        request = agent.act(obs)
        if not isinstance(request, float) or not np.isfinite(request):
            raise ValueError('Agent returned a nonfinite/non-float action')
        p = float(np.clip(request, -500., 500.))
        violations += int(p != request)
        actual = min(p, max(0., (2000.-soc)/eta)) if p>0 else -min(-p, max(0., soc*eta))
        violations += int(actual != p)
        soc += actual*eta if actual >= 0 else actual/eta
        grid = r.net_kw+actual
        rows.append(dict(Fecha=t, action_kw=actual, soc_kwh=soc, grid_kw=max(0., grid),
                         export_kw=max(0., -grid), net_kw=r.net_kw,
                         price=r.precio_eur_per_kwh, co2=r.co2_kg_per_kwh))
    trace = pd.DataFrame(rows).set_index('Fecha')
    baseline = np.maximum(frame.net_kw, 0.)
    result = dict(cost_base=float((baseline*frame.precio_eur_per_kwh).sum()),
                  co2_base=float((baseline*frame.co2_kg_per_kwh).sum()),
                  cost=float((trace.grid_kw*trace.price-trace.export_kw*export_price).sum()),
                  co2=float((trace.grid_kw*trace.co2).sum()), violations=violations,
                  hours=len(frame), final_soc=float(soc))
    return result, trace


def aggregate(results):
    total = {k:sum(r[k] for r in results) for k in ('cost_base','co2_base','cost','co2','violations','hours')}
    total['co2_saved_pct'] = 100*(1-total['co2']/total['co2_base']) if total['co2_base']>0 else 0.
    total['cost_saved_pct'] = 100*(1-total['cost']/total['cost_base']) if total['cost_base']>0 else 0.
    total['score'] = max(0., .7*total['co2_saved_pct']+.3*total['cost_saved_pct']-.2*total['violations'])
    return total
