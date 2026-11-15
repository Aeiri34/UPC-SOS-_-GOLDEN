"""Independent dashboard. Launch: streamlit run dashboard.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from agent import BatteryAgent
from data_pipeline import load_data, read_raw, NAMES, segments, simulate, aggregate

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='UPC · Battery Lab', page_icon='⚡', layout='wide')
st.markdown('''<style>
.stApp {background: #0b1120; color: #e5edf7;}
[data-testid="stMetric"] {background:#162237; border:1px solid #29415b; padding:18px; border-radius:14px;}
.block-container {padding-top:2rem;}
h1,h2,h3 {letter-spacing:-.025em;}
</style>''', unsafe_allow_html=True)
st.caption('UPC DATATHON 2026  /  ENERGÍA · DATOS · DECISIONES')
st.title('⚡ Battery Lab')
st.write('De los sensores a una decisión por hora. Explora el campus, contrasta los modelos y comprueba el impacto medido.')


@st.cache_data
def cached_data(folder):
    return load_data(folder)


@st.cache_data
def cached_raw(folder, name):
    return read_raw(folder, name)


folder = st.sidebar.text_input('Carpeta de los seis CSV', str(ROOT/'data'))
try:
    data, sources, audit = cached_data(folder)
except (FileNotFoundError, KeyError, ValueError) as error:
    st.error(f'No se pudieron cargar los datos: {error}')
    st.stop()
st.sidebar.header('Explorar un período')
months = st.sidebar.multiselect('Mes', list(range(1,13)), default=list(range(1,13)))
days = st.sidebar.multiselect('Día (0=lunes, 6=domingo)', list(range(7)), default=list(range(7)))
hours = st.sidebar.slider('Horas del día', 0, 23, (0,23))
dates = st.sidebar.date_input('Fechas', (data.index.min().date(), data.index.max().date()),
                              min_value=data.index.min().date(), max_value=data.index.max().date())
mask = data.index.month.isin(months) & data.index.dayofweek.isin(days) & (data.index.hour>=hours[0]) & (data.index.hour<=hours[1])
if len(dates)==2:
    mask &= (data.index.date>=dates[0]) & (data.index.date<=dates[1])
view = data.loc[mask]
if view.empty:
    st.info('Selecciona al menos una fecha, un mes y un día con datos.'); st.stop()
valid = view.dropna()
a, b, c, d = st.columns(4)
a.metric('Horas con las 4 señales', f'{len(valid):,}')
b.metric('Precio medio', f'{view.precio_eur_per_kwh.mean():.3f} €/kWh')
c.metric('Intensidad media', f'{view.co2_kg_per_kwh.mean():.3f} kg/kWh')
d.metric('Demanda neta media', f'{view.net_kw.mean():,.0f} kW')
st.caption('Las medias describen las horas observadas. Los huecos se conservan; no se sustituyen por ceros.')
explore, relations, quality, model_tab, control = st.tabs(['01 · Series', '02 · Patrones', '03 · Calidad', '04 · Modelos', '05 · Control'])

with explore:
    st.subheader('Seis fuentes, una línea temporal')
    picked = st.multiselect('Señales del campus', ['load_kw','pv_kw','net_kw'], default=['load_kw','pv_kw'])
    if picked:
        st.line_chart(view[picked], y_label='kW', height=280)
    left, right = st.columns(2)
    with left:
        st.line_chart(view[['precio_eur_per_kwh']], y_label='€/kWh', height=230)
    with right:
        st.line_chart(view[['co2_kg_per_kwh']], y_label='kg CO₂/kWh', height=230)
    name = st.selectbox('Explorar un dataset individual', NAMES)
    numeric = sources[name]
    available = list(numeric.columns)
    default = [x for x in available if 'POTENCIA ACTIVA (kW)' in x or x in ['precio_eur_per_kwh','co2_kg_per_kwh']][:3] or available[:1]
    columns = st.multiselect('Variables originales (unidades en las cabeceras)', available, default=default)
    if columns:
        st.line_chart(numeric.reindex(view.index)[columns], height=250)
    with st.expander('Tabla original y muestreo'):
        raw = cached_raw(folder, name)
        st.write(f'{len(raw):,} filas originales; gráfico agregado mediante media horaria.')
        st.dataframe(raw.head(200), width='stretch')
    st.download_button('Descargar selección horaria CSV', view.to_csv(), 'seleccion_horaria.csv', 'text/csv')

with relations:
    st.subheader('Cuándo es más limpia y más barata la electricidad')
    group = st.radio('Agrupar por', ['Hora','Día de semana','Mes'], horizontal=True)
    group_key = {'Hora':view.index.hour, 'Día de semana':view.index.dayofweek, 'Mes':view.index.month}[group]
    grouped = view.groupby(group_key)[['precio_eur_per_kwh','co2_kg_per_kwh']].mean()
    left, right = st.columns(2)
    with left:
        st.bar_chart(grouped[['precio_eur_per_kwh']], y_label='€/kWh')
    with right:
        st.bar_chart(grouped[['co2_kg_per_kwh']], y_label='kg CO₂/kWh')
    st.dataframe(view.groupby(group_key).agg(hours=('net_kw','count'), price=('precio_eur_per_kwh','mean'), co2=('co2_kg_per_kwh','mean')), width='stretch')
    pairs = view[['precio_eur_per_kwh','co2_kg_per_kwh']].dropna()
    if len(pairs)>2:
        corr = pairs.corr().iloc[0,1]
        st.info(f'En esta selección, la correlación precio–CO₂ es {corr:.2f}. Describe asociación, no causalidad; un precio bajo no garantiza emisiones bajas.')
        st.scatter_chart(pairs, x='precio_eur_per_kwh', y='co2_kg_per_kwh', height=260)
    if grouped.notna().all().all() and len(grouped):
        st.success(f'El grupo con menor precio medio es {grouped.precio_eur_per_kwh.idxmin()}; el de menor CO₂ es {grouped.co2_kg_per_kwh.idxmin()}. Compara ambos antes de programar una carga.')
    st.markdown('**Correlación con desfase** · precio(t) frente a CO₂(t − k).')
    lag = st.slider('Desfase k, en horas', -48,48,0)
    # Shift on the full hourly grid FIRST: filtering must not turn gaps into adjacent hours.
    shifted = data.co2_kg_per_kwh.shift(lag).reindex(view.index)
    lagcorr = view.precio_eur_per_kwh.corr(shifted)
    st.metric('Correlación en el desfase seleccionado', f'{lagcorr:.3f}')
    st.caption('Los desfases negativos son exploratorios; no se usan como información futura medida en el modelo.')
    mix = sources['co2_red'].reindex(view.index)
    mix_cols = [c for c in mix if c.startswith('gen_')]
    if mix_cols:
        st.area_chart(mix[mix_cols], y_label='MW', height=230)
    st.markdown('**Relación entre sensores de fuentes distintas**')
    first, second = st.columns(2)
    with first:
        source_x = st.selectbox('Fuente X', NAMES, index=2)
        cols_x = list(sources[source_x].columns)
        col_x = st.selectbox('Variable X', cols_x, index=next((i for i,c in enumerate(cols_x) if c.endswith('IRRADIANCIA (W/m2)')),0))
    with second:
        source_y = st.selectbox('Fuente Y', NAMES, index=1)
        cols_y = list(sources[source_y].columns)
        col_y = st.selectbox('Variable Y', cols_y, index=next((i for i,c in enumerate(cols_y) if c.endswith('POTENCIA ACTIVA (kW)')),0))
    paired = pd.concat([sources[source_x][col_x].rename('X'), sources[source_y][col_y].rename('Y')], axis=1).reindex(view.index).dropna()
    if len(paired)>2:
        st.scatter_chart(paired, x='X', y='Y', height=230)
        st.caption(f'{len(paired):,} horas compartidas; correlación {paired.X.corr(paired.Y):.3f}. X: {col_x}. Y: {col_y}.')
    st.caption('La intensidad agregada de CO₂ es el objetivo del simulador. Estos campos no aíslan por sí solos la generación de ciclo combinado.')

with quality:
    st.subheader('Datos observados, límites visibles')
    st.dataframe(audit, width='stretch')
    coverage = data[['load_kw','pv_kw','precio_eur_per_kwh','co2_kg_per_kwh']].notna().groupby(data.index.to_period('M').astype(str)).sum()
    st.bar_chart(coverage, stack=False, y_label='Horas observadas')
    st.warning('No existe un año continuo de telemetría. La validación reinicia el entorno por bloque completo; sus resultados no equivalen a una prueba secreta continua de 2.016 horas.')
    st.write('El consumo sigue la receta del reto: −potencia exportada del contador, limitada a [0,5000]. Solar: suma de la potencia activa de los tres inversores. Se exige disponibilidad de los tres.')
    st.write('La documentación llama al contador tanto demanda como demanda neta. Se mantiene la receta del evaluador; si el contador ya descuenta la solar en la instalación real, habría que aclararlo con la organización.')
    block_rows = [dict(inicio=str(p.index[0]), fin=str(p.index[-1]), horas=len(p)) for p in segments(data)]
    st.dataframe(pd.DataFrame(block_rows), width='stretch')

summary_path = ROOT/'results/summary.json'
summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
with model_tab:
    st.subheader('Predicción causal y validación temporal')
    st.write('Se predice demanda neta, no precios ni CO₂ ya proporcionados. Retardos 1/2/24/48/168 h, medias y dispersión históricas, calendario y previsiones oficiales del mercado.')
    st.write('Tres folds temporales con embargo de etiquetas multihorizonte. La selección se hace en desarrollo; julio–agosto se reserva antes de entrenar. Los huecos no se rellenan con datos futuros.')
    if summary:
        st.json({'configuración':summary['chosen_config'], 'política':summary['policy'], 'inicio_holdout':summary['cutoff']})
        a,b = st.columns(2)
        a.metric('MAE modelo en CV', f"{summary['cv_forecast_mae']:.1f} kW")
        b.metric('MAE referencia estacional', f"{summary['cv_seasonal_mae']:.1f} kW")
        st.dataframe(pd.read_csv(ROOT/'results/cv_models.csv'), width='stretch')
        st.write('Se comparan 10 configuraciones × 3 folds. Es una búsqueda finita reproducible; no una garantía de óptimo global.')
        st.dataframe(pd.read_csv(ROOT/'results/cv_policies.csv'), width='stretch')
    else:
        st.info('Ejecuta python train.py para generar resultados medidos.')

with control:
    st.subheader('Impacto medido y laboratorio de control')
    st.latex(r'S=\max(0,\;0.70\,\Delta CO_2(\%)+0.30\,\Delta Coste(\%)-0.20\,N_{infracciones})')
    if summary:
        metric = summary['selected_export_0.0']
        a,b,c,d = st.columns(4)
        a.metric('Score holdout local', f"{metric['score']:.2f}")
        b.metric('Reducción CO₂', f"{metric['co2_saved_pct']:.2f}%")
        c.metric('Ahorro económico', f"{metric['cost_saved_pct']:.2f}%")
        d.metric('Infracciones', metric['violations'])
        bound_path = ROOT/'results/oracle_bound.json'
        if bound_path.exists():
            bound = json.loads(bound_path.read_text())['perfect_future_relaxed_score_upper_bound']
            st.warning(f'Límite superior con futuro perfecto en estos bloques: {bound:.2f} puntos. El objetivo 10–15 no es alcanzable aquí; no extrapolar este límite al test secreto.')
        st.caption(f"{metric['hours']} horas completas, agregadas por sumas de costes/emisiones. Modelo validado antes de {summary['cutoff']}. No es el resultado del servidor UPC.")
        trace = pd.read_csv(ROOT/'results/holdout_trace.csv', parse_dates=['Fecha']).set_index('Fecha')
        chosen = st.selectbox('Bloque de validación', sorted(trace.block_id.unique()))
        section = trace[trace.block_id==chosen]
        st.line_chart(section[['net_kw','grid_kw','action_kw']], y_label='kW', height=260)
        st.area_chart(section[['soc_kwh']], y_label='kWh', height=180)
        holdout = data.loc[data.index>=pd.Timestamp(summary['cutoff'])]
        blocks = segments(holdout, 48)
        idx = st.selectbox('Escenario interactivo (primeras 48 h)', range(len(blocks)), format_func=lambda i:str(blocks[i].index[0]))
        terminal = st.slider('Valor relativo de energía al final del horizonte', .0,1.5,float(summary['policy']['terminal_factor']),.1)
        export = st.selectbox('Compensación de excedentes €/kWh', [0.,.04])
        if st.button('Simular escenario', type='primary'):
            saved = joblib.load(ROOT/'results/validated_model.joblib')
            agent = BatteryAgent()
            agent.model = saved['model']; agent.cost_scale = saved['cost_scale']; agent.co2_scale = saved['co2_scale']
            agent.model_weight = saved['model_weight']; agent.terminal_factor = terminal; agent.export_price = export
            result, local_trace = simulate(agent, blocks[idx].iloc[:48], export)
            st.json(aggregate([result])); st.line_chart(local_trace[['action_kw','grid_kw']])
            st.caption('Escenario exploratorio posterior a la selección. No sustituye el holdout ni valida nuevos hiperparámetros.')
    st.info('La programación dinámica compara acciones físicamente posibles sobre estados separados 25 kWh, ejecuta solo la primera y replantea cada hora. El modelo final se reentrena con los datos publicados; su resultado futuro todavía no se conoce.')
