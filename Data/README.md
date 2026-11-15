# UPC Battery Lab — proyecto completo

## Resultado real

La política seleccionada obtuvo **7,389 puntos** en **955 horas completas fuera del entrenamiento**, con **7,888% menos CO₂**, **6,225% de ahorro económico** y **0 infracciones**. Son tres bloques de julio–agosto de 2026 (576, 80 y 299 horas), cada uno iniciado al 50% de SOC. No es una prueba continua de 2.016 horas ni un resultado del servidor UPC.

Una programación lineal con conocimiento perfecto de todo el futuro y relajaciones favorables al controlador establece un **límite superior de 7,420 puntos para esos mismos bloques**, con la tarifa de exportación cero del simulador. Por tanto, exigir 10–15 puntos en esta validación no es compatible con sus datos y restricciones. El conjunto secreto puede tener otra distribución; no se garantiza su puntuación.

El límite es un diagnóstico separado: `oracle_bound.py` nunca interviene en el agente. La comparación incluye la energía inicial gratuita que proporciona el entorno a cada bloque. No impone SOC terminal mínimo, igual que el evaluador suministrado.

## Inicio rápido

Extrae `upc_battery_project.zip`, entra en la carpeta resultante y ejecuta:

```bash
python -m pip install -r requirements.txt
streamlit run dashboard.py
```

El paquete incluye los seis CSV, el modelo entrenado, los resultados y todos los códigos. El dashboard ya funciona sin volver a entrenar.

Para repetir el entrenamiento completo:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python train.py
python oracle_bound.py
python test_agent.py
```

Las variables de entorno limitan los hilos en entrenamiento local, especialmente para HistGradientBoosting. No se usan llamadas a sistemas operativos en el código del agente.

Para comprobar el ZIP de entrega con el verificador que recibiste:

```bash
python /ruta/al/starter_kit/check_submission.py submission.zip
```

**Entrega `submission.zip`, no el ZIP del proyecto completo.** Contiene solo `agent.py` y `model.joblib` en su raíz. El dashboard y los datasets no se envían al evaluador. Si reentrenas, vuelve a comprimir esos dos archivos con el gestor de archivos de tu ordenador. No cargues modelos joblib de fuentes desconocidas.

## Archivos

| Archivo | Función |
|---|---|
| `agent.py` | Clase `BatteryAgent`, funciones de features/inferencia y control completo |
| `model.joblib` | Modelo y configuración final reentrenados con todos los datos publicados |
| `data_pipeline.py` | Lectura, auditoría, agregación horaria y simulación local |
| `train.py` | Ingeniería temporal, búsqueda, selección y evaluación retenida |
| `dashboard.py` | Aplicación Streamlit independiente |
| `oracle_bound.py` | Cota diagnóstica con futuro perfecto, fuera del agente |
| `test_agent.py` | Pruebas de seguridad física y causalidad |
| `results/validated_model.joblib` | Modelo anterior a julio, usado para el holdout |
| `results/summary.json` | Resultados medidos y política elegida |
| `results/cv_models.csv` | Cada configuración y cada fold |
| `results/cv_policies.csv` | Comparación de control con/sin ML y valor terminal |
| `results/holdout_trace.csv` | Acciones, SOC, importaciones, precios y emisiones |
| `results/oracle_bound.json` | Cota superior y alcance |
| `results/data_audit.csv` | Filas, duplicados y celdas vacías por fuente |

## Fase 1 — datos y dashboard

Los CSV cubren un intervalo de 8.784 horas de reloj, pero solo **5.993 horas** tienen simultáneamente consumo, solar, precio y CO₂. Precio y CO₂ tienen 6.768 filas, con huecos mensuales; el inversor termina antes que otras fuentes. Las copias con `(1)` son idénticas a los archivos con nombre canónico incluidos.

Se conserva un índice horario completo. Se agregan potencias con media horaria, se agrupan timestamps repetidos con media y no se rellenan huecos con observaciones futuras ni se fabrican etiquetas. La suma fotovoltaica exige los tres inversores. Los tests se dividen en bloques completos sin saltar artificialmente de un mes a otro.

El dashboard ofrece:

- Filtros de fechas, mes, día de semana y hora.
- Potencia del campus, solar, demanda neta, precio y CO₂.
- Inspección individual de los seis datasets y sus variables originales.
- Perfiles horarios/semanales/mensuales y explicaciones calculadas para la selección.
- Relación precio–CO₂, correlación con desfases y comparación entre sensores de distintas fuentes.
- Cobertura mensual y huecos visibles.
- Resultados de modelos, acciones y SOC del holdout.
- Laboratorio interactivo de 48 horas con compensación de excedentes y valor terminal configurables.

Los gráficos de datos filtrados son descriptivos. Un desfase negativo en la exploración no autoriza a usar mediciones futuras en entrenamiento. Las tablas meteorológicas y digitales ayudan a explorar relaciones, pero no se usan como covariables futuras en el agente porque el contrato suministrado no las entrega.

## Fase 2 — qué se predice y por qué

El entorno ya proporciona las próximas 24 observaciones de precio y CO₂, **incluida la hora actual en la posición cero**. Sustituirlas por una predicción aprendida perdería información. El objetivo del modelo es la demanda neta `load_kw - pv_kw`, necesaria para anticipar el punto donde la descarga empieza a exportar en lugar de reducir importaciones.

El modelo único predice varios horizontes con el horizonte como feature. Entrena en 0, 1, 3, 6, 12, 18 y 23 horas y se consulta en los 24 horizontes del control. Los intermedios son interpolación del modelo, no modelos independientes entrenados para cada horizonte.

Features comunes a entrenamiento y producción:

- Retardos de demanda neta de 1, 2, 24, 48 y 168 horas.
- Media, desviación y proporción observada en las últimas 24/168 horas.
- Retardos de precio/CO₂ y correlación histórica entre ambos.
- Hora, día semanal, mes, seno/coseno de hora y época del año.
- Demanda de ayer y de hace una semana alineadas con la hora objetivo.
- Precio y CO₂ del horizonte, conocidos por el contrato del evaluador.

No se usa la demanda de la hora actual como feature offline. En la primera observación de cada bloque, el entorno oficial entrega excepcionalmente la telemetría actual como si fuera la anterior; el agente acepta exactamente esa observación. No lee filas ocultas. Después recibe solo la hora anterior.

Se emplean tres folds `TimeSeriesSplit` sobre orígenes ordenados, con `gap=24` orígenes y un filtro adicional que impide que las etiquetas futuras de entrenamiento sobrepasen su fin. Con muestreo cada dos horas, el embargo efectivo suele ser de al menos 48 horas. Julio y agosto se reservan antes de seleccionar. Los datos que faltan permanecen como NaN en features; los tres estimadores los soportan. La estacionalidad cae a la última medida o a 1.000 kW cuando no existe historia.

### Comparación medida

Se completaron **10 configuraciones × 3 folds = 30 ajustes**: cuatro LightGBM, cuatro XGBoost y dos HistGradientBoosting. Es una búsqueda finita reproducible sobre profundidades y tasas de aprendizaje, con regularización conservadora; no una búsqueda exhaustiva de todos los hiperparámetros posibles.

| Familia (mejor configuración según criterio robusto) | MAE medio CV, kW |
|---|---:|
| HistGradientBoosting | 96,28 |
| LightGBM | 98,46 |
| XGBoost | 101,64 |
| Referencia estacional | 185,56 |

El criterio de selección predictiva es `MAE medio + 0,25 × desviación entre folds`. Gana HistGradientBoosting con profundidad 3, learning rate 0,06, regularización L2=10 y 150 iteraciones. Se prefiere por su evidencia de validación, no por una supuesta superioridad universal. La búsqueda puede ampliarse en `CONFIGS`, pero deberá conservarse un nuevo conjunto final sin utilizar.

Después se compara el controlador con predicción aprendida frente a referencia estacional y factores de valor terminal 0,5/0,8/1,0. La selección usa el score medio menos 0,25 veces su desviación. Se usan hasta 168 horas completas finales de cada fold para esta comparación, no todo su período: es una limitación del presupuesto de validación.

**La política ganadora usa referencia estacional (`model_weight=0`) y factor terminal 0,5.** El código integra plenamente la predicción aprendida y el modelo entrenado se entrega, pero la configuración ganadora no lo consulta para decidir. En el holdout, activar ML daría 7,38955 frente a 7,38904 de la seleccionada: una diferencia irrelevante y observada después de seleccionar, por lo que no se cambia el ganador.

La explicación física es que la demanda neta suele ser mayor que los 500 kW máximos de descarga. Mientras la batería no exporta, una mejor previsión de demanda apenas cambia el beneficio marginal; mandan precio, CO₂ y restricciones energéticas. La innovación útil aquí es combinar una validación causal con control explícito y comprobar cuándo ML aporta valor, sin atribuirle mejoras que no existen.

## Fase 3 — optimización y contrato

`BatteryAgent().act(obs)` devuelve un `float` finito: positivo carga, negativo descarga, cero reposo, entre −500 y +500 kW.

Con `eta=sqrt(0,9)` y pasos de una hora:

- Si `u >= 0`: `E_siguiente = E + eta*u`.
- Si `u < 0`: `E_siguiente = E + u/eta`.
- `G = demanda_neta + u`.
- Importación: `max(G,0)`; exportación: `max(-G,0)`.

El objetivo horario empleado es:

`0,70 * co2[h] * importacion / Bco2 + 0,30 * (precio[h] * importacion - tarifa_exportacion * exportacion) / Bcoste`

`Bco2` y `Bcoste` son estimaciones de emisiones y coste del baseline por hora, calculadas únicamente con el entrenamiento. La división es necesaria porque la puntuación combina **reducciones porcentuales**, no magnitudes en unidades distintas. Los totales reales del período secreto son desconocidos; los factores de entrenamiento son una aproximación a esos denominadores. La cota oráculo sí utiliza los denominadores exactos de sus bloques, exclusivamente como diagnóstico.

La programación dinámica usa 81 estados entre 0 y 2.000 kWh, separados 25 kWh. Cada transición representa una sola acción con signo; por construcción no carga y descarga simultáneamente. Admite precios negativos y costes distintos para exportación sin necesitar supuestos convexos. Una valoración de energía al final reduce el vaciado artificial del horizonte de 24 horas. La discretización y el valor terminal son aproximaciones, no una prueba de optimalidad del agente.

La primera acción parte del SOC continuo real, se ejecuta y el resto del plan se descarta para recalcular en la siguiente observación. Hay limitación final independiente de potencia y energía, con margen numérico de `1e-7` kW para evitar penalizaciones por redondeo.

## Discrepancias documentales y alcance

1. Las capturas dicen 0,04 €/kWh de exportación; `CampusBatteryEnv` usa **0 por defecto**. La entrega conserva ese valor y reporta también 0,04: en este holdout el score sigue siendo 7,39.
2. El evaluador resta exportaciones en el coste del participante, pero no las incluye en el coste baseline. Se replica exactamente para comparar, sin presentarlo como una definición universal de ahorro eléctrico.
3. El contador aparece descrito como demanda neta en una imagen, pero la receta lo convierte en demanda y después resta la solar. Aquí se sigue esa receta. Si el contador físico ya mide neto, habría que aclarar el doble descuento con los organizadores antes de interpretar físicamente los resultados.
4. El archivo de CO₂ ofrece intensidad y generación agregada con/sin emisiones; no identifica de forma separada todas las centrales de ciclo combinado. Se optimiza el factor exacto que usa el evaluador.
5. Los vectores futuros locales son valores reales. Si en el servidor son pronósticos con error o están desplazados a `t+1`, el rendimiento/contrato deberá reevaluarse. No se asume que el test secreto tenga las mismas condiciones.
6. El modelo final se reentrena sobre todos los datos publicados después de evaluar. Las métricas pertenecen al modelo y configuración previos al corte, conservados aparte, no a una evaluación futura del refit.
7. No se ha enviado nada a la web UPC. El verificador local no garantiza que el entorno remoto tenga las mismas versiones. Se incluyen versiones del entorno utilizado; los artefactos scikit-learn requieren compatibilidad.

## Verificación realizada

- Verificador oficial AST y contrato de `agent.py` y del ZIP de entrega.
- Pruebas de SOC vacío/lleno/casi límite y 200 casos de límites físicos.
- Prueba de causalidad: alterar la telemetría futura no altera features del origen pasado.
- Vectores de predicción vacíos: fallback finito.
- Comparación de acciones, SOC y red del simulador local con el `CampusBatteryEnv` original en 168 horas: coinciden.
- Prueba automática Streamlit: las cinco pestañas se ejecutan sin excepciones y el botón del escenario funciona.

Fuentes primarias: los seis CSV, `battery_spec.json`, `environment.py`, `check_submission.py`, la plantilla `agent.py` y las capturas adjuntas. La [web del reto](https://repteupc.datathon.cat/) no pudo recuperarse desde la herramienta de consulta; no se han inventado reglas adicionales del servidor.
