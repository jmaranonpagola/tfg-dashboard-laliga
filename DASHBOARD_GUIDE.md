# Dashboard interactivo del TFG

Este dashboard usa Streamlit y se ejecuta desde `app.py`.

## Ejecutar en local

Instala dependencias:

```powershell
pip install -r requirements.txt
```

Si ya tienes `data/processed/players_scored.parquet`, lanza:

```powershell
streamlit run app.py
```

Si no tienes datos procesados, primero genera el dataset desde el notebook `codigotfg.ipynb`.

## Generar datos procesados

En el notebook ejecuta:

```python
player_stats, team_stats, competition_info = load_statsbomb_laliga()
master_df = build_master_dataset(player_stats, team_stats)
pca_results = fit_all_role_pcas(master_df)
players_modeled, cluster_results = fit_all_clusters(master_df, pca_results)
players_scored = calculate_impact_scores(players_modeled)
players_scored.to_parquet(PROCESSED_DIR / "players_scored.parquet", index=False)
```

## Subir a GitHub

Recomendado subir:

- `app.py`
- `tfg_pipeline.py`
- `requirements.txt`
- `statsbomb_column_mapping.json`
- `codigotfg.ipynb`
- `.streamlit/config.toml`
- `README_TFG.md`
- `DASHBOARD_GUIDE.md`

No subas por defecto:

- `data/raw/`
- `data/processed/`
- `.streamlit/secrets.toml`

Esto ya está configurado en `.gitignore`.

## Desplegar en Streamlit Community Cloud

1. Crea un repositorio en GitHub.
2. Sube los archivos del proyecto.
3. Entra en https://streamlit.io/cloud.
4. Conecta tu cuenta de GitHub.
5. Selecciona el repositorio.
6. Main file path: `app.py`.
7. En `Settings > Secrets`, añade:

```toml
SB_USERNAME = "tu_usuario"
SB_PASSWORD = "tu_password"
```

8. Despliega la app.

## Nota importante sobre datos

Si tus datos StatsBomb son privados o pertenecen a un acuerdo con un club/universidad, evita subir ficheros `parquet` procesados a un repositorio público. Usa repositorio privado o Streamlit secrets.

## Añadir datos económicos

La app puede incorporar valor de mercado, salarios o presupuestos, pero no scrapea Transfermarkt automáticamente. Lo recomendable es usar un CSV descargado de una fuente permitida o construido manualmente.

### Opción automática con Kaggle

Puedes probar a descargar el dataset público `davidcariboo/player-scores`, que en Kaggle aparece con licencia CC0.

Instala dependencias:

```powershell
pip install -r requirements.txt
```

Ejecuta:

```powershell
python scripts/download_kaggle_economics.py
```

El script crea:

```text
data/external/player_market_values.csv
```

Después lanza de nuevo:

```powershell
streamlit run app.py
```

Si Kaggle te pide autenticación, inicia sesión en Kaggle o configura tu API token. En algunos entornos `kagglehub` permite descargar datasets públicos directamente; en otros pide credenciales.

Coloca un archivo en:

```text
data/external/player_market_values.csv
```

Columnas recomendadas:

```csv
Name,Team,market_value_eur,annual_wage_eur,contract_until,age
Jugador Ejemplo,Equipo Ejemplo,10000000,1200000,2028,24
```

También puedes añadir presupuestos de equipo en:

```text
data/external/team_budgets.csv
```

Formato:

```csv
Team,budget_eur,squad_cost_eur
Equipo Ejemplo,80000000,120000000
```

Cuando existan esos archivos, el dashboard mostrará la pestaña `Economía` con:

- impacto deportivo vs valor de mercado
- impacto por millón de euros
- oportunidades de valor
- valor agregado de plantilla vs impacto medio

El cruce económico se hace por nombre y equipo. Como StatsBomb y Kaggle/Transfermarkt usan nombres distintos (`Eric García Martret` frente a `Eric García`, o `Futbol Club Barcelona` frente a `Barcelona`), el pipeline aplica normalización, alias de clubes y matching aproximado. Aun así, no se imputan valores inventados: si no hay coincidencia fiable, el jugador queda sin valor económico.

En la pestaña `Encaje`, si activas economía:

- se penaliza a los jugadores que superan el presupuesto objetivo
- se descartan automáticamente los jugadores cuyo valor es igual o superior a `x` veces ese presupuesto
- por defecto, `x = 2`, es decir, si el presupuesto objetivo es 4M, se descartan jugadores de 8M o más

Además, si el CSV económico contiene `age`, la app permite activar la edad en el `fit_score`:

- no se penaliza hasta la edad ideal máxima configurada
- después la puntuación `age_fit` baja progresivamente
- esto representa menor valor de reventa, menor horizonte de proyecto y mayor riesgo físico
- se puede desactivar si buscas rendimiento inmediato sin penalizar veteranos
