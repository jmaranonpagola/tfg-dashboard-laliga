# TFG - Dashboard de impacto, perfiles y encaje

Esta carpeta contiene una versión preparada para ejecutar la aplicación del TFG sin depender de archivos externos del ordenador original.

## Archivos principales

- `app.py`: aplicación Streamlit.
- `tfg_pipeline.py`: carga de datos, limpieza, clustering, impactos, similitud de equipos y cálculo de encaje.
- `codigotfg.ipynb`: notebook reproducible/resumen técnico del pipeline.
- `data/processed/players_scored.parquet`: dataset principal ya procesado que usa la app.
- `data/raw/`: datos base cacheados de StatsBomb.
- `data/external/`: datos auxiliares de Transfermarkt, valores de mercado y diccionarios de enlace.
- `assets/`: escudos, logo de LaLiga e imágenes locales de jugadores.
- `statsbomb_column_mapping.json`: renombrado de variables.
- `statsbomb_metric_descriptions_es.json`: descripciones en español de las variables.

## Instalación

Se recomienda Python 3.11 o 3.12.

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

En macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Acceso a la aplicación

La pantalla inicial permite iniciar sesión o registrar nuevos usuarios.

Las contraseñas se guardan localmente en `data/external/app_users.json` como hash PBKDF2 con sal, no en texto plano.

Usuario inicial:

Usuario:

```text
AlumnoOsasuna
```

Contraseña:

```text
Osasuna2526
```

## Informes PDF y correo

La descarga de informes PDF funciona desde la aplicación. Para generar PDFs con el mismo estilo visual de la app, el sistema debe tener instalado Microsoft Edge o Google Chrome, porque Streamlit genera el PDF desde HTML.

El envío por correo es opcional. Si no se configuran credenciales, la app permite descargar el PDF igualmente. Para activar el envío por Gmail hay que crear un archivo `.streamlit/secrets.toml` con una contraseña de aplicación de Google. No se incluye ningún archivo de secretos en esta entrega.

Ejemplo:

```toml
[email]
email_username = "tu_correo@gmail.com"
email_password = "clave_de_16_caracteres_de_google"
```

Para obtener esa clave, activa la verificación en dos pasos de Google y crea una contraseña de aplicación para la app.

## Reproducibilidad

La app está preparada para abrirse directamente con el dataset procesado incluido. Si se quisiera reconstruir todo desde cero usando StatsBomb, habría que configurar credenciales de StatsBomb mediante variables de entorno o secrets:

```text
SB_USERNAME
SB_PASSWORD
```

Para evaluar la aplicación no es necesario reconstruir los datos, porque `data/processed/players_scored.parquet` ya está incluido.
