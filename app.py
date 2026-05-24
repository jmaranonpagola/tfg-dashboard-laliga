from __future__ import annotations

import os
import logging
import base64
import hashlib
import hmac
import io
import html
import json
import re
import shutil
import subprocess
import smtplib
import secrets
import tempfile
import textwrap
import unicodedata
import uuid
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from urllib.request import urlopen

os.environ.setdefault("OMP_NUM_THREADS", "1")

logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent
LALIGA_LOGO_PATH = BASE_DIR / "assets" / "logos" / "laliga.png"
BADGES_DIR = BASE_DIR / "assets" / "escudos"
PLAYER_IMAGES_DIR = BASE_DIR / "assets" / "jugadores"
FLAGS_DIR = BASE_DIR / "assets" / "banderas"
TRANSFERMARKT_EXTERNAL_PATH = BASE_DIR / "data" / "external" / "Combined_Transfermarkt (5).xlsx"
METRIC_DESCRIPTIONS_PATH = BASE_DIR / "statsbomb_metric_descriptions.json"
METRIC_DESCRIPTIONS_ES_PATH = BASE_DIR / "statsbomb_metric_descriptions_es.json"
AUTH_USERS_PATH = BASE_DIR / "data" / "external" / "app_users.json"
ATHLETIC_ELIGIBLE_PATH = BASE_DIR / "data" / "external" / "athletic_eligible_players.csv"
REPORT_LISTS_PATH = BASE_DIR / "data" / "external" / "report_lists.json"
SAVED_REPORTS_DIR = BASE_DIR / "data" / "external" / "saved_reports"
PASSWORD_HASH_ITERATIONS = 200_000

MANUAL_NATIONALITIES = {
    "Abdulay Juma Bah": "Sierra Leone",
    "Yan Diomande": "Cote d'Ivoire",
    "Adam Aznou Ben Cheikh": "Morocco",
    "Tamás Nikitscher": "Hungary",
    "Antonio Candela": "Italy",
    "Juan Camilo Hernández Suárez": "Colombia",
    "Antony Matheus dos Santos": "Brazil",
    "Arthur Henrique Ramos de Oliveira Melo": "Brazil",
    "Ruben Vargas": "Switzerland",
    "Facundo Tomás Garcés": "Argentina",
    "Florian Grillitsch": "Austria",
    "Maroan Sannadi Harrouch": "Spain",
    "Coba Gomes da Costa": "Guinea-Bissau",
    "Juan Herzog": "Spain",
    "Raúl Asencio del Rosario": "Spain",
    "Stefan Bajcetic Maquieira": "Spain",
    "Fernando López González": "Spain",
    "Jesús Rodriguez Caraballo": "Spain",
    "Yoel Lago Amil": "Spain",
    "Jon Martín Vicente": "Spain",
    "Pau Navarro Badenes": "Spain",
    "Roberto Fernández Jaén": "Spain",
}

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tfg_pipeline import (
    METRIC_SUBGROUPS,
    NEGATIVE_METRICS,
    PROCESSED_DIR,
    add_economic_data,
    assign_cluster_labels,
    build_master_dataset,
    calculate_impact_scores,
    calculate_team_similarity,
    ensure_model_role,
    existing_columns,
    fit_all_clusters,
    fit_all_role_pcas,
    load_statsbomb_laliga,
    normalize_name,
    player_team_fit,
    similar_teams,
)


st.set_page_config(
    page_title="TFG StatsBomb | Impacto y encaje",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROLE_LABELS = {
    "POR": "Porteros",
    "DEF": "Centrales",
    "LAT": "Laterales",
    "MED": "Mediocentros",
    "EXT": "Extremos",
    "DEL": "Delanteros",
}

DISPLAY_NAMES = {
    "Name": "Jugador",
    "Team": "Equipo",
    "Primary Position": "Posición principal",
    "Secondary Position": "Posición secundaria",
    "role": "Rol",
    "profile_cluster": "Cluster técnico",
    "cluster_label": "Perfil futbolístico",
    "Minutes": "Minutos",
    "age": "Edad",
    "birth_date": "Fecha de nacimiento",
    "market_value_eur": "Valor de mercado",
    "market_value_million_eur": "Valor de mercado (M EUR)",
    "player_height": "Altura",
    "Height": "Altura",
    "contract_until": "Fin de contrato",
    "fit_score": "Puntuación de encaje",
    "context_fit": "Encaje táctico",
    "team_need_fit": "Necesidad del equipo",
    "role_fit": "Encaje de rol",
    "economic_fit": "Encaje económico",
    "age_fit": "Encaje por edad",
    "source_team_similarity": "Similitud club origen-destino",
    "ranking_reason": "Motivo de recomendación",
    "impacto_global": "Impacto global",
    "impacto_ofensivo": "Impacto ofensivo",
    "impacto_asociativo": "Impacto asociativo",
    "impacto_defensivo": "Impacto defensivo",
    "impacto_porteria": "Impacto de portería",
    "impacto_finalizacion": "Finalización",
    "impacto_amenaza_area": "Amenaza en área",
    "impacto_creacion": "Creación",
    "impacto_progresion_pase": "Progresión por pase",
    "impacto_progresion_conduccion": "Progresión por conducción",
    "impacto_circulacion": "Circulación",
    "impacto_duelos": "Duelos",
    "impacto_anticipacion": "Anticipación",
    "impacto_defensa_area": "Defensa de área",
    "impacto_presion": "Presión",
    "impacto_shot_stopping": "Paradas",
    "impacto_porteria_juego_aereo": "Juego aéreo del portero",
    "impacto_porteria_distribucion": "Distribución del portero",
    "similitud_coseno": "Similitud de estilo",
    "Team_PC1": "Componente táctica 1",
    "Team_PC2": "Componente táctica 2",
    "Team_PC3": "Componente táctica 3",
    "cluster": "Grupo de equipos",
    "PC1": "Componente 1",
    "PC2": "Componente 2",
    "jugadores": "Jugadores",
    "minutos_medios": "Minutos medios",
    "impacto_medio": "Impacto medio",
    "perfil": "Perfil",
    "fortalezas_relativas": "Fortalezas relativas",
    "debilidades_relativas": "Debilidades relativas",
}

RANKING_COLOR_SCALE = [
    (0.00, "#b91c1c"),  # peor: rojo
    (0.35, "#f97316"),  # bajo: naranja
    (0.65, "#fde047"),  # medio: amarillo
    (1.00, "#15803d"),  # mejor: verde
]

SUBIMPACT_LABELS = {
    "finalizacion": "Finalizacion",
    "amenaza_area": "Amenaza en area",
    "creacion": "Creacion",
    "progresion_pase": "Progresion por pase",
    "progresion_conduccion": "Progresion por conduccion",
    "circulacion": "Circulacion",
    "duelos": "Duelos",
    "anticipacion": "Anticipacion",
    "defensa_area": "Defensa de area",
    "presion": "Presion",
    "shot_stopping": "Paradas",
    "porteria_juego_aereo": "Juego aereo",
    "porteria_distribucion": "Distribucion",
}

COMPARE_PLAYER_A_KEY = "compare_player_a"
COMPARE_PLAYER_B_KEY = "compare_player_b"
COMPARE_PLAYER_KEYS = [
    COMPARE_PLAYER_A_KEY,
    COMPARE_PLAYER_B_KEY,
    "compare_player_c",
    "compare_player_d",
    "compare_player_e",
]
COMPARE_PLAYER_LABELS = ["Jugador A", "Jugador B", "Jugador C", "Jugador D", "Jugador E"]
COMPARE_COUNT_KEY = "compare_player_count"
COMPARE_VISIBLE_COUNT_KEY = "compare_visible_count"
COMPARE_REQUESTED_COUNT_KEY = "compare_requested_count"

TEAM_STYLE_FEATURE_LABELS = {
    "team_season_possession": ("Posesion / circulacion", "Posesion"),
    "team_season_passing_ratio": ("Posesion / circulacion", "Precision de pase"),
    "team_season_passes_pg": ("Posesion / circulacion", "Pases por partido"),
    "team_season_op_passes_pg": ("Posesion / circulacion", "Pases en juego abierto por partido"),
    "team_season_directness": ("Verticalidad / ritmo", "Directness"),
    "team_season_pace_towards_goal": ("Verticalidad / ritmo", "Ritmo hacia porteria"),
    "team_season_gk_pass_distance": ("Verticalidad / ritmo", "Distancia media del pase del portero"),
    "team_season_gk_long_pass_ratio": ("Verticalidad / ritmo", "Porcentaje de pase largo del portero"),
    "team_season_deep_progressions_pg": ("Progresion", "Progresiones profundas por partido"),
    "team_season_deep_completions_pg": ("Progresion", "Recepciones/completions profundas por partido"),
    "team_season_passes_inside_box_pg": ("Progresion", "Pases dentro del area por partido"),
    "team_season_crosses_into_box_pg": ("Juego exterior", "Centros al area por partido"),
    "team_season_successful_crosses_into_box_pg": ("Juego exterior", "Centros al area completados por partido"),
    "team_season_successful_box_cross_ratio": ("Juego exterior", "Eficacia de centros al area"),
    "team_season_completed_dribbles_pg": ("Conduccion / regate", "Regates completados por partido"),
    "team_season_dribble_ratio": ("Conduccion / regate", "Eficacia en regate"),
    "team_season_total_dribbles_pg": ("Conduccion / regate", "Regates totales por partido"),
    "team_season_pressures_pg": ("Presion", "Presiones por partido"),
    "team_season_counterpressures_pg": ("Presion", "Contrapresiones por partido"),
    "team_season_pressure_regains_pg": ("Presion", "Recuperaciones tras presion por partido"),
    "team_season_counterpressure_regains_pg": ("Presion", "Recuperaciones tras contrapresion por partido"),
    "team_season_fhalf_pressures_pg": ("Presion", "Presiones en campo rival por partido"),
    "team_season_fhalf_counterpressures_pg": ("Presion", "Contrapresiones en campo rival por partido"),
    "team_season_fhalf_pressures_ratio": ("Presion", "Porcentaje de presiones en campo rival"),
    "team_season_fhalf_counterpressures_ratio": ("Presion", "Porcentaje de contrapresiones en campo rival"),
    "team_season_ppda": ("Presion", "PPDA"),
    "team_season_aggressive_actions_pg": ("Presion", "Acciones agresivas por partido"),
    "team_season_aggression": ("Presion", "Agresividad"),
    "team_season_defensive_distance": ("Altura defensiva", "Distancia defensiva"),
    "team_season_defensive_distance_ppda": ("Altura defensiva", "Distancia defensiva ajustada por PPDA"),
    "team_season_counter_attacking_shots_pg": ("Transiciones", "Tiros en contraataque por partido"),
    "team_season_high_press_shots_pg": ("Transiciones", "Tiros tras presion alta por partido"),
    "team_season_np_shot_distance": ("Perfil ofensivo", "Distancia media de tiro"),
    "team_season_shots_in_clear_pg": ("Perfil ofensivo", "Tiros claros por partido"),
    "team_season_ball_in_play_time": ("Contexto del partido", "Tiempo de balon en juego"),
}


TEAM_BADGES = {
    "Athletic Club": BADGES_DIR / "athletic_club.png",
    "Atlético Madrid": BADGES_DIR / "atletico_madrid.png",
    "Club Atlético de Madrid": BADGES_DIR / "atletico_madrid.png",
    "Atletico Madrid": BADGES_DIR / "atletico_madrid.png",
    "Deportivo Alavés": BADGES_DIR / "deportivo_alaves.png",
    "Deportivo Alaves": BADGES_DIR / "deportivo_alaves.png",
    "Alavés": BADGES_DIR / "deportivo_alaves.png",
    "Alaves": BADGES_DIR / "deportivo_alaves.png",
    "FC Barcelona": BADGES_DIR / "fc_barcelona.png",
    "Barcelona": BADGES_DIR / "fc_barcelona.png",
    "F.C. Barcelona": BADGES_DIR / "fc_barcelona.png",
    "RC Celta": BADGES_DIR / "celta_vigo.png",
    "Celta Vigo": BADGES_DIR / "celta_vigo.png",
    "RC Celta de Vigo": BADGES_DIR / "celta_vigo.png",
    "Celta de Vigo": BADGES_DIR / "celta_vigo.png",
    "Espanyol": BADGES_DIR / "espanyol.png",
    "RCD Espanyol": BADGES_DIR / "espanyol.png",
    "RCD Espanyol de Barcelona": BADGES_DIR / "espanyol.png",
    "Getafe": BADGES_DIR / "getafe.png",
    "Getafe CF": BADGES_DIR / "getafe.png",
    "Getafe C.F.": BADGES_DIR / "getafe.png",
    "Girona": BADGES_DIR / "girona.png",
    "Girona FC": BADGES_DIR / "girona.png",
    "Las Palmas": BADGES_DIR / "las_palmas.png",
    "UD Las Palmas": BADGES_DIR / "las_palmas.png",
    "U.D. Las Palmas": BADGES_DIR / "las_palmas.png",
    "Leganés": BADGES_DIR / "leganes.png",
    "Leganes": BADGES_DIR / "leganes.png",
    "CD Leganés": BADGES_DIR / "leganes.png",
    "C.D. Leganés": BADGES_DIR / "leganes.png",
    "Mallorca": BADGES_DIR / "mallorca.png",
    "RCD Mallorca": BADGES_DIR / "mallorca.png",
    "Osasuna": BADGES_DIR / "osasuna.png",
    "CA Osasuna": BADGES_DIR / "osasuna.png",
    "C.A. Osasuna": BADGES_DIR / "osasuna.png",
    "Rayo Vallecano": BADGES_DIR / "rayo_vallecano.png",
    "Rayo Vallecano de Madrid": BADGES_DIR / "rayo_vallecano.png",
    "Real Betis": BADGES_DIR / "betis.png",
    "Betis": BADGES_DIR / "betis.png",
    "Real Betis Balompié": BADGES_DIR / "betis.png",
    "Real Betis Balompie": BADGES_DIR / "betis.png",
    "Real Madrid": BADGES_DIR / "real_madrid.png",
    "Real Madrid CF": BADGES_DIR / "real_madrid.png",
    "Real Madrid C.F.": BADGES_DIR / "real_madrid.png",
    "Real Sociedad": BADGES_DIR / "real_sociedad.png",
    "Real Sociedad de Fútbol": BADGES_DIR / "real_sociedad.png",
    "Real Sociedad de Futbol": BADGES_DIR / "real_sociedad.png",
    "Real Valladolid": BADGES_DIR / "real_valladolid.png",
    "Real Valladolid CF": BADGES_DIR / "real_valladolid.png",
    "Sevilla": BADGES_DIR / "sevilla.png",
    "Sevilla FC": BADGES_DIR / "sevilla.png",
    "Sevilla F.C.": BADGES_DIR / "sevilla.png",
    "Valencia": BADGES_DIR / "valencia.png",
    "Valencia CF": BADGES_DIR / "valencia.png",
    "Valencia C.F.": BADGES_DIR / "valencia.png",
    "Villarreal": BADGES_DIR / "villarreal.png",
    "Villarreal CF": BADGES_DIR / "villarreal.png",
    "Villarreal C.F.": BADGES_DIR / "villarreal.png",
}

IMPACT_COLS = [
    "impacto_finalizacion",
    "impacto_amenaza_area",
    "impacto_creacion",
    "impacto_progresion_pase",
    "impacto_progresion_conduccion",
    "impacto_circulacion",
    "impacto_duelos",
    "impacto_anticipacion",
    "impacto_defensa_area",
    "impacto_presion",
    "impacto_shot_stopping",
    "impacto_porteria_juego_aereo",
    "impacto_porteria_distribucion",
    "impacto_global",
    "impacto_ofensivo",
    "impacto_asociativo",
    "impacto_defensivo",
    "impacto_porteria",
]

FIELD_IMPACT_COLS = [
    "impacto_global",
    "impacto_ofensivo",
    "impacto_asociativo",
    "impacto_defensivo",
    "impacto_finalizacion",
    "impacto_amenaza_area",
    "impacto_creacion",
    "impacto_progresion_pase",
    "impacto_progresion_conduccion",
    "impacto_circulacion",
    "impacto_duelos",
    "impacto_anticipacion",
    "impacto_defensa_area",
    "impacto_presion",
]

GOALKEEPER_IMPACT_COLS = [
    "impacto_global",
    "impacto_porteria",
    "impacto_shot_stopping",
    "impacto_porteria_juego_aereo",
    "impacto_porteria_distribucion",
]


def impact_columns_for_role(role: str, detailed: bool = True) -> list[str]:
    if role == "POR":
        return GOALKEEPER_IMPACT_COLS if detailed else ["impacto_porteria"]
    return FIELD_IMPACT_COLS if detailed else ["impacto_ofensivo", "impacto_asociativo", "impacto_defensivo"]


def cluster_analysis_columns(role: str) -> list[str]:
    if role == "POR":
        return [
            "impacto_shot_stopping",
            "impacto_porteria_juego_aereo",
            "impacto_porteria_distribucion",
        ]
    return [
        "impacto_finalizacion",
        "impacto_amenaza_area",
        "impacto_creacion",
        "impacto_progresion_pase",
        "impacto_progresion_conduccion",
        "impacto_circulacion",
        "impacto_duelos",
        "impacto_anticipacion",
        "impacto_defensa_area",
        "impacto_presion",
    ]


def help_box(title: str, body: str, expanded: bool = False) -> None:
    with st.expander(f"? {title}", expanded=expanded):
        st.markdown(body)


@st.cache_data(show_spinner=False)
def image_as_base64(path: str, signature: tuple[float, int] | None = None) -> str:
    image_path = Path(path)
    if not image_path.exists():
        return ""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def file_signature(path: Path) -> tuple[float, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return (stat.st_mtime, stat.st_size)


def laliga_logo_html(css_class: str = "laliga-logo", alt: str = "LALIGA") -> str:
    encoded = image_as_base64(str(LALIGA_LOGO_PATH), file_signature(LALIGA_LOGO_PATH))
    if not encoded:
        return ""
    return f'<img class="{css_class}" src="data:image/png;base64,{encoded}" alt="{alt}" />'


def team_badge_html(team: str, css_class: str = "team-badge") -> str:
    path = TEAM_BADGES.get(team)
    if path is None:
        return ""
    encoded = image_as_base64(str(path), file_signature(path))
    if not encoded:
        return ""
    return f'<img class="{css_class}" src="data:image/png;base64,{encoded}" alt="{team}" />'


def team_badge_uri(team: str) -> str:
    path = TEAM_BADGES.get(team)
    if path is None:
        return ""
    encoded = image_as_base64(str(path), file_signature(path))
    if not encoded:
        return ""
    return f"data:image/png;base64,{encoded}"


def slugify_asset_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return text


def first_existing_asset(directory: Path, stem: str) -> Path | None:
    if not stem:
        return None
    for extension in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def image_tag_from_path(path: Path | None, css_class: str, alt: str = "") -> str:
    if path is None:
        return ""
    encoded = image_as_base64(str(path), file_signature(path))
    if not encoded:
        return ""
    suffix = path.suffix.lower().replace(".", "")
    mime = "jpeg" if suffix == "jpg" else suffix
    safe_alt = html.escape(str(alt), quote=True)
    return f'<img class="{css_class}" src="data:image/{mime};base64,{encoded}" alt="{safe_alt}" />'


def clean_image_url(url: object) -> str:
    if is_blank_value(url):
        return ""
    url_text = str(url).strip()
    if url_text.startswith("//"):
        url_text = f"https:{url_text}"
    if not (url_text.startswith("http://") or url_text.startswith("https://") or url_text.startswith("data:image/")):
        return ""
    return url_text


def image_tag_from_url(url: object, css_class: str, alt: str = "") -> str:
    url_text = clean_image_url(url)
    if not url_text:
        return ""
    safe_url = html.escape(url_text, quote=True)
    safe_alt = html.escape(str(alt), quote=True)
    return f'<img class="{css_class}" src="{safe_url}" alt="{safe_alt}" />'


def player_photo_source(player_name: str, photo_url: object = None) -> str | None:
    path = first_existing_asset(PLAYER_IMAGES_DIR, slugify_asset_name(player_name))
    if path is not None:
        return str(path)
    url_text = clean_image_url(photo_url)
    return url_text or None


def player_photo_html(player_name: str, photo_url: object = None) -> str:
    path = first_existing_asset(PLAYER_IMAGES_DIR, slugify_asset_name(player_name))
    image = image_tag_from_path(path, "player-photo", player_name)
    if not image:
        image = image_tag_from_url(photo_url, "player-photo", player_name)
    if image:
        return image
    initials = "".join(part[0] for part in str(player_name).split()[:2]).upper() or "?"
    return f'<div class="player-photo-placeholder"><span>{html.escape(initials)}</span><small>Foto pendiente</small></div>'


def list_player_photo_html(player_name: str, photo_url: object = None, css_class: str = "list-player-photo") -> str:
    path = first_existing_asset(PLAYER_IMAGES_DIR, slugify_asset_name(player_name))
    image = image_tag_from_path(path, css_class, player_name)
    if not image:
        image = image_tag_from_url(photo_url, css_class, player_name)
    if image:
        return image
    initials = "".join(part[0] for part in str(player_name).split()[:2]).upper() or "?"
    return f'<div class="{css_class} list-photo-placeholder"><span>{html.escape(initials)}</span></div>'


def player_row_lookup_by_name(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df.empty or "Name" not in df.columns:
        return {}
    lookup_df = df.sort_values("Minutes", ascending=False) if "Minutes" in df.columns else df.copy()
    lookup_df = lookup_df.drop_duplicates("Name")
    return {normalize_name(str(row["Name"])): row for _, row in lookup_df.iterrows()}


def list_player_photo_from_lookup(player_name: str, lookup: dict[str, pd.Series], photo_url: object = None, css_class: str = "list-player-photo") -> str:
    player_key = normalize_name(player_name)
    row = lookup.get(player_key)
    if row is None and player_key:
        player_tokens = player_key.split()
        for candidate_key, candidate_row in lookup.items():
            candidate_tokens = candidate_key.split()
            same_main_name = len(player_tokens) >= 2 and len(candidate_tokens) >= 2 and player_tokens[:2] == candidate_tokens[:2]
            near_full_name = candidate_key.startswith(player_key) or player_key.startswith(candidate_key)
            if same_main_name or near_full_name:
                row = candidate_row
                break
    if row is not None:
        photo_url = row_value(row, ["transfermarkt_photo_url", "Imagen"])
        player_name = str(row.get("Name", player_name))
    return list_player_photo_html(player_name, photo_url, css_class)


def is_blank_value(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)):
            return bool(missing)
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return not text or text.lower() in {"nan", "none", "<na>", "nat"}


def row_value(row: pd.Series, candidates: list[str]) -> object:
    for column in candidates:
        value = row.get(column) if column in row.index else None
        if not is_blank_value(value):
            return value
    return None


def add_player_to_comparison(player_name: str, slot_index: int) -> str:
    name = str(player_name)
    slot_index = max(0, min(int(slot_index), len(COMPARE_PLAYER_KEYS) - 1))
    target_key = COMPARE_PLAYER_KEYS[slot_index]
    for key in COMPARE_PLAYER_KEYS:
        if key != target_key and st.session_state.get(key) == name:
            st.session_state[key] = None
    st.session_state[target_key] = name
    st.session_state[COMPARE_REQUESTED_COUNT_KEY] = max(
        int(st.session_state.get(COMPARE_REQUESTED_COUNT_KEY, 2)),
        slot_index + 1,
    )
    return f"{name} enviado a {COMPARE_PLAYER_LABELS[slot_index]}."


def comparison_button(player_name: str, key: str) -> None:
    slot = st.selectbox(
        "Destino",
        range(len(COMPARE_PLAYER_KEYS)),
        key=f"{key}_slot",
        format_func=lambda idx: COMPARE_PLAYER_LABELS[idx],
        label_visibility="collapsed",
    )
    if st.button("Comparar", key=key, width="stretch"):
        st.success(add_player_to_comparison(player_name, int(slot)))


def nationality_label(row: pd.Series) -> str:
    value = row_value(row, ["nationality", "Nationality", "Nacionalidad", "country_name", "Country", "pais", "País"])
    if value is not None:
        return str(value)
    return "Nacionalidad pendiente"


def nationality_html(row: pd.Series) -> str:
    label = nationality_label(row)
    path = first_existing_asset(FLAGS_DIR, slugify_asset_name(label))
    flag = image_tag_from_path(path, "flag-icon", label)
    if not flag:
        flag = image_tag_from_url(row_value(row, ["nationality_flag_url", "Imagen Nacionalidad"]), "flag-icon", label)
    if not flag:
        flag = '<span class="flag-placeholder">?</span>'
    return f'{flag}<span>{html.escape(label)}</span>'


def format_player_height(value: object) -> str:
    height = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(height) or float(height) <= 0:
        return "Altura no disponible"
    height = float(height)
    height_m = height if height <= 3 else height / 100
    return f"{height_m:.2f} m"


def player_identity_card(row: pd.Series) -> None:
    name = html.escape(str(row.get("Name", "Jugador")), quote=True)
    team = str(row.get("Team", "Equipo pendiente"))
    role = ROLE_LABELS.get(row.get("role"), row.get("role", "N/D"))
    profile = row.get("cluster_label", row.get("profile_cluster", "N/D"))
    impact = row.get("impacto_global", np.nan)
    impact_text = f"{float(impact):.1f}" if pd.notna(impact) else "N/D"
    minutes = row.get("Minutes", np.nan)
    minutes_text = f"{int(minutes):,} minutos" if pd.notna(minutes) else "Minutos no disponibles"
    height_text = format_player_height(row_value(row, ["player_height", "Height", "Altura"]))

    st.markdown(
        f"""
        <section class="player-identity-card">
            <div class="player-photo-frame">
                {player_photo_html(str(row.get("Name", "")), row_value(row, ["transfermarkt_photo_url", "Imagen"]))}
            </div>
            <div class="player-identity-main">
                <div class="player-eyebrow">Ficha de jugador</div>
                <h2>{name}</h2>
                <div class="player-meta-row">
                    <div class="player-meta-pill">
                        {team_badge_html(team, "player-team-badge")}
                        <span>{html.escape(team)}</span>
                    </div>
                    <div class="player-meta-pill">
                        {nationality_html(row)}
                    </div>
                    <div class="player-meta-pill muted-pill">{minutes_text}</div>
                    <div class="player-meta-pill muted-pill">{height_text}</div>
                </div>
                <div class="player-tag-grid">
                    <div class="player-tag">
                        <span>Rol modelado</span>
                        <strong>{html.escape(str(role))}</strong>
                    </div>
                    <div class="player-tag">
                        <span>Perfil</span>
                        <strong>{html.escape(str(profile))}</strong>
                    </div>
                    <div class="player-tag highlight">
                        <span>Impacto global</span>
                        <strong>{impact_text}</strong>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_top_fit_recommendations(fit: pd.DataFrame) -> None:
    if fit.empty:
        return
    st.subheader("Top 5 recomendaciones")
    st.caption("Candidatos prioritarios para el perfil seleccionado.")

    columns = st.columns(5)
    for rank, ((_, row), column) in enumerate(zip(fit.head(5).iterrows(), columns), start=1):
        name = str(row.get("Name", "Jugador"))
        team = str(row.get("Team", "Equipo"))
        score = row.get("fit_score", np.nan)
        impact = row.get("impacto_global", np.nan)
        profile = row.get("cluster_label", "Perfil no disponible")
        score_text = f"{float(score):.1f}" if pd.notna(score) else "N/D"
        impact_text = f"{float(impact):.1f}" if pd.notna(impact) else "N/D"
        photo = player_photo_source(name, row_value(row, ["transfermarkt_photo_url", "Imagen"]))

        with column.container(border=True):
            st.markdown(f"**#{rank}**")
            if photo:
                st.image(photo, width="stretch")
            else:
                initials = "".join(part[0] for part in name.split()[:2]).upper() or "?"
                st.markdown(f"### {html.escape(initials)}")
                st.caption("Foto pendiente")
            st.markdown(f"**{name}**")
            st.caption(team)
            st.caption(str(profile))
            metric_left, metric_right = st.columns(2)
            metric_left.metric("Encaje", score_text)
            metric_right.metric("Impacto", impact_text)
            comparison_button(name, f"compare_top_{rank}_{slugify_asset_name(name)}")


def render_fit_recommendations_table(fit: pd.DataFrame, show_athletic_origin: bool = False) -> None:
    if fit.empty:
        return
    st.subheader("Tabla de recomendaciones")
    rows = []
    for _, row in fit.iterrows():
        team = str(row.get("Team", ""))
        row_data = {
            "Jugador": html.escape(str(row.get("Name", ""))),
            "Equipo": f'<div class="team-name-cell">{team_badge_html(team, "team-badge-small")}<span>{html.escape(team)}</span></div>',
            "Valor de mercado": f"EUR {float(row.get('market_value_million_eur')):.1f}M" if pd.notna(row.get("market_value_million_eur")) else "",
            "Rol": html.escape(str(ROLE_LABELS.get(row.get("role"), row.get("role", "")))),
            "Perfil futbolístico": html.escape(str(row.get("cluster_label", ""))),
            "Minutos": f"{float(row.get('Minutes')):.5f}" if pd.notna(row.get("Minutes")) else "",
            "Impacto global": f"{float(row.get('impacto_global')):.2f}" if pd.notna(row.get("impacto_global")) else "",
            "Impacto ofensivo": f"{float(row.get('impacto_ofensivo')):.2f}" if pd.notna(row.get("impacto_ofensivo")) else "",
            "Impacto asociativo": f"{float(row.get('impacto_asociativo')):.2f}" if pd.notna(row.get("impacto_asociativo")) else "",
            "Impacto defensivo": f"{float(row.get('impacto_defensivo')):.2f}" if pd.notna(row.get("impacto_defensivo")) else "",
            "Encaje": f"{float(row.get('fit_score')):.2f}" if pd.notna(row.get("fit_score")) else "",
        }
        if show_athletic_origin:
            row_data["Origen Athletic"] = html.escape(str(row.get("athletic_birth_place", "")))
        rows.append(row_data)

    shown = pd.DataFrame(rows)
    st.markdown(shown.to_html(escape=False, index=False, classes="laliga-rank-table"), unsafe_allow_html=True)

    with st.expander("Enviar recomendado a Comparaciones", expanded=False):
        action_cols = st.columns([2.4, 1.2, 0.9], vertical_alignment="bottom")
        selected_name = action_cols[0].selectbox(
            "Jugador recomendado",
            fit["Name"].astype(str).tolist(),
            key="fit_compare_player_select",
        )
        selected_slot = action_cols[1].selectbox(
            "Destino",
            range(len(COMPARE_PLAYER_KEYS)),
            key="fit_compare_slot_select",
            format_func=lambda idx: COMPARE_PLAYER_LABELS[idx],
        )
        if action_cols[2].button("Comparar", key="fit_compare_send_button", width="stretch"):
            st.success(add_player_to_comparison(selected_name, int(selected_slot)))


def minmax_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(50.0, index=values.index)
    min_value = numeric.min()
    max_value = numeric.max()
    if pd.isna(min_value) or pd.isna(max_value) or np.isclose(min_value, max_value):
        return pd.Series(50.0, index=values.index)
    score = (numeric - min_value) / (max_value - min_value) * 100
    if not higher_is_better:
        score = 100 - score
    return score.fillna(50).clip(0, 100)


def render_market_opportunities_table(data: pd.DataFrame, role: str, selected_profile: str) -> None:
    required_cols = {"market_value_million_eur", "impacto_global", "Minutes", "Name", "Team"}
    if data.empty or not required_cols.issubset(data.columns):
        return

    opportunities = data.copy()
    opportunities["market_value_million_eur"] = pd.to_numeric(opportunities["market_value_million_eur"], errors="coerce")
    opportunities["impacto_global"] = pd.to_numeric(opportunities["impacto_global"], errors="coerce")
    opportunities["Minutes"] = pd.to_numeric(opportunities["Minutes"], errors="coerce")
    if "age" in opportunities.columns:
        opportunities["age"] = pd.to_numeric(opportunities["age"], errors="coerce")
    else:
        opportunities["age"] = np.nan
    opportunities = opportunities.dropna(subset=["market_value_million_eur", "impacto_global", "Minutes"])
    opportunities = opportunities.loc[opportunities["market_value_million_eur"].gt(0)].copy()
    if opportunities.empty:
        return

    opportunities["impact_per_market_million"] = opportunities["impacto_global"] / opportunities["market_value_million_eur"].replace(0, np.nan)
    opportunities["impact_score"] = minmax_score(opportunities["impacto_global"], higher_is_better=True)
    opportunities["value_score"] = minmax_score(opportunities["impact_per_market_million"], higher_is_better=True)
    opportunities["minutes_score"] = minmax_score(opportunities["Minutes"], higher_is_better=True)
    if opportunities["age"].notna().any():
        opportunities["age_project_score"] = np.where(
            opportunities["age"].isna(),
            50,
            np.where(opportunities["age"].le(24), 100, np.where(opportunities["age"].ge(32), 35, 100 - ((opportunities["age"] - 24) / 8 * 65))),
        )
    else:
        opportunities["age_project_score"] = 50.0
    opportunities["market_opportunity_score"] = (
        0.40 * opportunities["impact_score"]
        + 0.35 * opportunities["value_score"]
        + 0.15 * opportunities["age_project_score"]
        + 0.10 * opportunities["minutes_score"]
    ).round(2)

    st.subheader("Oportunidades de mercado")
    st.caption(
        "Ranking orientativo para detectar jugadores con buen rendimiento, valor razonable, edad de proyecto y muestra suficiente de minutos."
    )
    shown = opportunities.sort_values("market_opportunity_score", ascending=False).head(15).copy()
    rows = []
    for _, row in shown.iterrows():
        team = str(row.get("Team", ""))
        rows.append(
            {
                "Jugador": html.escape(str(row.get("Name", ""))),
                "Equipo": f'<div class="team-name-cell">{team_badge_html(team, "team-badge-small")}<span>{html.escape(team)}</span></div>',
                "Perfil futbolístico": html.escape(str(row.get("cluster_label", ""))),
                "Edad": "" if pd.isna(row.get("age")) else f"{float(row.get('age')):.0f}",
                "Valor de mercado": f"EUR {float(row.get('market_value_million_eur')):.1f}M",
                "Impacto global": f"{float(row.get('impacto_global')):.2f}",
                "Impacto / €M": f"{float(row.get('impact_per_market_million')):.2f}",
                "Minutos": f"{float(row.get('Minutes')):.0f}",
                "Oportunidad": f"{float(row.get('market_opportunity_score')):.2f}",
            }
        )
    st.markdown(pd.DataFrame(rows).to_html(escape=False, index=False, classes="laliga-rank-table"), unsafe_allow_html=True)


def fit_component_weights(fit: pd.DataFrame) -> dict[str, float]:
    has_economic = "economic_fit" in fit.columns and fit["economic_fit"].notna().any()
    has_age = "age_fit" in fit.columns and fit["age_fit"].notna().any()
    if has_economic and has_age:
        return {
            "impacto_global": 0.35,
            "context_fit": 0.18,
            "team_need_fit": 0.14,
            "role_fit": 0.13,
            "economic_fit": 0.12,
            "age_fit": 0.08,
        }
    if has_economic:
        return {
            "impacto_global": 0.40,
            "context_fit": 0.20,
            "team_need_fit": 0.15,
            "role_fit": 0.15,
            "economic_fit": 0.10,
        }
    if has_age:
        return {
            "impacto_global": 0.43,
            "context_fit": 0.22,
            "team_need_fit": 0.18,
            "role_fit": 0.10,
            "age_fit": 0.07,
        }
    return {
        "impacto_global": 0.50,
        "context_fit": 0.25,
        "team_need_fit": 0.15,
        "role_fit": 0.10,
    }


def render_fit_breakdown_radar(fit: pd.DataFrame) -> None:
    if fit.empty:
        return
    weights = {col: weight for col, weight in fit_component_weights(fit).items() if col in fit.columns}
    if not weights:
        return

    st.subheader("Desglose de la puntuación de encaje")
    st.caption(
        "Radar de los 5 mejores candidatos según los componentes del encaje. "
        "Cada eje representa los puntos que aporta ese componente a la puntuación final."
    )
    shown = fit.sort_values("fit_score", ascending=False).head(5).copy()
    labels = [display_label(component) for component in weights]
    max_axis = max((weight * 100 for weight in weights.values()), default=100)
    colors = ["#111111", "#c2410c", "#2563eb", "#16a34a", "#ca8a04"]
    fig = go.Figure()
    for idx, (_, row) in enumerate(shown.iterrows()):
        values = [
            round(float(row.get(component)) * weight, 2) if pd.notna(row.get(component)) else 0.0
            for component, weight in weights.items()
        ]
        hover_text = [
            f"{label}<br>Aportación: {value:.2f}<br>Valor original: {float(row.get(component)):.2f}<br>Peso: {weight:.0%}"
            if pd.notna(row.get(component))
            else f"{label}<br>Sin dato"
            for label, value, (component, weight) in zip(labels, values, weights.items())
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                name=str(row.get("Name", f"Candidato {idx + 1}")),
                mode="lines+markers",
                fill="toself",
                line=dict(color=colors[idx % len(colors)], width=2.5),
                marker=dict(size=6),
                opacity=0.72,
                hoverinfo="text+name",
                text=hover_text + [hover_text[0]],
            )
        )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max_axis], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        title="Contribución de cada componente al encaje final | Top 5",
        legend_title="Jugador",
        margin=dict(l=40, r=40, t=70, b=40),
        height=620,
    )
    st.plotly_chart(fig, width="stretch")


def display_label(column: str) -> str:
    return DISPLAY_NAMES.get(column, str(column).replace("_", " ").title())


@st.cache_data(show_spinner=False)
def load_metric_descriptions() -> dict[str, dict[str, str]]:
    if METRIC_DESCRIPTIONS_ES_PATH.exists():
        return json.loads(METRIC_DESCRIPTIONS_ES_PATH.read_text(encoding="utf-8"))
    if not METRIC_DESCRIPTIONS_PATH.exists():
        return {}
    return json.loads(METRIC_DESCRIPTIONS_PATH.read_text(encoding="utf-8"))


def metric_description_text(metric: str) -> str:
    info = load_metric_descriptions().get(metric, {})
    description = str(info.get("description", "")).strip()
    if description:
        return description
    return "Descripción no disponible."


def team_style_feature_table(columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        block, label = TEAM_STYLE_FEATURE_LABELS.get(
            column,
            ("Otras variables", column.replace("team_season_", "").replace("_", " ").title()),
        )
        rows.append({"Bloque tactico": block, "Variable": label, "Columna tecnica": column})
    return pd.DataFrame(rows)


def display_dataframe(data: pd.DataFrame, **kwargs) -> None:
    shown = data.copy()
    if "role" in shown.columns:
        shown["role"] = shown["role"].map(lambda value: ROLE_LABELS.get(value, value))
    shown = shown.rename(columns={col: display_label(col) for col in shown.columns})
    if "Equipo" in shown.columns:
        shown["Equipo"] = shown["Equipo"].apply(
            lambda team: f'<div class="team-name-cell">{team_badge_html(str(team), "team-badge-small")}<span>{team}</span></div>'
        )
        st.markdown(shown.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.dataframe(shown, **kwargs)


def display_series(data: pd.Series, value_name: str = "Valor") -> None:
    shown = data.copy()
    if "role" in shown.index:
        shown.loc["role"] = ROLE_LABELS.get(shown.loc["role"], shown.loc["role"])
    shown.index = [display_label(idx) for idx in shown.index]
    st.dataframe(shown.to_frame(value_name), width="stretch")


def render_laliga_comparison_table(rows: list[dict[str, object]], player_names: list[str]) -> None:
    if not rows:
        return
    headers = ["Metrica", *player_names, "Rango"]
    html_rows = []
    for row in rows:
        numeric_values = {
            player: float(row[player])
            for player in player_names
            if player in row and pd.notna(row[player])
        }
        max_value = max(numeric_values.values()) if numeric_values else None
        cells = []
        for header in headers:
            value = row.get(header, "")
            text = "" if pd.isna(value) else html.escape(str(value))
            cls = ""
            if header in numeric_values and max_value is not None and np.isclose(numeric_values[header], max_value):
                cls = ' class="best-impact-cell"'
            cells.append(f"<td{cls}>{text}</td>")
        html_rows.append("<tr>" + "".join(cells) + "</tr>")
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    table_html = (
        '<table class="laliga-rank-table comparison-impact-table">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(html_rows)}</tbody>"
        "</table>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_laliga_simple_table(data: pd.DataFrame) -> None:
    if data.empty:
        return
    shown = data.copy().reset_index(drop=True)
    shown = shown.rename(columns={col: display_label(col) for col in shown.columns})
    for col in shown.columns:
        if pd.api.types.is_numeric_dtype(shown[col]):
            shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.2f}")
    st.markdown(shown.to_html(escape=False, index=False, classes="laliga-rank-table"), unsafe_allow_html=True)


def render_laliga_compact_table(data: pd.DataFrame) -> None:
    if data.empty:
        return
    shown = data.copy().reset_index(drop=True)
    shown = shown.rename(columns={col: display_label(col) for col in shown.columns})
    for col in shown.columns:
        if pd.api.types.is_numeric_dtype(shown[col]):
            shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.1f}")
    table_html = shown.to_html(escape=False, index=False, classes="laliga-rank-table laliga-compact-table")
    st.markdown(f'<div class="compact-table-wrap">{table_html}</div>', unsafe_allow_html=True)


def secret_or_env(name: str, default: str | None = None) -> str | None:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


def nested_secret(section: str, name: str, default: str | None = None) -> str | None:
    try:
        if section in st.secrets and name in st.secrets[section]:
            return str(st.secrets[section][name])
    except Exception:
        pass
    return default


def smtp_config() -> dict[str, object]:
    email_username = nested_secret("email", "email_username")
    email_password = nested_secret("email", "email_password")
    email_from = nested_secret("email", "email_from", email_username)

    host = (
        secret_or_env("SMTP_HOST")
        or nested_secret("email", "smtp_host")
        or ("smtp.gmail.com" if email_username and email_password else None)
    )
    username = secret_or_env("SMTP_USERNAME") or email_username
    password = secret_or_env("SMTP_PASSWORD") or email_password
    sender = secret_or_env("SMTP_FROM") or email_from or username
    port = int(secret_or_env("SMTP_PORT") or nested_secret("email", "smtp_port", "587") or "587")
    use_tls_raw = secret_or_env("SMTP_USE_TLS") or nested_secret("email", "smtp_use_tls", "true")
    use_tls = str(use_tls_raw).lower() not in {"0", "false", "no"}
    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "sender": sender,
        "use_tls": use_tls,
    }


def smtp_ready() -> bool:
    config = smtp_config()
    return bool(config["host"] and config["username"] and config["password"] and config["sender"])


def send_pdf_email(recipient: str, subject: str, body: str, filename: str, pdf_bytes: bytes) -> None:
    config = smtp_config()
    host = str(config["host"] or "")
    username = str(config["username"] or "")
    password = str(config["password"] or "")
    sender = str(config["sender"] or "")
    port = int(config["port"] or 587)
    use_tls = bool(config["use_tls"])
    if not host or not username or not password or not sender:
        raise RuntimeError("Faltan credenciales SMTP para enviar correos.")

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)
    msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)

    with smtplib.SMTP(host, port, timeout=30) as server:
        if use_tls:
            server.starttls()
        server.login(username, password)
        server.send_message(msg)


def add_pdf_text_page(pdf: PdfPages, title: str, sections: list[tuple[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.05, 0.96, title, fontsize=20, weight="bold", va="top")
    y = 0.90
    for heading, body in sections:
        ax.text(0.05, y, heading, fontsize=13, weight="bold", va="top")
        y -= 0.035
        for line in textwrap.wrap(str(body), width=95):
            ax.text(0.05, y, line, fontsize=9.5, va="top")
            y -= 0.022
            if y < 0.06:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis("off")
                y = 0.94
        y -= 0.02
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def image_array_for_pdf(source: str | Path | None):
    if not source:
        return None
    try:
        source_text = str(source)
        if source_text.startswith("data:image/"):
            encoded = source_text.split(",", 1)[1]
            return plt.imread(io.BytesIO(base64.b64decode(encoded)))
        if source_text.startswith("http://") or source_text.startswith("https://"):
            with urlopen(source_text, timeout=8) as response:
                return plt.imread(io.BytesIO(response.read()))
        path = Path(source_text)
        if path.exists():
            return plt.imread(str(path))
    except Exception:
        return None
    return None


def add_pdf_player_cover_page(pdf: PdfPages, row: pd.Series, title: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("#fbfaf7")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.86), 1, 0.14, color="#111111", transform=ax.transAxes))
    ax.text(0.06, 0.94, title, color="white", fontsize=20, weight="bold", va="center")
    ax.text(0.06, 0.89, "Informe generado desde el dashboard de scouting", color="#f2f2f2", fontsize=10, va="center")

    photo = image_array_for_pdf(player_photo_source(str(row.get("Name", "")), row_value(row, ["transfermarkt_photo_url", "Imagen"])))
    if photo is not None:
        img_ax = fig.add_axes([0.06, 0.58, 0.28, 0.24])
        img_ax.imshow(photo)
        img_ax.axis("off")

    badge = image_array_for_pdf(TEAM_BADGES.get(str(row.get("Team", ""))))
    if badge is not None:
        badge_ax = fig.add_axes([0.78, 0.77, 0.12, 0.10])
        badge_ax.imshow(badge)
        badge_ax.axis("off")

    role = ROLE_LABELS.get(row.get("role"), row.get("role", "N/D"))
    profile = row.get("cluster_label", row.get("profile_cluster", "N/D"))
    facts = [
        ("Jugador", row.get("Name", "N/D")),
        ("Equipo", row.get("Team", "N/D")),
        ("Rol", role),
        ("Perfil", profile),
        ("Impacto global", f"{float(row.get('impacto_global')):.2f}" if pd.notna(row.get("impacto_global")) else "N/D"),
        ("Minutos", f"{float(row.get('Minutes')):.0f}" if pd.notna(row.get("Minutes")) else "N/D"),
        ("Edad", f"{float(row.get('age')):.0f}" if pd.notna(row.get("age", np.nan)) else "N/D"),
        ("Valor mercado", f"EUR {float(row.get('market_value_million_eur')):.1f}M" if pd.notna(row.get("market_value_million_eur", np.nan)) else "N/D"),
    ]
    y = 0.78
    for label, value in facts:
        ax.text(0.40, y, label.upper(), fontsize=8, color="#777777", weight="bold")
        ax.text(0.40, y - 0.025, str(value), fontsize=15, color="#111111", weight="bold")
        y -= 0.085
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_pdf_radar_page(pdf: PdfPages, title: str, series: list[tuple[str, list[str], list[float]]]) -> None:
    if not series:
        return
    labels = series[0][1]
    if not labels:
        return
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8.27, 8.27), subplot_kw=dict(polar=True))
    colors = ["#2f6f73", "#c2410c", "#2563eb", "#7c3aed", "#ca8a04"]
    for idx, (name, _, values) in enumerate(series):
        closed_values = values + values[:1]
        ax.plot(angles, closed_values, color=colors[idx % len(colors)], linewidth=2, label=name)
        ax.fill(angles, closed_values, color=colors[idx % len(colors)], alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=16, weight="bold", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_pdf_table_page(pdf: PdfPages, title: str, table: pd.DataFrame, max_rows: int = 28) -> None:
    if table.empty:
        return
    shown = table.head(max_rows).copy()
    shown = shown.fillna("")
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.text(0.02, 0.98, title, fontsize=16, weight="bold", va="top")
    table_artist = ax.table(
        cellText=shown.astype(str).values,
        colLabels=shown.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(7.5)
    table_artist.scale(1, 1.35)
    for (row_idx, _), cell in table_artist.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#111111")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#ffffff" if row_idx % 2 else "#f7f7f7")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_pdf_bar_page(pdf: PdfPages, title: str, data: pd.DataFrame, x_col: str, y_col: str, hue_col: str | None = None) -> None:
    if data.empty or x_col not in data.columns or y_col not in data.columns:
        return
    plot_data = data.dropna(subset=[y_col]).copy()
    if plot_data.empty:
        return
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.set_title(title, fontsize=16, fontweight="bold", pad=18)
    if hue_col and hue_col in plot_data.columns:
        labels = plot_data[x_col].drop_duplicates().tolist()
        players = plot_data[hue_col].drop_duplicates().tolist()
        x = np.arange(len(labels))
        width = min(0.8 / max(len(players), 1), 0.28)
        for offset, player in enumerate(players):
            values = (
                plot_data[plot_data[hue_col].eq(player)]
                .set_index(x_col)
                .reindex(labels)[y_col]
                .fillna(0)
                .to_numpy()
            )
            ax.bar(x + (offset - (len(players) - 1) / 2) * width, values, width=width, label=str(player))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=8)
    else:
        plot_data = plot_data.sort_values(y_col)
        ax.barh(plot_data[x_col].astype(str), plot_data[y_col].astype(float), color="#111111")
        ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, 100) if hue_col else ax.set_xlim(0, 100)
    ax.grid(axis="y" if hue_col else "x", alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_pdf_subimpact_detail_pages(pdf: PdfPages, df: pd.DataFrame, row: pd.Series, radar_cols: list[str], player_label: str | None = None) -> None:
    subgroups = [
        subgroup_from_impact_column(col)
        for col in radar_cols
        if subgroup_from_impact_column(col) in METRIC_SUBGROUPS
    ]
    for subgroup in dict.fromkeys(subgroups):
        impact_col = f"impacto_{subgroup}"
        impact_value = row.get(impact_col, np.nan)
        title_suffix = f" | {player_label}" if player_label else ""
        table = subimpact_variable_table(df, row, subgroup)
        if table.empty:
            continue
        compact = table.sort_values("Percentil ajustado", ascending=False).copy()
        compact = compact[["Variable", "Valor del jugador", "Percentil ajustado"]].rename(
            columns={"Valor del jugador": "Valor", "Percentil ajustado": "Percentil"}
        )
        add_pdf_text_page(
            pdf,
            f"Subimpacto: {SUBIMPACT_LABELS.get(subgroup, display_label(impact_col))}{title_suffix}",
            [
                (
                    "Valor del subimpacto",
                    f"{float(impact_value):.2f}" if pd.notna(impact_value) else "N/D",
                )
            ],
        )
        add_pdf_table_page(pdf, "Variables y percentiles ajustados", compact, max_rows=24)

        descriptions = []
        for _, detail in table.iterrows():
            descriptions.append((str(detail["Variable"]), str(detail["Explicacion"])))
        add_pdf_text_page(pdf, f"Descripcion de variables | {SUBIMPACT_LABELS.get(subgroup, subgroup)}", descriptions)


def player_report_pdf(df: pd.DataFrame, row: pd.Series) -> bytes:
    buffer = io.BytesIO()
    name = str(row.get("Name", "Jugador"))
    role = ROLE_LABELS.get(row.get("role"), row.get("role", "N/D"))
    profile = str(row.get("cluster_label", row.get("profile_cluster", "N/D")))
    with PdfPages(buffer) as pdf:
        add_pdf_player_cover_page(pdf, row, f"Informe de jugador | {name}")
        summary = [
            ("Jugador", name),
            ("Equipo", str(row.get("Team", "N/D"))),
            ("Rol y perfil", f"{role} | {profile}"),
            ("Impacto global", f"{float(row.get('impacto_global')):.2f}" if pd.notna(row.get("impacto_global")) else "N/D"),
            ("Minutos", f"{float(row.get('Minutes')):.0f}" if pd.notna(row.get("Minutes")) else "N/D"),
            ("Valor de mercado", f"EUR {float(row.get('market_value_million_eur')):.1f}M" if pd.notna(row.get("market_value_million_eur", np.nan)) else "N/D"),
        ]
        add_pdf_text_page(pdf, "Resumen ejecutivo", summary)

        impact_cols = [col for col in impact_columns_for_role(str(row.get("role")), detailed=True) if col in row.index and pd.notna(row.get(col))]
        impacts = pd.DataFrame(
            [{"Metrica": display_label(col), "Valor": round(float(row.get(col)), 2)} for col in impact_cols]
        )
        add_pdf_radar_page(
            pdf,
            "Radar de impactos",
            [(name, [display_label(col) for col in impact_cols], [float(row.get(col)) for col in impact_cols])],
        )
        add_pdf_table_page(pdf, "Impactos del jugador", impacts)
        add_pdf_bar_page(pdf, "Grafico de impactos del jugador", impacts, "Metrica", "Valor")
        add_pdf_subimpact_detail_pages(pdf, df, row, impact_cols, name)

        role_df = df[df["role"].eq(row.get("role"))].copy()
        if "impacto_global" in role_df.columns:
            ranking = role_df.sort_values("impacto_global", ascending=False).reset_index(drop=True)
            player_pos = ranking.index[ranking["Name"].astype(str).eq(name)]
            pos_text = f"{int(player_pos[0]) + 1} de {len(ranking)}" if len(player_pos) else "N/D"
            peers = ranking[existing_columns(ranking, ["Name", "Team", "cluster_label", "Minutes", "impacto_global"])].head(15)
            peers = peers.rename(columns={col: display_label(col) for col in peers.columns})
            add_pdf_text_page(pdf, "Contexto competitivo", [("Ranking dentro del rol", pos_text)])
            add_pdf_table_page(pdf, "Top del rol por impacto global", peers)

        profile_df = df[df["cluster_label"].eq(row.get("cluster_label"))].copy() if "cluster_label" in df.columns else pd.DataFrame()
        if not profile_df.empty:
            profile_top = profile_df.sort_values("impacto_global", ascending=False)
            profile_top = profile_top[existing_columns(profile_top, ["Name", "Team", "role", "Minutes", "impacto_global"])].head(15)
            profile_top = profile_top.rename(columns={col: display_label(col) for col in profile_top.columns})
            add_pdf_table_page(pdf, f"Comparativa dentro del perfil {profile}", profile_top)
    return buffer.getvalue()


def add_pdf_comparison_cover_page(pdf: PdfPages, rows_to_compare: list[pd.Series], selected_players: list[str]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#fbfaf7")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.86), 1, 0.14, color="#111111", transform=ax.transAxes))
    ax.text(0.04, 0.94, "Informe de comparacion de jugadores", color="white", fontsize=20, weight="bold", va="center")
    ax.text(0.04, 0.89, "Radar, impactos, datos generales y desglose de variables", color="#f2f2f2", fontsize=10, va="center")

    card_width = 0.88 / max(len(rows_to_compare), 1)
    for idx, row in enumerate(rows_to_compare):
        left = 0.04 + idx * card_width
        ax.add_patch(plt.Rectangle((left, 0.12), card_width - 0.02, 0.68, color="white", ec="#dddddd", transform=ax.transAxes))
        photo = image_array_for_pdf(player_photo_source(str(row.get("Name", "")), row_value(row, ["transfermarkt_photo_url", "Imagen"])))
        if photo is not None:
            img_ax = fig.add_axes([left + 0.025, 0.57, min(card_width - 0.07, 0.16), 0.19])
            img_ax.imshow(photo)
            img_ax.axis("off")
        y = 0.51
        role = ROLE_LABELS.get(row.get("role"), row.get("role", "N/D"))
        facts = [
            selected_players[idx],
            str(row.get("Team", "N/D")),
            str(role),
            str(row.get("cluster_label", "N/D")),
            f"Impacto: {float(row.get('impacto_global')):.2f}" if pd.notna(row.get("impacto_global")) else "Impacto: N/D",
            f"Min: {float(row.get('Minutes')):.0f}" if pd.notna(row.get("Minutes")) else "Min: N/D",
        ]
        for line_idx, text in enumerate(facts):
            ax.text(left + 0.025, y, textwrap.fill(text, width=22), fontsize=10 if line_idx else 12, weight="bold" if line_idx in {0, 4} else "normal", va="top")
            y -= 0.075
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def comparison_report_pdf(df: pd.DataFrame, rows_to_compare: list[pd.Series], impact_rows: list[dict[str, object]], selected_players: list[str]) -> bytes:
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        add_pdf_comparison_cover_page(pdf, rows_to_compare, selected_players)
        sections = []
        for label, row in zip(COMPARE_PLAYER_LABELS, rows_to_compare):
            role = ROLE_LABELS.get(row.get("role"), row.get("role", "N/D"))
            profile = row.get("cluster_label", row.get("profile_cluster", "N/D"))
            sections.append(
                (
                    f"{label}: {row.get('Name', 'Jugador')}",
                    f"Equipo: {row.get('Team', 'N/D')} | Rol: {role} | Perfil: {profile} | "
                    f"Minutos: {row.get('Minutes', 'N/D')} | Impacto global: {row.get('impacto_global', 'N/D')}",
                )
            )
        add_pdf_text_page(pdf, "Informe de comparacion de jugadores", sections)

        impact_table = pd.DataFrame(impact_rows)
        radar_metrics = [row.get("Metrica") for row in impact_rows if row.get("Metrica") != display_label("impacto_global")]
        radar_series = []
        for player in selected_players:
            values = [float(row.get(player, 0)) if pd.notna(row.get(player, np.nan)) else 0.0 for row in impact_rows if row.get("Metrica") != display_label("impacto_global")]
            radar_series.append((player, radar_metrics, values))
        add_pdf_radar_page(pdf, "Radar comparativo de impactos", radar_series)

        winners = []
        for row in impact_rows:
            values = {player: row.get(player) for player in selected_players if pd.notna(row.get(player))}
            winner = max(values, key=values.get) if values else ""
            winners.append(winner)
        if not impact_table.empty:
            impact_table["Mejor"] = winners
            add_pdf_table_page(pdf, "Tabla de impactos comparados", impact_table, max_rows=35)
            plot_table = impact_table.drop(columns=["Rango", "Mejor"], errors="ignore").melt(
                id_vars="Metrica",
                value_vars=selected_players,
                var_name="Jugador",
                value_name="Impacto",
            )
            add_pdf_bar_page(pdf, "Grafico comparativo de impactos", plot_table, "Metrica", "Impacto", "Jugador")

        identity = pd.DataFrame(rows_to_compare)
        identity_cols = existing_columns(
            identity,
            ["Name", "Team", "role", "cluster_label", "Minutes", "age", "market_value_million_eur", "contract_until"],
        )
        if identity_cols:
            identity_shown = identity[identity_cols].rename(columns={col: display_label(col) for col in identity_cols})
            add_pdf_table_page(pdf, "Datos generales", identity_shown, max_rows=10)

        for row in rows_to_compare:
            metric_cols = comparison_metric_columns_for_rows([row])
            add_pdf_subimpact_detail_pages(pdf, df, row, metric_cols, str(row.get("Name", "Jugador")))
    return buffer.getvalue()


def browser_executable_for_pdf() -> str | None:
    candidates = [
        os.getenv("PDF_BROWSER_PATH"),
        shutil.which("google-chrome"),
        shutil.which("chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("microsoft-edge"),
        shutil.which("msedge"),
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def html_to_pdf_bytes(document: str) -> bytes:
    browser = browser_executable_for_pdf()
    if not browser:
        raise RuntimeError("No encuentro Chrome, Edge o Chromium para convertir el informe HTML a PDF.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        html_path = tmp_path / "report.html"
        pdf_path = tmp_path / "report.pdf"
        html_path.write_text(document, encoding="utf-8")
        command = [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if completed.returncode != 0 or not pdf_path.exists():
            detail = completed.stderr.strip() or completed.stdout.strip() or "sin detalle"
            raise RuntimeError(f"No se pudo generar el PDF con el navegador: {detail}")
        return pdf_path.read_bytes()


def report_image_src(source: str | Path | None) -> str:
    if not source:
        return ""
    source_text = str(source)
    if source_text.startswith("http://") or source_text.startswith("https://") or source_text.startswith("data:image/"):
        return source_text
    path = Path(source_text)
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    suffix = path.suffix.lower().replace(".", "")
    mime = "jpeg" if suffix == "jpg" else suffix or "png"
    return f"data:image/{mime};base64,{encoded}"


def report_badge_img(team: object, css_class: str = "team-badge-small") -> str:
    src = report_image_src(TEAM_BADGES.get(str(team)))
    if not src:
        return ""
    return f'<img class="{css_class}" src="{src}" alt="{html.escape(str(team), quote=True)}" />'


def report_photo_img(row: pd.Series, css_class: str = "report-player-photo") -> str:
    name = str(row.get("Name", "Jugador"))
    src = report_image_src(player_photo_source(name, row_value(row, ["transfermarkt_photo_url", "Imagen"])))
    if src:
        return f'<img class="{css_class}" src="{src}" alt="{html.escape(name, quote=True)}" />'
    initials = "".join(part[0] for part in name.split()[:2]).upper() or "?"
    return f'<div class="{css_class} photo-placeholder">{html.escape(initials)}</div>'


def report_css() -> str:
    return """
    @page { size: A4; margin: 12mm; }
    * { box-sizing: border-box; }
    body {
        margin: 0;
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
        color: #111;
        background:
          radial-gradient(circle at 8% 10%, rgba(255,43,43,.20), transparent 22rem),
          linear-gradient(135deg, rgba(7,7,7,.045) 25%, transparent 25%) 0 0 / 28px 28px,
          linear-gradient(180deg, #f7f5ef 0%, #ece8df 100%);
    }
    .report-page { page-break-after: always; padding: 0; }
    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 18px;
        padding: 28px;
        color: #fff;
        background:
          linear-gradient(90deg, rgba(255,255,255,.08) 49.7%, rgba(255,255,255,.55) 50%, rgba(255,255,255,.08) 50.3%),
          radial-gradient(circle at 50% 50%, transparent 0 70px, rgba(255,255,255,.35) 71px 73px, transparent 74px),
          linear-gradient(135deg, #050505 0%, #171717 58%, #d71920 100%);
        box-shadow: 0 18px 42px rgba(0,0,0,.18);
    }
    .hero h1 { margin: 12px 0 8px; font-size: 38px; line-height: .96; letter-spacing: -.04em; }
    .hero .eyebrow { display:inline-block; background:#ff2b2b; padding:6px 10px; border-radius:999px; font-weight:900; font-size:11px; letter-spacing:.12em; }
    .hero p { color: rgba(255,255,255,.82); max-width: 760px; }
    .player-card {
        display: grid;
        grid-template-columns: 150px 1fr;
        gap: 18px;
        margin: 18px 0;
        padding: 16px;
        background: rgba(255,255,255,.94);
        border: 1px solid rgba(7,7,7,.10);
        border-left: 7px solid #ff2b2b;
        border-radius: 16px;
        box-shadow: 0 14px 32px rgba(0,0,0,.10);
        page-break-inside: avoid;
    }
    .report-player-photo {
        width: 150px; height: 190px; object-fit: cover; border-radius: 12px;
        background: #111; border: 1px solid rgba(0,0,0,.14);
        display: grid; place-items: center; color: white; font-size: 34px; font-weight: 900;
    }
    .player-card h2 { margin: 2px 0 8px; font-size: 30px; letter-spacing: -.03em; }
    .meta-row { display:flex; align-items:center; gap: 10px; flex-wrap: wrap; margin: 8px 0 14px; }
    .pill { display:inline-flex; align-items:center; gap:7px; padding:7px 10px; border:1px solid rgba(7,7,7,.12); border-radius:999px; background:#fff; font-weight:800; }
    .team-badge-small { width:24px; height:24px; object-fit:contain; }
    .tag-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .tag { border:1px solid rgba(7,7,7,.10); border-radius:10px; padding:10px; background:#fff; }
    .tag span { display:block; color:#626262; font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.06em; }
    .tag strong { display:block; margin-top:5px; font-size:18px; }
    .tag.highlight { border-left: 5px solid #ff2b2b; }
    h2.section-title { margin: 24px 0 10px; font-size: 22px; letter-spacing:-.02em; }
    .card {
        background: rgba(255,255,255,.92);
        border: 1px solid rgba(7,7,7,.10);
        border-radius: 12px;
        padding: 16px;
        margin: 14px 0;
        page-break-inside: avoid;
    }
    table.laliga-rank-table {
        border-collapse: collapse; width:100%; background:rgba(255,255,255,.88);
        font-size: 12px; page-break-inside:auto;
    }
    .laliga-rank-table th {
        background:#111 !important; color:#fff !important; text-transform:uppercase; letter-spacing:.035em;
        font-weight:950; font-size:10px; text-align:center; padding:9px 8px; border:0;
    }
    .laliga-rank-table td {
        color:#050505; padding:8px 8px; border-top:1px solid rgba(0,0,0,.08);
        border-right:1px solid rgba(0,0,0,.06); vertical-align:middle; line-height:1.35;
    }
    .laliga-rank-table td:first-child { font-weight:700; }
    .best-impact-cell { background:rgba(22,163,74,.18); color:#14532d; font-weight:950; box-shadow:inset 0 0 0 1px rgba(22,163,74,.35); }
    .variable-report-table { font-size: 10px; }
    .variable-report-table td:nth-child(2) { width: 36%; color:#333; font-size: 9.5px; line-height:1.32; }
    .variable-report-table td:not(:nth-child(1)):not(:nth-child(2)) { text-align:right; white-space:nowrap; }
    .report-percentile-cell { min-width: 150px; }
    .report-percentile-track {
        height: 10px;
        border-radius: 999px;
        background: linear-gradient(90deg,#b91c1c 0%,#f97316 35%,#fde047 65%,#15803d 100%);
        position: relative;
        margin-bottom: 3px;
    }
    .report-percentile-marker {
        position: absolute;
        top: -3px;
        width: 2px;
        height: 16px;
        background: #10251d;
    }
    .report-percentile-label {
        color: #47564e;
        font-size: 10px;
        text-align: left;
        font-weight: 800;
    }
    .radar-card { text-align:center; }
    .comparison-grid { display:grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
    .comparison-cover .player-card {
        grid-template-columns: 86px 1fr;
        gap: 12px;
        margin: 10px 0;
        padding: 10px;
        border-radius: 12px;
    }
    .comparison-cover .report-player-photo {
        width: 86px;
        height: 108px;
        border-radius: 9px;
        font-size: 22px;
    }
    .comparison-cover .player-card h2 {
        font-size: 20px;
        margin-bottom: 5px;
    }
    .comparison-cover .tag-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 6px;
    }
    .comparison-cover .tag {
        padding: 7px;
    }
    .comparison-cover .tag strong {
        font-size: 13px;
    }
    .comparison-cover .tag span {
        font-size: 9px;
    }
    .player-report-cover .hero {
        padding: 18px 22px;
        border-radius: 14px;
    }
    .player-report-cover .hero h1 {
        margin: 8px 0 5px;
        font-size: 30px;
    }
    .player-report-cover .hero p {
        margin: 4px 0 0;
        font-size: 12px;
    }
    .player-report-cover .player-card {
        grid-template-columns: 105px 1fr;
        gap: 12px;
        margin: 10px 0;
        padding: 10px;
        border-radius: 12px;
    }
    .player-report-cover .report-player-photo {
        width: 105px;
        height: 132px;
        border-radius: 9px;
        font-size: 26px;
    }
    .player-report-cover .player-card h2 {
        font-size: 22px;
        margin-bottom: 5px;
    }
    .player-report-cover .meta-row {
        margin: 5px 0 8px;
        gap: 6px;
    }
    .player-report-cover .pill {
        padding: 5px 8px;
        font-size: 11px;
    }
    .player-report-cover .tag-grid {
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 6px;
    }
    .player-report-cover .tag {
        padding: 7px;
    }
    .player-report-cover .tag span {
        font-size: 9px;
    }
    .player-report-cover .tag strong {
        font-size: 13px;
    }
    .player-report-cover h2.section-title {
        margin: 10px 0 5px;
        font-size: 17px;
    }
    .player-report-cover .radar-card {
        margin: 6px 0 0;
        padding: 6px;
    }
    .small { color:#626262; font-size:12px; }
    .description-list h4 { margin: 10px 0 4px; }
    .description-list p { margin: 0 0 8px; font-size: 11px; line-height: 1.4; color:#333; }
    """


def radar_svg(series: list[tuple[str, list[str], list[float]]], width: int = 720, height: int = 520) -> str:
    if not series or not series[0][1]:
        return ""
    labels = series[0][1]
    count = len(labels)
    cx, cy, radius = width / 2, height / 2 + 10, min(width, height) * 0.34
    angles = [(-np.pi / 2) + (2 * np.pi * idx / count) for idx in range(count)]
    grid = []
    for level in [20, 40, 60, 80, 100]:
        pts = []
        for angle in angles:
            r = radius * level / 100
            pts.append(f"{cx + np.cos(angle) * r:.1f},{cy + np.sin(angle) * r:.1f}")
        grid.append(f'<polygon points="{" ".join(pts)}" fill="none" stroke="rgba(0,0,0,.12)" stroke-width="1"/>')
    axes = []
    label_nodes = []
    for label, angle in zip(labels, angles):
        x = cx + np.cos(angle) * radius
        y = cy + np.sin(angle) * radius
        axes.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="rgba(0,0,0,.10)" />')
        lx = cx + np.cos(angle) * (radius + 38)
        ly = cy + np.sin(angle) * (radius + 30)
        label_nodes.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="10" font-weight="700">{html.escape(label)}</text>')
    colors = ["#2f6f73", "#c2410c", "#2563eb", "#7c3aed", "#ca8a04"]
    polygons = []
    legend = []
    for idx, (name, _, values) in enumerate(series):
        pts = []
        for angle, value in zip(angles, values):
            r = radius * max(0, min(100, float(value))) / 100
            pts.append(f"{cx + np.cos(angle) * r:.1f},{cy + np.sin(angle) * r:.1f}")
        color = colors[idx % len(colors)]
        polygons.append(f'<polygon points="{" ".join(pts)}" fill="{color}" fill-opacity=".16" stroke="{color}" stroke-width="3"/>')
        legend.append(f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px;"><b style="width:12px;height:12px;background:{color};display:inline-block;border-radius:3px;"></b>{html.escape(name)}</span>')
    return f'<div class="card radar-card"><svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(grid + axes + polygons + label_nodes)}</svg><div>{"".join(legend)}</div></div>'


def dataframe_to_report_table(df: pd.DataFrame, classes: str = "laliga-rank-table", escape: bool = True) -> str:
    if df.empty:
        return ""
    return df.to_html(index=False, escape=escape, classes=classes)


def report_percentile_bar(value: float | int | None) -> str:
    if pd.isna(value):
        return ""
    pct = max(0.0, min(100.0, float(value)))
    return (
        '<div class="report-percentile-cell">'
        '<div class="report-percentile-track">'
        f'<div class="report-percentile-marker" style="left:{pct:.1f}%;"></div>'
        '</div>'
        f'<div class="report-percentile-label">{pct:.1f}</div>'
        '</div>'
    )


def percentile_for_report_metric(df: pd.DataFrame, row: pd.Series, metric: str) -> float | None:
    if metric not in df.columns or "role" not in df.columns or "role" not in row.index:
        return None
    role_df = df[df["role"].eq(row.get("role"))].copy()
    return metric_percentile_for_player(role_df, row.name, metric)


def comparison_impact_report_table(df: pd.DataFrame, rows: list[dict[str, object]], selected_players: list[str], rows_to_compare: list[pd.Series]) -> str:
    if not rows:
        return ""
    metric_by_label = {display_label(col): col for row in rows_to_compare for col in comparison_metric_columns_for_rows([row])}
    headers = ["Metrica", *selected_players, "Percentil ganador"]
    body = []
    for row in rows:
        values = {player: row.get(player) for player in selected_players if pd.notna(row.get(player))}
        best = max(values.values()) if values else None
        winner = max(values, key=values.get) if values else None
        winner_percentile = None
        metric = metric_by_label.get(str(row.get("Metrica", "")))
        if winner and metric:
            winner_idx = selected_players.index(winner)
            winner_percentile = percentile_for_report_metric(df, rows_to_compare[winner_idx], metric)
        cells = []
        for header in headers:
            if header == "Percentil ganador":
                cells.append(f"<td>{report_percentile_bar(winner_percentile)}</td>")
                continue
            value = row.get(header, "")
            cls = ""
            if header in values and best is not None and np.isclose(float(values[header]), float(best)):
                cls = ' class="best-impact-cell"'
            cells.append(f"<td{cls}>{html.escape('' if pd.isna(value) else str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    head = "".join(f"<th>{html.escape(col)}</th>" for col in headers)
    return f'<table class="laliga-rank-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def format_report_number(value: object, decimals: int = 3) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    text = f"{float(numeric):.{decimals}f}"
    return text.rstrip("0").rstrip(".")


def comparison_variable_report_sections(df: pd.DataFrame, rows_to_compare: list[pd.Series], selected_players: list[str]) -> str:
    metric_cols = comparison_metric_columns_for_rows(rows_to_compare)
    subgroups = [
        subgroup_from_impact_column(col)
        for col in metric_cols
        if subgroup_from_impact_column(col) in METRIC_SUBGROUPS
    ]
    sections = []
    for subgroup in dict.fromkeys(subgroups):
        table_rows = []
        for metric in METRIC_SUBGROUPS.get(subgroup, []):
            values = {}
            for player, row in zip(selected_players, rows_to_compare):
                raw_value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
                if pd.notna(raw_value):
                    values[player] = float(raw_value)
            if not values:
                continue

            best_value = min(values.values()) if metric in NEGATIVE_METRICS else max(values.values())
            winner = next((player for player, value in values.items() if np.isclose(value, best_value)), "")
            winner_percentile = None
            if winner:
                winner_row = rows_to_compare[selected_players.index(winner)]
                winner_percentile = percentile_for_report_metric(df, winner_row, metric)
            cells = [
                f"<td>{html.escape(DISPLAY_NAMES.get(metric, metric))}</td>",
                f"<td>{html.escape(metric_description_text(metric))}</td>",
            ]
            for player in selected_players:
                value = values.get(player)
                cls = ""
                if value is not None and np.isclose(value, best_value):
                    cls = ' class="best-impact-cell"'
                cells.append(f"<td{cls}>{html.escape(format_report_number(value))}</td>")
            cells.append(f"<td>{report_percentile_bar(winner_percentile)}</td>")
            table_rows.append("<tr>" + "".join(cells) + "</tr>")

        if not table_rows:
            continue
        headers = ["Variable", "Explicacion", *selected_players, "Percentil ganador"]
        head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        table_html = (
            '<table class="laliga-rank-table variable-report-table">'
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(table_rows)}</tbody></table>"
        )
        sections.append(
            f'<section class="report-page"><h2 class="section-title">{html.escape(SUBIMPACT_LABELS.get(subgroup, subgroup))}</h2>{table_html}</section>'
        )
    return "".join(sections)


def player_report_html(df: pd.DataFrame, row: pd.Series) -> str:
    name = str(row.get("Name", "Jugador"))
    role = ROLE_LABELS.get(row.get("role"), row.get("role", "N/D"))
    profile = str(row.get("cluster_label", row.get("profile_cluster", "N/D")))
    impact = row.get("impacto_global", np.nan)
    minutes = row.get("Minutes", np.nan)
    team = str(row.get("Team", "Equipo"))
    impact_text = f"{float(impact):.2f}" if pd.notna(impact) else "N/D"
    minutes_text = f"{float(minutes):.0f}" if pd.notna(minutes) else "N/D"
    market_value = row.get("market_value_million_eur", np.nan)
    market_text = f"EUR {float(market_value):.1f}M" if pd.notna(market_value) else "N/D"
    height_text = format_player_height(row_value(row, ["player_height", "Height", "Altura"]))
    impact_cols = [col for col in impact_columns_for_role(str(row.get("role")), detailed=True) if col in row.index and pd.notna(row.get(col))]
    impact_rows = []
    for col in impact_cols:
        percentile = percentile_for_report_metric(df, row, col)
        impact_rows.append(
            {
                "Metrica": display_label(col),
                "Valor": round(float(row.get(col)), 2),
                "Percentil por rol": report_percentile_bar(percentile),
            }
        )
    impacts = pd.DataFrame(impact_rows)
    radar = radar_svg([(name, [display_label(col) for col in impact_cols], [float(row.get(col)) for col in impact_cols])], width=560, height=320)

    subimpact_html = []
    for subgroup in dict.fromkeys(subgroup_from_impact_column(col) for col in impact_cols if subgroup_from_impact_column(col) in METRIC_SUBGROUPS):
        table = subimpact_variable_table(df, row, subgroup)
        if table.empty:
            continue
        shown = table.sort_values("Percentil ajustado", ascending=False).copy()
        shown = shown[["Variable", "Explicacion", "Valor del jugador", "Percentil ajustado"]].rename(columns={"Valor del jugador": "Valor", "Percentil ajustado": "Percentil"})
        shown["Percentil"] = shown["Percentil"].apply(report_percentile_bar)
        subimpact_html.append(
            f'<section class="report-page"><h2 class="section-title">{html.escape(SUBIMPACT_LABELS.get(subgroup, display_label(f"impacto_{subgroup}")))}</h2>'
            f'{dataframe_to_report_table(shown, classes="laliga-rank-table variable-report-table", escape=False)}</section>'
        )

    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{report_css()}</style></head><body>
    <section class="report-page player-report-cover">
      <div class="hero"><div class="eyebrow">LALIGA STYLE ANALYTICS</div><h1>Informe de jugador</h1><p>Perfil individual, impacto, radar y desglose completo de subimpactos.</p></div>
      <div class="player-card">
        {report_photo_img(row)}
        <div>
          <h2>{html.escape(name)}</h2>
          <div class="meta-row"><span class="pill">{report_badge_img(team)}{html.escape(team)}</span><span class="pill">{html.escape(role)}</span><span class="pill">{html.escape(profile)}</span></div>
          <div class="tag-grid">
            <div class="tag highlight"><span>Impacto global</span><strong>{impact_text}</strong></div>
            <div class="tag"><span>Minutos</span><strong>{minutes_text}</strong></div>
            <div class="tag"><span>Valor mercado</span><strong>{market_text}</strong></div>
            <div class="tag"><span>Altura</span><strong>{height_text}</strong></div>
          </div>
        </div>
      </div>
      <h2 class="section-title">Radar de impactos</h2>{radar}
    </section>
    <section class="report-page">
      <h2 class="section-title">Tabla de impactos</h2>{dataframe_to_report_table(impacts, escape=False)}
    </section>
    {''.join(subimpact_html)}
    </body></html>"""


def comparison_report_html(df: pd.DataFrame, rows_to_compare: list[pd.Series], impact_rows: list[dict[str, object]], selected_players: list[str]) -> str:
    metric_labels = [row.get("Metrica") for row in impact_rows if row.get("Metrica") != display_label("impacto_global")]
    radar_series = []
    for player in selected_players:
        values = [float(row.get(player, 0)) if pd.notna(row.get(player, np.nan)) else 0.0 for row in impact_rows if row.get("Metrica") != display_label("impacto_global")]
        radar_series.append((player, metric_labels, values))
    cards = []
    for label, row in zip(selected_players, rows_to_compare):
        role = ROLE_LABELS.get(row.get("role"), row.get("role", "N/D"))
        team = str(row.get("Team", "Equipo"))
        profile = str(row.get("cluster_label", "N/D"))
        impact = row.get("impacto_global", np.nan)
        impact_text = f"{float(impact):.2f}" if pd.notna(impact) else "N/D"
        minutes = row.get("Minutes", np.nan)
        minutes_text = f"{float(minutes):.0f}" if pd.notna(minutes) else "N/D"
        cards.append(
            f'<div class="player-card">{report_photo_img(row)}<div><h2>{html.escape(label)}</h2>'
            f'<div class="meta-row"><span class="pill">{report_badge_img(team)}{html.escape(team)}</span><span class="pill">{html.escape(str(role))}</span></div>'
            f'<div class="tag-grid"><div class="tag highlight"><span>Impacto global</span><strong>{impact_text}</strong></div>'
            f'<div class="tag"><span>Perfil</span><strong>{html.escape(profile)}</strong></div>'
            f'<div class="tag"><span>Minutos</span><strong>{minutes_text}</strong></div></div></div></div>'
        )
    identity = pd.DataFrame(rows_to_compare)
    identity_cols = existing_columns(identity, ["Name", "Team", "role", "cluster_label", "Minutes", "age", "market_value_million_eur", "contract_until"])
    identity_table = dataframe_to_report_table(identity[identity_cols].rename(columns={col: display_label(col) for col in identity_cols})) if identity_cols else ""
    sub_sections = comparison_variable_report_sections(df, rows_to_compare, selected_players)
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{report_css()}</style></head><body>
    <section class="report-page comparison-cover">
      <div class="hero"><div class="eyebrow">LALIGA STYLE ANALYTICS</div><h1>Comparacion de jugadores</h1><p>Vista imprimible con el mismo lenguaje visual del dashboard.</p></div>
      {''.join(cards)}
    </section>
    <section class="report-page">
      <h2 class="section-title">Radar comparativo</h2>{radar_svg(radar_series)}
    </section>
    <section class="report-page">
      <h2 class="section-title">Tabla de impactos</h2>{comparison_impact_report_table(df, impact_rows, selected_players, rows_to_compare)}
    </section>
    <section class="report-page">
      <h2 class="section-title">Datos generales</h2>{identity_table}
    </section>
    {sub_sections}
    </body></html>"""


def player_report_pdf(df: pd.DataFrame, row: pd.Series) -> bytes:
    return html_to_pdf_bytes(player_report_html(df, row))


def comparison_report_pdf(df: pd.DataFrame, rows_to_compare: list[pd.Series], impact_rows: list[dict[str, object]], selected_players: list[str]) -> bytes:
    return html_to_pdf_bytes(comparison_report_html(df, rows_to_compare, impact_rows, selected_players))


def empty_report_lists() -> dict[str, list[dict[str, object]]]:
    return {"players": [], "comparisons": []}


def load_report_lists() -> dict[str, list[dict[str, object]]]:
    if not REPORT_LISTS_PATH.exists():
        return empty_report_lists()
    try:
        data = json.loads(REPORT_LISTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return empty_report_lists()
    return {
        "players": data.get("players", []) if isinstance(data.get("players", []), list) else [],
        "comparisons": data.get("comparisons", []) if isinstance(data.get("comparisons", []), list) else [],
    }


def save_report_lists(data: dict[str, list[dict[str, object]]]) -> None:
    REPORT_LISTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_LISTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def report_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def current_auth_user() -> str:
    return str(st.session_state.get("auth_user", "Usuario"))


def save_player_report_entry(row_dict: dict[str, object], filename: str, pdf_bytes: bytes) -> None:
    data = load_report_lists()
    report_id = uuid.uuid4().hex
    SAVED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{report_id}_{filename}"
    stored_path = SAVED_REPORTS_DIR / stored_filename
    stored_path.write_bytes(pdf_bytes)
    data["players"].insert(
        0,
        {
            "id": report_id,
            "created_at": report_timestamp(),
            "user": current_auth_user(),
            "player": str(row_dict.get("Name", "Jugador")),
            "team": str(row_dict.get("Team", "")),
            "role": str(ROLE_LABELS.get(row_dict.get("role"), row_dict.get("role", ""))),
            "profile": str(row_dict.get("cluster_label", row_dict.get("profile_cluster", ""))),
            "impacto_global": None if pd.isna(row_dict.get("impacto_global", np.nan)) else round(float(row_dict.get("impacto_global")), 2),
            "photo_url": str(row_dict.get("transfermarkt_photo_url", row_dict.get("Imagen", ""))),
            "filename": filename,
            "path": str(stored_path.relative_to(BASE_DIR)),
        },
    )
    save_report_lists(data)


def save_comparison_entry(rows_to_compare: list[pd.Series], selected_players: list[str]) -> None:
    data = load_report_lists()
    data["comparisons"].insert(
        0,
        {
            "id": uuid.uuid4().hex,
            "created_at": report_timestamp(),
            "user": current_auth_user(),
            "players": list(selected_players),
            "teams": [str(row.get("Team", "")) for row in rows_to_compare],
            "roles": [str(ROLE_LABELS.get(row.get("role"), row.get("role", ""))) for row in rows_to_compare],
            "profiles": [str(row.get("cluster_label", row.get("profile_cluster", ""))) for row in rows_to_compare],
            "photo_urls": [str(row.get("transfermarkt_photo_url", row.get("Imagen", ""))) for row in rows_to_compare],
        },
    )
    save_report_lists(data)


def update_comparison_entry_pdf(report_id: str, filename: str, pdf_bytes: bytes) -> None:
    data = load_report_lists()
    SAVED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{report_id}_{filename}"
    stored_path = SAVED_REPORTS_DIR / stored_filename
    stored_path.write_bytes(pdf_bytes)
    for entry in data.get("comparisons", []):
        if str(entry.get("id")) == str(report_id):
            entry["filename"] = filename
            entry["path"] = str(stored_path.relative_to(BASE_DIR))
            break
    save_report_lists(data)


def delete_report_entry(section: str, report_id: str) -> None:
    data = load_report_lists()
    entries = data.get(section, [])
    kept = []
    for entry in entries:
        if str(entry.get("id")) == str(report_id):
            if entry.get("path"):
                path = BASE_DIR / str(entry["path"])
                if path.exists() and path.is_file():
                    path.unlink()
            continue
        kept.append(entry)
    data[section] = kept
    save_report_lists(data)


def render_pdf_preview(pdf_bytes: bytes, height: int = 760) -> None:
    encoded = base64.b64encode(pdf_bytes).decode("utf-8")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="{height}" type="application/pdf"></iframe>',
        unsafe_allow_html=True,
    )


def render_report_delivery(
    prefix: str,
    filename: str,
    pdf_bytes: bytes,
    subject: str,
    body: str,
    on_download=None,
    on_download_args: tuple = (),
) -> None:
    st.download_button(
        "Descargar informe PDF",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        key=f"{prefix}_download_pdf",
        width="stretch",
        on_click=on_download,
        args=on_download_args,
    )
    with st.expander("Enviar informe por correo", expanded=False):
        recipient = st.text_input("Cuenta de correo destinataria", key=f"{prefix}_email")
        if not smtp_ready():
            st.info(
                "La descarga funciona, pero el envio por correo necesita una cuenta remitente configurada. "
                "Para Gmail debes crear un archivo .streamlit/secrets.toml con tu correo y una contraseña de aplicación de Google."
            )
            st.code(
                '[email]\n'
                'email_username = "tu_correo@gmail.com"\n'
                'email_password = "clave_de_16_caracteres_de_google"\n',
                language="toml",
            )
            return
        if st.button("Enviar informe", key=f"{prefix}_send_email"):
            if not recipient or "@" not in recipient:
                st.error("Introduce una direccion de correo valida.")
            else:
                try:
                    send_pdf_email(recipient, subject, body, filename, pdf_bytes)
                    st.success(f"Informe enviado a {recipient}.")
                except Exception as exc:
                    st.error(f"No se pudo enviar el correo: {exc}")


def dynamic_range(data: pd.DataFrame, metric: str) -> list[float] | None:
    values = pd.to_numeric(data[metric], errors="coerce").dropna()
    if values.empty:
        return None
    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        padding = max(abs(min_value) * 0.05, 1.0)
        return [min_value - padding, max_value + padding]
    return [min_value, max_value]


def subgroup_from_impact_column(column: str) -> str:
    return column.removeprefix("impacto_")


def metric_percentile_for_player(role_df: pd.DataFrame, player_index: int, metric: str) -> float | None:
    values = pd.to_numeric(role_df[metric], errors="coerce")
    if values.notna().sum() < 2 or player_index not in values.index or pd.isna(values.loc[player_index]):
        return None
    higher_is_better = metric not in NEGATIVE_METRICS
    percentiles = values.rank(pct=True, ascending=higher_is_better) * 100
    return float(percentiles.loc[player_index])


def subimpact_variable_table(df: pd.DataFrame, row: pd.Series, subgroup: str) -> pd.DataFrame:
    role_df = df[df["role"].eq(row["role"])].copy()
    metrics = [metric for metric in METRIC_SUBGROUPS.get(subgroup, []) if metric in role_df.columns]
    rows = []
    for metric in metrics:
        raw_value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
        percentile = metric_percentile_for_player(role_df, row.name, metric)
        rows.append(
            {
                "Variable": DISPLAY_NAMES.get(metric, metric),
                "Explicacion": metric_description_text(metric),
                "Valor del jugador": np.nan if pd.isna(raw_value) else round(float(raw_value), 3),
                "Percentil ajustado": np.nan if percentile is None else round(percentile, 1),
            }
        )
    return pd.DataFrame(rows)


def percentil_bar(value: float | int | None) -> str:
    if pd.isna(value):
        return ""
    pct = max(0.0, min(100.0, float(value)))
    return (
        f'<div style="width:100%; min-width:180px;">'
        f'<div style="height:12px; border-radius:999px; '
        f'background:linear-gradient(90deg,#b91c1c 0%,#f97316 35%,#fde047 65%,#15803d 100%);">'
        f'<div style="height:18px; width:2px; margin-left:{pct:.1f}%; '
        f'background:#10251d; transform:translateY(-3px);"></div>'
        f'</div>'
        f'<div style="font-size:.82rem; margin-top:.15rem; color:#47564e;">{pct:.1f}</div>'
        f'</div>'
    )


def render_subimpact_breakdown(df: pd.DataFrame, row: pd.Series, radar_cols: list[str]) -> None:
    subgroups = [
        subgroup_from_impact_column(col)
        for col in radar_cols
        if subgroup_from_impact_column(col) in METRIC_SUBGROUPS
    ]
    subgroups = list(dict.fromkeys(subgroups))
    if not subgroups:
        return

    st.subheader("Variables que forman cada subimpacto")
    st.caption(
        "Selecciona un subimpacto para ver sus variables. El percentil esta ajustado: verde siempre significa mejor rendimiento, tambien en variables donde menor valor es mejor."
    )

    selected = st.selectbox(
        "Subimpacto",
        subgroups,
        format_func=lambda subgroup: SUBIMPACT_LABELS.get(subgroup, display_label(f"impacto_{subgroup}")),
        help="Elige el subimpacto para ver las variables que lo componen.",
    )
    impact_col = f"impacto_{selected}"
    if impact_col in row.index and pd.notna(row.get(impact_col)):
        st.metric(f"Subimpacto seleccionado: {SUBIMPACT_LABELS.get(selected, display_label(impact_col))}", f"{float(row[impact_col]):.1f}")

    table = subimpact_variable_table(df, row, selected)
    if table.empty:
        st.info("No hay variables disponibles en el dataset para este subimpacto.")
        return
    shown = table.sort_values("Percentil ajustado", ascending=False).copy()
    shown["Percentil ajustado"] = shown["Percentil ajustado"].apply(percentil_bar)
    shown = shown[["Variable", "Explicacion", "Valor del jugador", "Percentil ajustado"]]
    st.markdown(shown.to_html(escape=False, index=False), unsafe_allow_html=True)



def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ll-red: #ff2b2b;
            --ll-deep-red: #b00020;
            --ll-black: #070707;
            --ll-ink: #151515;
            --ll-muted: #626262;
            --ll-line: rgba(7, 7, 7, .10);
            --ll-card: rgba(255, 255, 255, .92);
            --ll-cream: #f4f1ea;
        }
        .stApp {
            background:
              radial-gradient(circle at 8% 10%, rgba(255, 43, 43, .20), transparent 22rem),
              radial-gradient(circle at 92% 4%, rgba(255, 255, 255, .86), transparent 24rem),
              linear-gradient(135deg, rgba(7,7,7,.045) 25%, transparent 25%) 0 0 / 28px 28px,
              linear-gradient(180deg, #f7f5ef 0%, #ece8df 100%);
            color: var(--ll-ink);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1480px;
        }
        [data-testid="stSidebar"] {
            background:
              linear-gradient(180deg, #090909 0%, #181818 62%, #2a0508 100%);
            border-right: 4px solid var(--ll-red);
        }
        [data-testid="stSidebar"] * {
            color: #ffffff;
        }
        [data-testid="stSidebar"] .stMultiSelect div,
        [data-testid="stSidebar"] .stSlider div,
        [data-testid="stSidebar"] input {
            color: var(--ll-ink);
        }
        [data-testid="stMetric"] {
            background: var(--ll-card);
            border: 1px solid var(--ll-line);
            border-left: 6px solid var(--ll-red);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: 0 18px 36px rgba(0, 0, 0, .08);
        }
        [data-testid="stMetricValue"] {
            color: var(--ll-black);
            font-weight: 900;
            font-size: 1.55rem;
            line-height: 1.05;
            letter-spacing: -.03em;
        }
        [data-testid="stMetricLabel"] {
            color: var(--ll-muted);
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .06em;
            font-size: .72rem;
        }
        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 18px;
            padding: 1.7rem 1.8rem;
            color: #fff;
            background:
              linear-gradient(90deg, rgba(255,255,255,.08) 49.7%, rgba(255,255,255,.55) 50%, rgba(255,255,255,.08) 50.3%),
              radial-gradient(circle at 50% 50%, transparent 0 70px, rgba(255,255,255,.35) 71px 73px, transparent 74px),
              linear-gradient(135deg, #050505 0%, #171717 58%, #d71920 100%);
            box-shadow: 0 28px 70px rgba(0, 0, 0, .20);
            border: 1px solid rgba(255,255,255,.10);
        }
        .hero:before {
            content: "LALIGA STYLE ANALYTICS";
            display: inline-block;
            background: var(--ll-red);
            color: #fff;
            font-size: .72rem;
            font-weight: 900;
            letter-spacing: .14em;
            padding: .35rem .55rem;
            border-radius: 999px;
            margin-bottom: .75rem;
        }
        .hero h1 {
            color: #fff;
            font-size: clamp(2rem, 4vw, 3.9rem);
            line-height: .95;
            letter-spacing: -.06em;
            margin: .15rem 0 .55rem 0;
            max-width: 980px;
        }
        .small-note {
            color: rgba(255,255,255,.82);
            font-size: 1rem;
            max-width: 780px;
        }
        .laliga-logo {
            display: block;
            object-fit: contain;
            filter: drop-shadow(0 14px 26px rgba(0,0,0,.24));
        }
        .hero-logo {
            position: absolute;
            right: 1.6rem;
            bottom: 1.2rem;
            width: min(18vw, 150px);
            max-height: 120px;
            opacity: .92;
        }
        .login-logo {
            width: 88px;
            margin-bottom: .8rem;
        }
        .sidebar-logo {
            width: 92px;
            margin: .2rem 0 .8rem 0;
            filter: drop-shadow(0 10px 18px rgba(0,0,0,.35));
        }
        .team-badge {
            width: 42px;
            height: 42px;
            object-fit: contain;
            vertical-align: middle;
            margin-right: .55rem;
            filter: drop-shadow(0 6px 10px rgba(0,0,0,.16));
        }
        .team-badge-small {
            width: 24px;
            height: 24px;
            object-fit: contain;
            vertical-align: middle;
            margin-right: .45rem;
        }
        .team-name-cell {
            display: flex;
            align-items: center;
            gap: .45rem;
            font-weight: 800;
        }
        .team-title-card {
            display: flex;
            align-items: center;
            gap: .35rem;
            background: rgba(255,255,255,.88);
            border: 1px solid var(--ll-line);
            border-left: 5px solid var(--ll-red);
            border-radius: 12px;
            padding: .75rem 1rem;
            margin: .4rem 0 1rem 0;
            box-shadow: 0 12px 26px rgba(0,0,0,.07);
            font-weight: 900;
        }
        .metric-card-custom {
            min-height: 112px;
            background: var(--ll-card);
            border: 1px solid var(--ll-line);
            border-left: 6px solid var(--ll-red);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: 0 18px 36px rgba(0, 0, 0, .08);
        }
        .metric-card-custom .metric-label {
            color: var(--ll-muted);
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .06em;
            font-size: .72rem;
            margin-bottom: .55rem;
        }
        .metric-card-custom .metric-value {
            display: flex;
            align-items: center;
            gap: .65rem;
            color: var(--ll-black);
            font-weight: 900;
            font-size: 1.55rem;
            line-height: 1.05;
            letter-spacing: -.03em;
        }
        .metric-card-custom .metric-badge {
            width: 42px;
            height: 42px;
            object-fit: contain;
            flex: 0 0 auto;
        }
        .player-identity-card {
            display: grid;
            grid-template-columns: 156px minmax(0, 1fr);
            gap: 1.2rem;
            align-items: stretch;
            background:
              linear-gradient(135deg, rgba(255,255,255,.96) 0%, rgba(255,255,255,.88) 100%),
              radial-gradient(circle at 100% 0%, rgba(255,43,43,.20), transparent 18rem);
            border: 1px solid var(--ll-line);
            border-left: 7px solid var(--ll-red);
            border-radius: 16px;
            padding: 1rem;
            margin: .9rem 0 1.1rem 0;
            box-shadow: 0 22px 48px rgba(0,0,0,.10);
        }
        .player-photo-frame {
            min-height: 190px;
            border-radius: 14px;
            overflow: hidden;
            background:
              linear-gradient(135deg, rgba(7,7,7,.08) 25%, transparent 25%) 0 0 / 18px 18px,
              linear-gradient(180deg, #111 0%, #2a2a2a 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(0,0,0,.14);
        }
        .player-photo {
            width: 100%;
            height: 100%;
            min-height: 190px;
            object-fit: cover;
            display: block;
        }
        .player-photo-placeholder {
            width: 100%;
            height: 100%;
            min-height: 190px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #fff;
            text-align: center;
            gap: .4rem;
        }
        .player-photo-placeholder span {
            width: 72px;
            height: 72px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: var(--ll-red);
            color: #fff;
            font-weight: 950;
            font-size: 1.8rem;
            box-shadow: 0 14px 26px rgba(255,43,43,.28);
        }
        .player-photo-placeholder small {
            color: rgba(255,255,255,.78);
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .06em;
            font-size: .68rem;
        }
        .player-identity-main {
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: .75rem;
        }
        .player-eyebrow {
            width: fit-content;
            background: var(--ll-black);
            color: #fff;
            border-radius: 999px;
            padding: .32rem .62rem;
            font-size: .68rem;
            font-weight: 950;
            text-transform: uppercase;
            letter-spacing: .12em;
        }
        .player-identity-main h2 {
            margin: 0;
            font-size: clamp(1.9rem, 4vw, 3.4rem);
            line-height: .92;
            letter-spacing: -.065em;
            color: var(--ll-black);
        }
        .player-meta-row {
            display: flex;
            gap: .65rem;
            flex-wrap: wrap;
            align-items: center;
        }
        .player-meta-pill {
            display: inline-flex;
            align-items: center;
            gap: .48rem;
            background: rgba(255,255,255,.82);
            border: 1px solid var(--ll-line);
            border-radius: 999px;
            padding: .45rem .72rem;
            font-weight: 850;
            color: var(--ll-ink);
        }
        .player-team-badge,
        .flag-icon {
            width: 28px;
            height: 28px;
            object-fit: contain;
            flex: 0 0 auto;
        }
        .flag-placeholder {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: #f2f2f2;
            border: 1px dashed rgba(0,0,0,.25);
            color: var(--ll-muted);
            font-weight: 950;
        }
        .muted-pill {
            color: var(--ll-muted);
        }
        .player-tag-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .75rem;
            margin-top: .15rem;
        }
        .player-tag {
            background: #fff;
            border: 1px solid var(--ll-line);
            border-radius: 12px;
            padding: .8rem .9rem;
            min-width: 0;
        }
        .player-tag span {
            display: block;
            color: var(--ll-muted);
            text-transform: uppercase;
            letter-spacing: .07em;
            font-weight: 850;
            font-size: .68rem;
            margin-bottom: .28rem;
        }
        .player-tag strong {
            display: block;
            color: var(--ll-black);
            font-weight: 950;
            font-size: clamp(1rem, 2vw, 1.45rem);
            line-height: 1.05;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .player-tag.highlight {
            background: linear-gradient(135deg, #111 0%, #2a2a2a 100%);
            border-color: rgba(255,255,255,.12);
        }
        .player-tag.highlight span,
        .player-tag.highlight strong {
            color: #fff;
        }
        .player-tag.highlight strong {
            font-size: clamp(1.35rem, 2.8vw, 2.1rem);
        }
        .saved-player-card {
            display: grid;
            grid-template-columns: 92px minmax(0, 1fr);
            gap: 1rem;
            align-items: center;
            margin-bottom: .85rem;
        }
        .saved-player-card h3 {
            margin: 0 0 .35rem 0;
            color: var(--ll-black);
            font-size: 1.2rem;
            letter-spacing: -.03em;
        }
        .saved-player-card p {
            margin: .15rem 0;
            color: var(--ll-muted);
            font-weight: 700;
        }
        .list-player-photo {
            width: 92px;
            height: 112px;
            object-fit: cover;
            border-radius: 12px;
            background: #111;
            border: 1px solid rgba(0,0,0,.12);
            box-shadow: 0 12px 24px rgba(0,0,0,.14);
        }
        .list-photo-placeholder {
            display: grid;
            place-items: center;
            color: #fff;
            font-size: 1.5rem;
            font-weight: 950;
        }
        .comparison-photo-strip {
            display: flex;
            flex-wrap: wrap;
            gap: .8rem;
            align-items: flex-start;
            margin: .85rem 0 1rem 0;
        }
        .comparison-photo-item {
            width: 92px;
            text-align: center;
        }
        .comparison-photo-item .list-player-photo {
            width: 76px;
            height: 90px;
            border-radius: 10px;
        }
        .comparison-photo-item strong {
            display: block;
            margin-top: .35rem;
            color: var(--ll-black);
            font-size: .78rem;
            line-height: 1.08;
            font-weight: 900;
        }
        .fit-top-section {
            margin: 1.15rem 0 1.25rem 0;
        }
        .fit-top-header {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: .75rem;
        }
        .fit-top-header span {
            background: var(--ll-red);
            color: #fff;
            border-radius: 999px;
            padding: .35rem .7rem;
            font-size: .72rem;
            font-weight: 950;
            text-transform: uppercase;
            letter-spacing: .1em;
        }
        .fit-top-header strong {
            color: var(--ll-black);
            font-weight: 950;
            font-size: clamp(1.05rem, 2vw, 1.45rem);
            letter-spacing: -.035em;
        }
        .fit-top-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: .85rem;
        }
        .fit-player-card {
            position: relative;
            overflow: hidden;
            background: rgba(255,255,255,.94);
            border: 1px solid var(--ll-line);
            border-radius: 16px;
            box-shadow: 0 18px 38px rgba(0,0,0,.10);
        }
        .fit-player-card:before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(255,43,43,.12), transparent 42%);
            pointer-events: none;
        }
        .fit-rank-badge {
            position: absolute;
            z-index: 3;
            top: .65rem;
            left: .65rem;
            width: 42px;
            height: 42px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: var(--ll-red);
            color: #fff;
            font-weight: 950;
            box-shadow: 0 12px 24px rgba(255,43,43,.34);
        }
        .fit-player-photo-frame {
            height: 210px;
            background:
              linear-gradient(135deg, rgba(7,7,7,.08) 25%, transparent 25%) 0 0 / 16px 16px,
              linear-gradient(180deg, #121212 0%, #303030 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .fit-player-photo-frame .player-photo {
            width: 100%;
            height: 100%;
            min-height: 210px;
            object-fit: cover;
        }
        .fit-player-photo-frame .player-photo-placeholder {
            min-height: 210px;
        }
        .fit-player-body {
            position: relative;
            z-index: 2;
            padding: .85rem .85rem .95rem .85rem;
        }
        .fit-player-body h3 {
            margin: 0 0 .55rem 0;
            color: var(--ll-black);
            font-size: 1.05rem;
            line-height: 1.05;
            letter-spacing: -.045em;
            min-height: 2.15rem;
        }
        .fit-team-line {
            display: flex;
            align-items: center;
            gap: .35rem;
            color: var(--ll-muted);
            font-weight: 850;
            font-size: .82rem;
            margin-bottom: .45rem;
        }
        .fit-team-badge {
            width: 24px;
            height: 24px;
            object-fit: contain;
            flex: 0 0 auto;
        }
        .fit-profile-line {
            color: var(--ll-muted);
            font-weight: 800;
            font-size: .76rem;
            line-height: 1.15;
            min-height: 1.8rem;
            margin-bottom: .7rem;
        }
        .fit-score-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: .5rem;
        }
        .fit-score-row div {
            background: #111;
            color: #fff;
            border-radius: 10px;
            padding: .55rem .6rem;
        }
        .fit-score-row span {
            display: block;
            color: rgba(255,255,255,.72);
            text-transform: uppercase;
            letter-spacing: .06em;
            font-size: .62rem;
            font-weight: 850;
        }
        .fit-score-row strong {
            display: block;
            color: #fff;
            font-size: 1.35rem;
            line-height: 1;
            font-weight: 950;
            letter-spacing: -.04em;
        }
        @media (max-width: 760px) {
            .player-identity-card {
                grid-template-columns: 1fr;
            }
            .player-photo-frame {
                min-height: 260px;
            }
            .player-tag-grid {
                grid-template-columns: 1fr;
            }
            .fit-top-header {
                align-items: flex-start;
                flex-direction: column;
            }
            .fit-top-grid {
                grid-template-columns: 1fr;
            }
        }
        @media (min-width: 761px) and (max-width: 1180px) {
            .fit-top-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        h1, h2, h3 {
            color: var(--ll-black);
            font-weight: 900;
            letter-spacing: -.04em;
        }
        p, li, label, span {
            color: inherit;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--ll-line);
            border-radius: 0;
            overflow: hidden;
            box-shadow: 0 12px 28px rgba(0,0,0,.06);
        }
        div[data-testid="stDataFrame"] div[role="columnheader"],
        div[data-testid="stDataFrame"] div[role="columnheader"] * {
            background: #111 !important;
            color: #fff !important;
            font-weight: 950 !important;
            text-transform: uppercase !important;
            letter-spacing: .04em !important;
        }
        div[data-testid="stDataFrame"] div[role="gridcell"],
        div[data-testid="stDataFrame"] div[role="gridcell"] * {
            font-size: 1rem !important;
            color: #050505;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: .45rem;
            background: rgba(255,255,255,.72);
            padding: .45rem;
            border-radius: 999px;
            border: 1px solid var(--ll-line);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 999px;
            padding: .55rem 1rem;
            color: var(--ll-ink);
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: .035em;
            font-size: .78rem;
        }
        .stTabs [aria-selected="true"] {
            background: var(--ll-red) !important;
            color: #fff !important;
            box-shadow: 0 10px 22px rgba(215,25,32,.28);
        }
        div[data-testid="stAlert"] {
            background: rgba(255, 255, 255, .92);
            color: var(--ll-ink);
            border-radius: 10px;
            border-left: 5px solid var(--ll-red);
        }
        .stPlotlyChart {
            background: rgba(255,255,255,.94);
            border-radius: 12px;
            padding: .75rem;
            border: 1px solid var(--ll-line);
            box-shadow: 0 18px 42px rgba(0, 0, 0, .08);
        }
        .login-card {
            max-width: 520px;
            margin: 7vh auto 1.4rem auto;
            padding: 1.7rem 1.9rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #050505 0%, #171717 58%, #d71920 100%);
            color: #fff;
            box-shadow: 0 28px 70px rgba(0, 0, 0, .24);
            border: 1px solid rgba(255,255,255,.16);
        }
        .login-card h1 {
            color: #fff;
            margin-bottom: .35rem;
        }
        .login-card p {
            color: rgba(255,255,255,.78);
            margin-bottom: 0;
        }
        button[kind="primary"], .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(7,7,7,.14);
            font-weight: 800;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            background: rgba(255,255,255,.82);
            border: 1px solid var(--ll-line);
            border-radius: 10px;
            overflow: hidden;
        }
        th {
            background: #111;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: .04em;
            font-size: .78rem;
            padding: .75rem;
        }
        td {
            border-top: 1px solid rgba(0,0,0,.08);
            padding: .7rem .75rem;
        }
        .laliga-rank-table {
            border-collapse: collapse;
            width: 100%;
            background: rgba(255,255,255,.84);
            border: 0;
            border-radius: 0;
            overflow: hidden;
            font-size: 1.05rem;
        }
        .laliga-rank-table th {
            background: #111 !important;
            color: #fff !important;
            text-transform: uppercase;
            letter-spacing: .035em;
            font-weight: 950;
            font-size: .82rem;
            text-align: center;
            padding: .9rem 1rem;
            border: 0;
        }
        .laliga-rank-table td {
            color: #050505;
            padding: .95rem 1rem;
            border-top: 1px solid rgba(0,0,0,.08);
            border-right: 1px solid rgba(0,0,0,.06);
            vertical-align: middle;
            line-height: 1.55;
        }
        .laliga-rank-table td:first-child {
            font-weight: 500;
            min-width: 190px;
        }
        .laliga-rank-table .team-name-cell {
            font-weight: 950;
            font-size: 1.05rem;
        }
        .laliga-rank-table .team-badge-small {
            width: 30px;
            height: 30px;
            margin-right: .6rem;
        }
        .compact-table-wrap {
            width: 100%;
            overflow-x: auto;
            border: 1px solid rgba(7,7,7,.10);
            background: rgba(255,255,255,.72);
        }
        .compact-table-wrap .laliga-rank-table {
            min-width: 0;
            table-layout: fixed;
            font-size: .82rem;
        }
        .compact-table-wrap .laliga-rank-table th {
            font-size: .68rem;
            padding: .55rem .45rem;
            white-space: normal;
        }
        .compact-table-wrap .laliga-rank-table td {
            padding: .55rem .45rem;
            line-height: 1.25;
            overflow-wrap: anywhere;
            word-break: normal;
        }
        .compact-table-wrap .laliga-rank-table td:first-child {
            min-width: 0;
        }
        .comparison-impact-table td {
            text-align: right;
        }
        .comparison-impact-table td:first-child {
            text-align: left;
        }
        .comparison-impact-table .best-impact-cell {
            background: rgba(22, 163, 74, .18);
            color: #14532d;
            font-weight: 950;
            box-shadow: inset 0 0 0 1px rgba(22, 163, 74, .35);
        }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        #MainMenu,
        header {
            visibility: hidden;
            height: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_processed_data(data_version: float | None = None) -> pd.DataFrame | None:
    scored_path = PROCESSED_DIR / "players_scored.parquet"
    master_path = PROCESSED_DIR / "master_laliga_players.parquet"

    if scored_path.exists():
        df = ensure_model_role(pd.read_parquet(scored_path))
        return assign_cluster_labels(df) if "cluster_label" not in df.columns else df

    if master_path.exists():
        df = ensure_model_role(pd.read_parquet(master_path))
        return calculate_impact_scores(df)

    return None


def processed_data_version() -> float | None:
    paths = [
        PROCESSED_DIR / "players_scored.parquet",
        PROCESSED_DIR / "master_laliga_players.parquet",
        ATHLETIC_ELIGIBLE_PATH,
    ]
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    return max(mtimes) if mtimes else None


def secrets_available() -> bool:
    return bool(secret_or_env("SB_USERNAME") and secret_or_env("SB_PASSWORD"))


@st.cache_data(show_spinner=False)
def load_transfermarkt_identity() -> pd.DataFrame:
    if not TRANSFERMARKT_EXTERNAL_PATH.exists():
        return pd.DataFrame()
    cols = ["IdJugador", "Nacionalidad", "Imagen Nacionalidad", "Imagen", "Altura"]
    raw = pd.read_excel(TRANSFERMARKT_EXTERNAL_PATH, usecols=lambda col: col in cols)
    rename = {
        "IdJugador": "player_id_tm",
        "Nacionalidad": "nationality",
        "Imagen Nacionalidad": "nationality_flag_url",
        "Imagen": "transfermarkt_photo_url",
        "Altura": "player_height",
    }
    out = raw.rename(columns=rename)
    out["player_id_tm"] = pd.to_numeric(out["player_id_tm"], errors="coerce")
    if "player_height" in out.columns:
        out["player_height"] = pd.to_numeric(out["player_height"], errors="coerce")
    return out.dropna(subset=["player_id_tm"]).drop_duplicates("player_id_tm")


def add_identity_display_data(df: pd.DataFrame) -> pd.DataFrame:
    identity = load_transfermarkt_identity()
    if identity.empty or "player_id_tm" not in df.columns:
        return df
    output = df.copy()
    output["player_id_tm"] = pd.to_numeric(output["player_id_tm"], errors="coerce")
    cols_to_add = [c for c in identity.columns if c not in output.columns or c == "player_id_tm"]
    output = output.merge(identity[cols_to_add], on="player_id_tm", how="left")

    flag_lookup = (
        identity.dropna(subset=["nationality", "nationality_flag_url"])
        .drop_duplicates("nationality")
        .set_index("nationality")["nationality_flag_url"]
        .to_dict()
    )
    manual_nationality = output["Name"].map(MANUAL_NATIONALITIES) if "Name" in output.columns else pd.Series(index=output.index)
    output["nationality"] = output["nationality"].fillna(manual_nationality)
    manual_flags = output["nationality"].map(flag_lookup)
    output["nationality_flag_url"] = output["nationality_flag_url"].fillna(manual_flags)
    return output


@st.cache_data(show_spinner=True)
def download_and_model_from_statsbomb() -> pd.DataFrame:
    os.environ["SB_USERNAME"] = secret_or_env("SB_USERNAME", "") or ""
    os.environ["SB_PASSWORD"] = secret_or_env("SB_PASSWORD", "") or ""
    player_stats, team_stats, _ = load_statsbomb_laliga(force_download=False)
    master = build_master_dataset(player_stats, team_stats)
    pcas = fit_all_role_pcas(master)
    modeled, _ = fit_all_clusters(master, pcas)
    scored = calculate_impact_scores(modeled)
    return scored


@st.cache_data(show_spinner=False)
def load_athletic_eligibility() -> pd.DataFrame:
    if not ATHLETIC_ELIGIBLE_PATH.exists():
        return pd.DataFrame(columns=["matched_name_key", "birth_place"])
    try:
        eligible = pd.read_csv(ATHLETIC_ELIGIBLE_PATH)
    except Exception:
        return pd.DataFrame(columns=["matched_name_key", "birth_place"])
    if "matched_name_key" not in eligible.columns:
        return pd.DataFrame(columns=["matched_name_key", "birth_place"])
    eligible = eligible.loc[eligible["matched_name_key"].notna()].copy()
    eligible["matched_name_key"] = eligible["matched_name_key"].astype(str).map(normalize_name)
    eligible = eligible.loc[eligible["matched_name_key"].ne("")]
    if "birth_place" not in eligible.columns:
        eligible["birth_place"] = ""
    return eligible.drop_duplicates("matched_name_key")[["matched_name_key", "birth_place"]]


def add_athletic_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Name" not in df.columns:
        return df
    eligible = load_athletic_eligibility()
    enriched = df.copy()
    enriched["athletic_eligible"] = False
    enriched["athletic_birth_place"] = ""
    if eligible.empty:
        return enriched
    birth_place_by_name = eligible.set_index("matched_name_key")["birth_place"].to_dict()
    name_keys = enriched["Name"].map(normalize_name)
    enriched["athletic_eligible"] = name_keys.isin(birth_place_by_name)
    enriched["athletic_birth_place"] = name_keys.map(birth_place_by_name).fillna("")
    return enriched


@st.cache_data(show_spinner=False)
def get_data(data_version: float | None = None) -> pd.DataFrame | None:
    df = load_processed_data(data_version)
    if df is not None:
        return add_athletic_eligibility(add_identity_display_data(add_economic_data(df)))
    if secrets_available():
        return add_athletic_eligibility(add_identity_display_data(add_economic_data(download_and_model_from_statsbomb())))
    return None


def normalize_auth_username(username: str) -> str:
    return re.sub(r"\s+", " ", username).strip()


def load_auth_users() -> dict[str, list[dict[str, object]]]:
    if not AUTH_USERS_PATH.exists():
        return {"users": []}
    try:
        data = json.loads(AUTH_USERS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"users": []}
    users = data.get("users", [])
    return {"users": users if isinstance(users, list) else []}


def save_auth_users(data: dict[str, list[dict[str, object]]]) -> None:
    AUTH_USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTH_USERS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def password_hash(password: str, salt_hex: str, iterations: int) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        int(iterations),
    )
    return digest.hex()


def make_password_record(password: str) -> dict[str, object]:
    salt_hex = secrets.token_hex(16)
    return {
        "algorithm": "pbkdf2_sha256",
        "iterations": PASSWORD_HASH_ITERATIONS,
        "salt": salt_hex,
        "password_hash": password_hash(password, salt_hex, PASSWORD_HASH_ITERATIONS),
    }


def find_auth_user(username: str, users: list[dict[str, object]]) -> dict[str, object] | None:
    normalized = normalize_auth_username(username).casefold()
    for user in users:
        if str(user.get("username", "")).casefold() == normalized:
            return user
    return None


def verify_auth_user(username: str, password: str) -> bool:
    user = find_auth_user(username, load_auth_users()["users"])
    if not user:
        return False
    salt = str(user.get("salt", ""))
    expected = str(user.get("password_hash", ""))
    iterations = int(user.get("iterations", PASSWORD_HASH_ITERATIONS))
    if not salt or not expected:
        return False
    candidate = password_hash(password, salt, iterations)
    return hmac.compare_digest(candidate, expected)


def register_auth_user(username: str, password: str, repeat_password: str) -> tuple[bool, str]:
    username = normalize_auth_username(username)
    if len(username) < 3:
        return False, "El usuario debe tener al menos 3 caracteres."
    if len(password) < 8:
        return False, "La contraseña debe tener al menos 8 caracteres."
    if password != repeat_password:
        return False, "Las contraseñas no coinciden."

    data = load_auth_users()
    if find_auth_user(username, data["users"]):
        return False, "Ese usuario ya existe."

    record = {
        "username": username,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **make_password_record(password),
    }
    data["users"].append(record)
    save_auth_users(data)
    return True, "Usuario registrado correctamente. Ya puedes iniciar sesión."


def login_screen() -> bool:
    if st.session_state.get("authenticated", False):
        return True

    st.markdown(
        f"""
        <div class="login-card">
            {laliga_logo_html("login-logo")}
            <h1>Acceso privado</h1>
            <p>Introduce tus credenciales para acceder al panel del TFG.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, register_tab = st.tabs(["Iniciar sesión", "Registrarse"])
    with login_tab:
        if not load_auth_users()["users"]:
            st.info("Todavía no hay usuarios registrados. Crea una cuenta en la pestaña Registrarse.")
        with st.form("login_form"):
            username = st.text_input("Usuario", key="login_username")
            password = st.text_input("Contraseña", type="password", key="login_password")
            submitted = st.form_submit_button("Entrar")

        if submitted:
            if verify_auth_user(username, password):
                st.session_state["authenticated"] = True
                st.session_state["auth_user"] = normalize_auth_username(username)
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Usuario", key="register_username")
            new_password = st.text_input("Contraseña", type="password", key="register_password")
            repeat_password = st.text_input("Repetir contraseña", type="password", key="register_repeat_password")
            register_submitted = st.form_submit_button("Crear cuenta")

        if register_submitted:
            ok, message = register_auth_user(new_username, new_password, repeat_password)
            if ok:
                st.success(message)
            else:
                st.error(message)
    return False


@st.cache_data(show_spinner=False)
def get_team_similarity(df: pd.DataFrame):
    return calculate_team_similarity(df)


def filter_players(df: pd.DataFrame) -> pd.DataFrame:
    teams = sorted(df["Team"].dropna().unique())
    roles = sorted(df["role"].dropna().unique())

    with st.sidebar:
        st.header("Filtros")
        selected_teams = st.multiselect(
            "Equipos",
            teams,
            default=teams,
            help="Filtra el panel para analizar solo determinados clubes.",
        )
        selected_roles = st.multiselect(
            "Roles",
            roles,
            default=roles,
            format_func=lambda r: ROLE_LABELS.get(r, r),
            help="Los roles son macro-posiciones modeladas: portero, central, lateral, mediocentro, extremo y delantero.",
        )
        min_minutes = int(df["Minutes"].min()) if "Minutes" in df.columns else 0
        max_minutes = int(df["Minutes"].max()) if "Minutes" in df.columns else 0
        minutes_range = st.slider(
            "Minutos",
            min_minutes,
            max_minutes,
            (min_minutes, max_minutes),
            help="Permite excluir jugadores con poca muestra de minutos o centrarse en futbolistas más habituales.",
        )

    out = df[df["Team"].isin(selected_teams) & df["role"].isin(selected_roles)].copy()
    if "Minutes" in out.columns:
        out = out[out["Minutes"].between(minutes_range[0], minutes_range[1])]
    return out


def metric_cards(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jugadores", f"{len(df):,}")
    c2.metric("Equipos", f"{df['Team'].nunique():,}")
    c3.metric("Roles modelados", f"{df['role'].nunique():,}")
    c4.metric("Impacto medio", f"{df['impacto_global'].mean():.1f}" if "impacto_global" in df else "N/D")


def team_title_card(team: str) -> None:
    badge = team_badge_html(team)
    if not badge:
        return
    st.markdown(
        f'<div class="team-title-card">{badge}<span>{team}</span></div>',
        unsafe_allow_html=True,
    )


def team_metric_card(team: str) -> None:
    badge = team_badge_html(team, "metric-badge")
    st.markdown(
        f"""
        <div class="metric-card-custom">
            <div class="metric-label">Equipo</div>
            <div class="metric-value">{badge}<span>{team}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_similar_teams_table(similar_df: pd.DataFrame) -> None:
    rows = []
    for _, row in similar_df.iterrows():
        team = row["Equipo"]
        badge = team_badge_html(team, "team-badge-small")
        similarity = float(row["similitud_coseno"])
        rows.append(
            {
                "Equipo": f'<div class="team-name-cell">{badge}<span>{team}</span></div>',
                "Similitud de estilo": f"{similarity:.3f}",
            }
        )
    html = pd.DataFrame(rows).to_html(escape=False, index=False)
    st.markdown(html, unsafe_allow_html=True)


def add_team_badges_to_pca_fig(fig: go.Figure, data: pd.DataFrame, size: float = 0.72) -> go.Figure:
    images = []
    missing_badge_rows = []
    for _, row in data.iterrows():
        team = row["Team"]
        x = float(row["Team_PC1"])
        y = float(row["Team_PC2"])
        badge_uri = team_badge_uri(team)
        if badge_uri:
            images.append(
                dict(
                    source=badge_uri,
                    xref="x",
                    yref="y",
                    x=x,
                    y=y,
                    sizex=size,
                    sizey=size,
                    xanchor="center",
                    yanchor="middle",
                    layer="above",
                )
            )
        else:
            missing_badge_rows.append(row)

    fig.update_layout(images=images)
    fig.update_traces(mode="markers", text=None, marker=dict(size=1, opacity=0))

    if missing_badge_rows:
        fallback = pd.DataFrame(missing_badge_rows)
        fig.add_trace(
            go.Scatter(
                x=fallback["Team_PC1"],
                y=fallback["Team_PC2"],
                mode="markers+text",
                text=fallback["Team"],
                textposition="top center",
                marker=dict(size=13, color="#ff2b2b", line=dict(width=1, color="#111")),
                name="Sin escudo",
                showlegend=False,
            )
        )
    return fig


def overview_tab(df: pd.DataFrame) -> None:
    st.markdown(
        f"""
        <div class="hero">
        {laliga_logo_html("hero-logo")}
        <h1>Impacto, perfiles y encaje en LALIGA</h1>
        <p class="small-note">
        Panel interactivo de scouting y rendimiento construido sobre datos de StatsBomb.
        Compara jugadores por rol, visualiza perfiles, mide similitud tactica entre clubes
        y estima encaje jugador-equipo.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    help_box(
        "Cómo leer esta pantalla",
        """
        Esta vista resume el dataset filtrado. El impacto global está en escala 0-100 y se calcula de forma distinta
        según el rol y el perfil del jugador. En jugadores de campo combina impacto ofensivo, asociativo y defensivo;
        en porteros usa solo impacto de portería.
        """,
    )
    metric_cards(df)

    c1, c2 = st.columns([1, 1])
    with c1:
        role_counts = df["role"].value_counts().rename_axis("role").reset_index(name="jugadores")
        fig = px.bar(
            role_counts,
            x="role",
            y="jugadores",
            color="role",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Distribución por rol",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with c2:
        fig = px.histogram(
            df,
            x="impacto_global",
            nbins=30,
            color="role",
            title="Distribución de impacto global",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, width="stretch")

    top_cols = existing_columns(
        df,
        [
            "Name",
            "Team",
            "role",
            "Minutes",
            "profile_cluster",
            "cluster_label",
            "impacto_global",
            "impacto_ofensivo",
            "impacto_asociativo",
            "impacto_defensivo",
            "impacto_porteria",
        ],
    )
    st.subheader("20 mejores jugadores")
    st.caption("Clasificación general según impacto global, ya ajustado por rol y perfil futbolístico.")
    display_dataframe(df[top_cols].sort_values("impacto_global", ascending=False).head(20), width="stretch")


def rankings_tab(df: pd.DataFrame) -> None:
    st.header("Clasificaciones de jugadores")
    help_box(
        "Qué muestra esta clasificación",
        """
        Aquí puedes comparar jugadores dentro de un rol y, si quieres, dentro de un perfil concreto.
        Las métricas de impacto son percentiles agregados de 0 a 100: cuanto mayor es el valor, mejor posición relativa
        tiene el jugador frente a futbolistas comparables.
        """,
    )
    if df.empty:
        st.warning("No hay jugadores con los filtros actuales.")
        return

    c1, c2, c3, c4 = st.columns([1, 1.35, 1, 1])
    role = c1.selectbox(
        "Rol",
        sorted(df["role"].unique()),
        format_func=lambda r: ROLE_LABELS.get(r, r),
        help="Primero selecciona la macro-posición que quieres analizar.",
    )
    role_df = df[df["role"].eq(role)].copy()

    profile_options = ["Todos"]
    if "cluster_label" in role_df.columns:
        profile_options += sorted(role_df["cluster_label"].dropna().unique())
    selected_profile = c2.selectbox(
        "Perfil",
        profile_options,
        help="Los perfiles salen del agrupamiento estadístico. Puedes elegir todos los jugadores del rol o solo un perfil concreto.",
    )

    metric_options = [c for c in impact_columns_for_role(role, detailed=True) if c in df.columns]
    metric = c3.selectbox(
        "Métrica de clasificación",
        metric_options,
        help="Selecciona la dimensión por la que se ordenará la clasificación.",
    )
    n = c4.slider("Jugadores", 5, 50, 20, help="Número de jugadores que se muestran en el gráfico y la tabla.")

    data = role_df.copy()
    if selected_profile != "Todos":
        data = data[data["cluster_label"].eq(selected_profile)]
    value_map_data = data.copy()
    data = data.sort_values(metric, ascending=False).head(n)
    if data.empty:
        st.warning("No hay jugadores para ese rol/perfil con los filtros actuales.")
        return

    ranking_plot = data.sort_values(metric).copy()
    chart_height = max(520, 30 * len(ranking_plot) + 180)
    fig = px.bar(
        ranking_plot,
        x=metric,
        y="Name",
        color=metric,
        orientation="h",
        labels={col: display_label(col) for col in ranking_plot.columns},
        color_continuous_scale=RANKING_COLOR_SCALE,
        range_color=dynamic_range(ranking_plot, metric),
        hover_data=existing_columns(ranking_plot, ["Minutes", "profile_cluster", "cluster_label"] + impact_columns_for_role(role, detailed=False)),
        title=f"{n} mejores | {ROLE_LABELS.get(role, role)} | {selected_profile} | {display_label(metric)}",
        height=chart_height,
    )
    fig.update_layout(
        coloraxis_colorbar_title=display_label(metric),
        margin=dict(l=220, r=80, t=70, b=60),
        yaxis=dict(
            automargin=True,
            tickmode="array",
            tickvals=ranking_plot["Name"].tolist(),
            ticktext=ranking_plot["Name"].tolist(),
        ),
    )
    ranking_cols = existing_columns(
        data,
        ["Name", "Team", "role", "cluster_label", "Minutes", metric]
        + impact_columns_for_role(role, detailed=False),
    )
    display_dataframe(data[ranking_cols], width="stretch")
    st.plotly_chart(fig, width="stretch")
    render_market_opportunities_table(value_map_data, role, selected_profile)

def player_profile_tab(df: pd.DataFrame) -> None:
    st.header("Perfil de jugador")
    help_box(
        "Cómo interpretar el perfil individual",
        """
        Esta sección muestra el rol, perfil de grupo e impactos del jugador seleccionado.
        El radar agregado resume grandes dimensiones. El radar detallado descompone esas dimensiones en subimpactos
        como finalización, creación, presión o distribución de portero.
        """,
    )
    if df.empty:
        st.warning("No hay jugadores con los filtros actuales.")
        return
    player_options = sorted(df["Name"].dropna().unique())
    if st.session_state.get("player_profile_selected") not in player_options:
        st.session_state["player_profile_selected"] = player_options[0]
    player = st.selectbox(
        "Jugador",
        player_options,
        key="player_profile_selected",
        help="Elige un jugador para ver su perfil estadístico individual.",
    )
    row = df[df["Name"].eq(player)].sort_values("Minutes", ascending=False).iloc[0]

    player_identity_card(row)

    radar_mode = st.radio(
        "Nivel del radar",
        ["Agregado", "Detallado"],
        horizontal=True,
        help="Agregado muestra grandes impactos; Detallado muestra subimpactos más específicos.",
    )
    if row["role"] == "POR":
        if radar_mode == "Detallado":
            candidate_radar_cols = [
                "impacto_shot_stopping",
                "impacto_porteria_juego_aereo",
                "impacto_porteria_distribucion",
            ]
        else:
            candidate_radar_cols = ["impacto_porteria"]
    elif radar_mode == "Detallado":
        candidate_radar_cols = [
            "impacto_finalizacion",
            "impacto_amenaza_area",
            "impacto_creacion",
            "impacto_progresion_pase",
            "impacto_progresion_conduccion",
            "impacto_circulacion",
            "impacto_duelos",
            "impacto_anticipacion",
            "impacto_defensa_area",
            "impacto_presion",
        ]
    else:
        candidate_radar_cols = ["impacto_ofensivo", "impacto_asociativo", "impacto_defensivo"]
    radar_cols = [c for c in candidate_radar_cols if c in df.columns and pd.notna(row.get(c))]
    if not radar_cols:
        st.info("No hay subimpactos disponibles para este jugador con los filtros actuales.")
        return
    radar_values = [float(row[c]) for c in radar_cols]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=[display_label(col) for col in radar_cols] + [display_label(radar_cols[0])],
            fill="toself",
            name=player,
            line_color="#2f6f73",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title="Radar de impactos",
    )
    st.plotly_chart(fig, width="stretch")
    render_subimpact_breakdown(df, row, radar_cols)

    st.subheader("Informe del jugador")
    report_key = f"player_report_{slugify_asset_name(player)}"
    filename = f"informe_jugador_{slugify_asset_name(player)}.pdf"
    if st.button("Generar informe PDF del jugador", key=f"{report_key}_generate", type="primary"):
        try:
            with st.spinner("Generando informe con estilo del dashboard..."):
                pdf_bytes = player_report_pdf(df, row)
                st.session_state[report_key] = pdf_bytes
                save_player_report_entry(row.to_dict(), filename, pdf_bytes)
                st.success("Informe generado y añadido a Listas jugadores.")
        except Exception as exc:
            st.error(f"No se pudo generar el informe: {exc}")
    if report_key in st.session_state:
        render_report_delivery(
            report_key,
            filename,
            st.session_state[report_key],
            f"Informe de jugador | {player}",
            f"Adjunto el informe completo de {player}.",
        )


def teams_tab(df: pd.DataFrame) -> None:
    st.header("Similitud tactica entre equipos")
    help_box(
        "Que significa esta similitud",
        """
        Cada equipo se representa mediante sus estadisticas colectivas de estilo: posesion, circulacion, verticalidad,
        progresion, juego exterior, presion, altura defensiva, transiciones y ritmo. Despues se normalizan esas variables
        y se calcula la similitud del coseno. Un valor cercano a 1 indica que dos equipos se parecen mucho en su forma
        de jugar; un valor mas bajo indica estilos mas diferentes.
        """,
    )
    if df["Team"].nunique() < 3:
        st.warning("Se necesitan al menos tres equipos para construir un mapa tactico util.")
        return
    sim = get_team_similarity(df)
    style_variables = team_style_feature_table(list(sim.team_profiles.columns))

    c1, c2 = st.columns([1, 1])
    with c1:
        team = st.selectbox(
            "Equipo de referencia",
            sorted(sim.cosine_matrix.index),
            help="El equipo elegido se compara contra todos los demas para encontrar los estilos mas parecidos.",
        )
    with c2:
        n = st.slider(
            "Equipos similares",
            3,
            10,
            5,
            help="Numero de equipos mas parecidos que apareceran en la tabla.",
        )
    team_title_card(team)

    equipos_similares = similar_teams(sim, team, n=n).to_frame("similitud_coseno").reset_index(names="Equipo")
    render_similar_teams_table(equipos_similares)

    with st.expander("Variables utilizadas en la matriz de similitud del coseno"):
        st.caption(
            "La matriz compara equipos con estas variables colectivas de estilo. Antes de calcular el coseno, las variables se imputan y se estandarizan."
        )
        st.dataframe(style_variables, width="stretch", hide_index=True)

    heat = sim.cosine_matrix.round(2)
    fig = px.imshow(
        heat,
        text_auto=True,
        color_continuous_scale="BrBG",
        zmin=-1,
        zmax=1,
        title="Matriz de similitud del coseno",
        height=760,
    )
    fig.update_layout(margin=dict(l=150, r=80, t=70, b=140))
    st.plotly_chart(fig, width="stretch")

    data = sim.pca_map.reset_index(names="Team")
    explained = sim.pca_model.explained_variance_ratio_
    pc1_var = float(explained[0] * 100) if len(explained) > 0 else np.nan
    pc2_var = float(explained[1] * 100) if len(explained) > 1 else np.nan
    total_var = float(explained[:2].sum() * 100) if len(explained) else np.nan

    with st.expander("Variables utilizadas en el mapa PCA de equipos"):
        st.caption(
            "El PCA usa las mismas variables colectivas normalizadas que la matriz de similitud, pero las resume en dos componentes principales para poder dibujar el mapa."
        )
        st.dataframe(style_variables, width="stretch", hide_index=True)

    v1, v2, v3 = st.columns(3)
    v1.metric("Varianza explicada PC1", f"{pc1_var:.1f}%" if pd.notna(pc1_var) else "N/D")
    v2.metric("Varianza explicada PC2", f"{pc2_var:.1f}%" if pd.notna(pc2_var) else "N/D")
    v3.metric("Varianza explicada total", f"{total_var:.1f}%" if pd.notna(total_var) else "N/D")

    fig = px.scatter(
        data,
        x="Team_PC1",
        y="Team_PC2",
        color="cluster",
        text="Team",
        labels={col: display_label(col) for col in data.columns},
        title=f"Mapa PCA de estilos de equipo | varianza explicada: {total_var:.1f}%",
        color_continuous_scale="Tealgrn",
        height=620,
    )
    fig = add_team_badges_to_pca_fig(fig, data, size=0.62)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

def profiles_tab(df: pd.DataFrame) -> None:
    st.header("Diagnóstico de perfiles")
    st.caption(
        "Esta pestaña sirve para revisar si los grupos y sus etiquetas futbolísticas tienen sentido. "
        "El mapa se recalcula sobre subimpactos normalizados, por lo que es una visualización diagnóstica."
    )
    help_box(
        "Cómo revisar los perfiles",
        """
        El agrupamiento agrupa jugadores parecidos dentro de un mismo rol. Esta pantalla ayuda a validar si las etiquetas
        asignadas tienen sentido futbolístico. El mapa PCA reduce los subimpactos a dos ejes para visualizar separaciones;
        la tabla de fortalezas muestra qué rasgos destacan en cada perfil frente a la media del rol.
        """,
    )
    if "cluster_label" not in df.columns or df["cluster_label"].dropna().empty:
        st.warning("No hay etiquetas de perfil calculadas todavía.")
        return

    role = st.selectbox(
        "Rol a revisar",
        sorted(df["role"].dropna().unique()),
        format_func=lambda r: ROLE_LABELS.get(r, r),
        help="El diagnóstico se hace por separado para cada rol porque no tiene sentido comparar perfiles de porteros, centrales o delanteros en el mismo espacio.",
    )
    role_df = df[df["role"].eq(role)].dropna(subset=["cluster_label"]).copy()
    cols = [col for col in cluster_analysis_columns(role) if col in role_df.columns and role_df[col].notna().any()]
    if len(role_df) < 3 or len(cols) < 2:
        st.warning("No hay suficientes datos para visualizar los perfiles de este rol.")
        return

    X = role_df[cols]
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=42)),
        ]
    )
    coords = pipe.fit_transform(X)
    plot_df = role_df[["Name", "Team", "role", "profile_cluster", "cluster_label", "Minutes", "impacto_global"]].copy()
    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]

    c1, c2 = st.columns([1.7, 1])
    with c1:
        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="cluster_label",
            symbol="profile_cluster",
            size="Minutes",
            hover_name="Name",
            labels={col: display_label(col) for col in plot_df.columns},
            hover_data=existing_columns(plot_df, ["Team", "profile_cluster", "impacto_global"]),
            title=f"Mapa diagnóstico de perfiles | {ROLE_LABELS.get(role, role)}",
        )
        st.plotly_chart(fig, width="stretch")
    with c2:
        counts = (
            role_df.groupby(["profile_cluster", "cluster_label"], as_index=False)
            .agg(jugadores=("Name", "count"), minutos_medios=("Minutes", "mean"), impacto_medio=("impacto_global", "mean"))
            .sort_values("jugadores", ascending=False)
        )
        counts["minutos_medios"] = counts["minutos_medios"].round(0).astype(int)
        counts["impacto_medio"] = counts["impacto_medio"].round(2)
        st.subheader("Tamaño de perfiles")
        display_dataframe(counts, width="stretch")

    st.subheader("Fortalezas medias por perfil")
    profile_means = role_df.groupby("cluster_label")[cols].mean().round(2)
    display_dataframe(profile_means.reset_index(), width="stretch")

    z = (profile_means - role_df[cols].mean()) / role_df[cols].std(ddof=0).replace(0, np.nan)
    z = z.fillna(0).round(2)
    strengths = []
    for label, row in z.iterrows():
        top = row.sort_values(ascending=False).head(4)
        low = row.sort_values(ascending=True).head(3)
        strengths.append(
            {
                "perfil": label,
                "fortalezas_relativas": ", ".join([f"{display_label(idx)} ({val:+.2f})" for idx, val in top.items()]),
                "debilidades_relativas": ", ".join([f"{display_label(idx)} ({val:+.2f})" for idx, val in low.items()]),
            }
        )
    st.subheader("Lectura para revisar etiquetas")
    display_dataframe(pd.DataFrame(strengths), width="stretch")

    z_display = z.rename(columns={col: display_label(col) for col in z.columns})
    fig = px.imshow(
        z_display,
        text_auto=True,
        color_continuous_scale="RdYlGn",
        zmin=-2,
        zmax=2,
        title="Perfil relativo de cada etiqueta frente a la media del rol",
    )
    st.plotly_chart(fig, width="stretch")


def player_select_label(name: str, lookup: dict[str, pd.Series]) -> str:
    row = lookup.get(name)
    if row is None:
        return name
    team = row.get("Team", "Equipo")
    role = ROLE_LABELS.get(row.get("role"), row.get("role", "Rol"))
    return f"{name} | {team} | {role}"


def ensure_compare_defaults(options: list[str]) -> None:
    if not options:
        return
    for idx, key in enumerate(COMPARE_PLAYER_KEYS):
        current = st.session_state.get(key)
        if current not in options:
            fallback_idx = min(idx, len(options) - 1)
            st.session_state[key] = options[fallback_idx]


def comparison_metric_columns_for_rows(rows_to_compare: list[pd.Series]) -> list[str]:
    base_cols = [
        "impacto_global",
        "impacto_ofensivo",
        "impacto_asociativo",
        "impacto_defensivo",
        "impacto_porteria",
        "impacto_finalizacion",
        "impacto_amenaza_area",
        "impacto_creacion",
        "impacto_progresion_pase",
        "impacto_progresion_conduccion",
        "impacto_circulacion",
        "impacto_duelos",
        "impacto_anticipacion",
        "impacto_defensa_area",
        "impacto_presion",
        "impacto_shot_stopping",
        "impacto_porteria_juego_aereo",
        "impacto_porteria_distribucion",
    ]
    cols = []
    for col in base_cols:
        if any(col in row.index and pd.notna(row.get(col)) for row in rows_to_compare):
            cols.append(col)
    return cols


def comparison_metric_columns(row_a: pd.Series, row_b: pd.Series) -> list[str]:
    return comparison_metric_columns_for_rows([row_a, row_b])


def build_comparison_rows_and_impacts(df: pd.DataFrame, selected_players: list[str]) -> tuple[list[pd.Series], list[str], list[dict[str, object]]]:
    rows_to_compare = [
        df[df["Name"].astype(str).eq(player)].sort_values("Minutes", ascending=False).iloc[0]
        for player in selected_players
    ]
    metric_cols = comparison_metric_columns_for_rows(rows_to_compare)
    rows = []
    for col in metric_cols:
        row_values = {"Metrica": display_label(col)}
        numeric_values = []
        for player, row in zip(selected_players, rows_to_compare):
            value = row.get(col)
            row_values[player] = np.nan if pd.isna(value) else round(float(value), 2)
            if pd.notna(value):
                numeric_values.append(float(value))
        row_values["Rango"] = np.nan if len(numeric_values) < 2 else round(max(numeric_values) - min(numeric_values), 2)
        rows.append(row_values)
    return rows_to_compare, metric_cols, rows


def render_comparison_summary(row: pd.Series, label: str) -> None:
    name = str(row.get("Name", "Jugador"))
    photo = player_photo_source(name, row_value(row, ["transfermarkt_photo_url", "Imagen"]))
    st.markdown(f"### {label}")
    if photo:
        st.image(photo, width=120)
    else:
        st.caption("Foto pendiente")
    st.markdown(f"**{name}**")
    st.caption(str(row.get("Team", "Equipo")))
    role = ROLE_LABELS.get(row.get("role"), row.get("role", "N/D"))
    profile = row.get("cluster_label", row.get("profile_cluster", "N/D"))
    minutes = row.get("Minutes", np.nan)
    minutes_text = f"{int(minutes):,}" if pd.notna(minutes) else "N/D"
    st.metric("Impacto global", f"{float(row.get('impacto_global')):.1f}" if pd.notna(row.get("impacto_global")) else "N/D")
    st.write(f"Rol: **{role}**")
    st.write(f"Perfil: **{profile}**")
    st.write(f"Minutos: **{minutes_text}**")
    if "market_value_million_eur" in row.index and pd.notna(row.get("market_value_million_eur")):
        st.write(f"Valor: **EUR {float(row.get('market_value_million_eur')):.1f}M**")


def comparison_tab(df: pd.DataFrame) -> None:
    st.header("Comparaciones de jugadores")
    help_box(
        "Como usar la comparativa",
        """
        Selecciona dos jugadores manualmente o mandalos desde la pestaña Encaje con el boton Comparar.
        La comparativa muestra identidad, impacto global, radar de subimpactos y una tabla de diferencias.
        """,
    )
    if df.empty:
        st.warning("No hay jugadores disponibles para comparar.")
        return

    options = sorted(df["Name"].dropna().astype(str).unique())
    ensure_compare_defaults(options)
    lookup = (
        df.sort_values("Minutes", ascending=False)
        .drop_duplicates("Name")
        .set_index("Name")
        .to_dict(orient="index")
    )
    lookup = {name: pd.Series(values) for name, values in lookup.items()}

    requested_count = max(2, min(5, int(st.session_state.get(COMPARE_REQUESTED_COUNT_KEY, 2))))
    current_visible_count = int(st.session_state.get(COMPARE_VISIBLE_COUNT_KEY, requested_count))
    if current_visible_count < requested_count or COMPARE_VISIBLE_COUNT_KEY not in st.session_state:
        st.session_state[COMPARE_VISIBLE_COUNT_KEY] = requested_count

    selected_count = st.slider(
        "Numero de jugadores a comparar",
        min_value=2,
        max_value=5,
        value=int(st.session_state.get(COMPARE_VISIBLE_COUNT_KEY, 2)),
        step=1,
        key=COMPARE_VISIBLE_COUNT_KEY,
        help="Puedes comparar desde 2 hasta 5 jugadores en la misma vista.",
    )

    selector_columns = st.columns(selected_count)
    selected_players = []
    for idx in range(selected_count):
        with selector_columns[idx]:
            selected_players.append(
                st.selectbox(
                    COMPARE_PLAYER_LABELS[idx],
                    options,
                    key=COMPARE_PLAYER_KEYS[idx],
                    format_func=lambda name: player_select_label(name, lookup),
                )
            )

    duplicated = pd.Series(selected_players).duplicated(keep=False)
    current_comparison_selection = tuple(selected_players)
    rows_to_compare, metric_cols, rows = build_comparison_rows_and_impacts(df, selected_players)

    summary_columns = st.columns(selected_count)
    for idx, (column, row) in enumerate(zip(summary_columns, rows_to_compare)):
        with column.container(border=True):
            render_comparison_summary(row, COMPARE_PLAYER_LABELS[idx])

    compare_action_cols = st.columns([1, 3])
    if compare_action_cols[0].button("Comparar", key="comparison_run_button", type="primary", width="stretch"):
        if duplicated.any():
            st.session_state.pop("comparison_confirmed_players", None)
        else:
            st.session_state["comparison_confirmed_players"] = current_comparison_selection
            save_comparison_entry(rows_to_compare, selected_players)

    if duplicated.any():
        st.warning("Hay jugadores repetidos en la comparativa. Cambia alguno de los selectores para comparar perfiles distintos.")
        return

    if st.session_state.get("comparison_confirmed_players") != current_comparison_selection:
        st.info("Selecciona los jugadores y pulsa Comparar para generar la comparativa.")
        return

    if metric_cols:
        st.subheader("Radar comparativo")
        radar_cols = [
            col
            for col in metric_cols
            if col != "impacto_global" and sum(pd.notna(row.get(col)) for row in rows_to_compare) >= 2
        ]
        if not radar_cols:
            radar_cols = ["impacto_global"]

        fig = go.Figure()
        colors = ["#2f6f73", "#c2410c", "#2563eb", "#7c3aed", "#ca8a04"]
        for name, row, color in zip(selected_players, rows_to_compare, colors):
            values = [float(row.get(col)) if pd.notna(row.get(col)) else 0.0 for col in radar_cols]
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=[display_label(col) for col in radar_cols] + [display_label(radar_cols[0])],
                    fill="toself",
                    name=name,
                    line_color=color,
                    opacity=0.78,
                )
            )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Comparacion de impactos",
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Tabla de impactos")
        render_laliga_comparison_table(rows, selected_players)

        bar_data = pd.DataFrame(rows).melt(
            id_vars="Metrica",
            value_vars=selected_players,
            var_name="Jugador",
            value_name="Impacto",
        )
        fig = px.bar(
            bar_data.dropna(subset=["Impacto"]),
            x="Metrica",
            y="Impacto",
            color="Jugador",
            barmode="group",
            range_y=[0, 100],
            title="Comparacion por dimensiones",
            color_discrete_sequence=colors,
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, width="stretch")

    identity_cols = existing_columns(
        df,
        [
            "Name",
            "Team",
            "role",
            "cluster_label",
            "Minutes",
            "age",
            "market_value_million_eur",
            "contract_until",
        ],
    )
    st.subheader("Datos generales")
    display_dataframe(pd.DataFrame([row[identity_cols] for row in rows_to_compare]), width="stretch")

    st.subheader("Informe de la comparacion")
    report_key = "comparison_report_" + "_".join(slugify_asset_name(player) for player in selected_players)
    if st.button("Generar informe PDF de la comparacion", key=f"{report_key}_generate", type="primary"):
        try:
            with st.spinner("Generando informe con estilo del dashboard..."):
                st.session_state[report_key] = comparison_report_pdf(df, rows_to_compare, rows if metric_cols else [], selected_players)
        except Exception as exc:
            st.error(f"No se pudo generar el informe: {exc}")
    if report_key in st.session_state:
        filename = f"informe_comparacion_{'_vs_'.join(slugify_asset_name(player) for player in selected_players)}.pdf"
        render_report_delivery(
            report_key,
            filename,
            st.session_state[report_key],
            "Informe de comparacion de jugadores",
            "Adjunto el informe completo de la comparacion solicitada.",
        )


def lists_tab(df: pd.DataFrame) -> None:
    st.header("Listas")
    help_box(
        "Qué se guarda en listas",
        """
        Las listas funcionan como un histórico local. Los informes de jugador se guardan cuando generas su PDF.
        Las comparativas se guardan cuando pulsas Comparar. Todo queda almacenado en data/external y se mantiene aunque cierres sesión.
        """,
    )
    player_list_tab, comparison_list_tab = st.tabs(["Listas jugadores", "Listas comparativas"])
    data = load_report_lists()
    player_lookup = player_row_lookup_by_name(df)

    with player_list_tab:
        st.subheader("Informes de jugadores guardados")
        player_reports = data.get("players", [])
        if not player_reports:
            st.info("Todavía no hay informes de jugadores guardados. Genera un PDF desde el apartado Jugador para añadirlo aquí.")
        for entry in player_reports:
            title = str(entry.get("player", "Jugador"))
            created = str(entry.get("created_at", ""))
            team = str(entry.get("team", ""))
            profile = str(entry.get("profile", ""))
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="saved-player-card">
                        {list_player_photo_from_lookup(title, player_lookup, entry.get("photo_url", ""))}
                        <div>
                            <h3>{html.escape(title)}</h3>
                            <p>{html.escape(team)} | {html.escape(profile)}</p>
                            <p>{html.escape(created)}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                cols = st.columns([1, 1, 1])
                report_path = BASE_DIR / str(entry.get("path", ""))
                if report_path.exists():
                    if cols[0].button("Ver PDF", key=f"view_player_report_{entry.get('id')}", width="stretch"):
                        st.session_state["open_saved_player_pdf"] = str(entry.get("id"))
                else:
                    cols[0].warning("PDF no encontrado")
                if cols[1].button("Ir a jugador", key=f"go_player_report_{entry.get('id')}", width="stretch"):
                    st.session_state["pending_nav_section"] = "Jugador"
                    st.session_state["pending_player_profile"] = title
                    st.rerun()
                if cols[2].button("Borrar", key=f"delete_player_report_{entry.get('id')}", width="stretch"):
                    delete_report_entry("players", str(entry.get("id")))
                    st.rerun()
                if st.session_state.get("open_saved_player_pdf") == str(entry.get("id")) and report_path.exists():
                    pdf_bytes = report_path.read_bytes()
                    render_pdf_preview(pdf_bytes)
                    st.download_button(
                        "Descargar PDF",
                        data=pdf_bytes,
                        file_name=str(entry.get("filename", report_path.name)),
                        mime="application/pdf",
                        key=f"saved_player_download_{entry.get('id')}",
                    )

    with comparison_list_tab:
        st.subheader("Comparativas guardadas")
        comparison_reports = data.get("comparisons", [])
        if not comparison_reports:
            st.info("Todavía no hay comparativas guardadas. Pulsa Comparar en el apartado Comparaciones para añadir una.")
        valid_players = set(df["Name"].dropna().astype(str)) if "Name" in df.columns else set()
        for entry in comparison_reports:
            players = [str(player) for player in entry.get("players", [])]
            created = str(entry.get("created_at", ""))
            with st.container(border=True):
                st.markdown(f"**{' vs '.join(players)}**")
                st.caption(created)
                if entry.get("teams"):
                    st.caption("Equipos: " + " | ".join(str(team) for team in entry.get("teams", [])))
                photo_urls = list(entry.get("photo_urls", []))
                photo_items = []
                for idx, player in enumerate(players[:5]):
                    photo_items.append(
                        '<div class="comparison-photo-item">'
                        f'{list_player_photo_from_lookup(player, player_lookup, photo_urls[idx] if idx < len(photo_urls) else "")}'
                        f'<strong>{html.escape(player)}</strong>'
                        '</div>'
                    )
                st.markdown(f'<div class="comparison-photo-strip">{"".join(photo_items)}</div>', unsafe_allow_html=True)
                cols = st.columns([1, 1, 1])
                can_load = len(players) >= 2 and all(player in valid_players for player in players[:5])
                if cols[0].button("Ver PDF", key=f"view_comparison_report_{entry.get('id')}", disabled=not can_load, width="stretch"):
                    report_id = str(entry.get("id"))
                    report_path = BASE_DIR / str(entry.get("path", ""))
                    if not report_path.exists():
                        try:
                            rows_to_compare, metric_cols, impact_rows = build_comparison_rows_and_impacts(df, players[:5])
                            filename = f"informe_comparacion_{'_vs_'.join(slugify_asset_name(player) for player in players[:5])}.pdf"
                            with st.spinner("Generando PDF de la comparativa..."):
                                pdf_bytes = comparison_report_pdf(df, rows_to_compare, impact_rows if metric_cols else [], players[:5])
                            update_comparison_entry_pdf(report_id, filename, pdf_bytes)
                            report_path = SAVED_REPORTS_DIR / f"{report_id}_{filename}"
                        except Exception as exc:
                            st.error(f"No se pudo generar el PDF: {exc}")
                    st.session_state["open_saved_comparison_pdf"] = report_id
                if cols[1].button("Volver a comparar", key=f"load_comparison_{entry.get('id')}", disabled=not can_load, width="stretch"):
                    count = max(2, min(5, len(players)))
                    st.session_state["pending_nav_section"] = "Comparaciones"
                    st.session_state["pending_comparison_players"] = players[:count]
                    st.session_state["pending_comparison_confirmed"] = True
                    st.rerun()
                if cols[2].button("Borrar", key=f"delete_comparison_report_{entry.get('id')}", width="stretch"):
                    delete_report_entry("comparisons", str(entry.get("id")))
                    st.rerun()
                refreshed = load_report_lists()
                refreshed_entry = next((item for item in refreshed.get("comparisons", []) if str(item.get("id")) == str(entry.get("id"))), entry)
                report_path = BASE_DIR / str(refreshed_entry.get("path", ""))
                if st.session_state.get("open_saved_comparison_pdf") == str(entry.get("id")) and report_path.exists():
                    pdf_bytes = report_path.read_bytes()
                    render_pdf_preview(pdf_bytes)
                    st.download_button(
                        "Descargar PDF",
                        data=pdf_bytes,
                        file_name=str(refreshed_entry.get("filename", report_path.name)),
                        mime="application/pdf",
                        key=f"saved_comparison_download_{entry.get('id')}",
                    )


def fit_tab(df: pd.DataFrame) -> None:
    st.header("Encaje jugador-club")
    help_box(
        "Cómo se calcula el encaje",
        """
        El encaje combina cuatro ideas: rendimiento individual del jugador, necesidad del equipo destino en ese perfil,
        similitud táctica entre el club actual y el club destino, y viabilidad de proyecto mediante valor de mercado y edad
        si esas opciones están activadas. La búsqueda se hace por perfil, no solo por posición, para recomendar jugadores
        que respondan a una necesidad futbolistica concreta.
        """,
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    target_team = c1.selectbox(
        "Equipo destino",
        sorted(df["Team"].dropna().unique()),
        help="Club para el que se quiere buscar un candidato.",
    )
    team_title_card(target_team)
    profile_df = df.dropna(subset=["cluster_label"]).copy() if "cluster_label" in df.columns else df.copy()
    selected_profile = c2.selectbox(
        "Perfil buscado",
        sorted(profile_df["cluster_label"].dropna().unique()),
        help="Perfil futbolístico que necesita el equipo, por ejemplo central dominador de área o extremo desequilibrante.",
    )
    role_values = profile_df.loc[profile_df["cluster_label"].eq(selected_profile), "role"].dropna().unique()
    role = role_values[0] if len(role_values) else None
    top_n = c3.slider(
        "Candidatos",
        5,
        50,
        20,
        help="Número de recomendaciones que se mostrarán en la tabla final.",
    )
    if role:
        st.caption(f"Rol asociado al perfil seleccionado: {ROLE_LABELS.get(role, role)}")

    include_economic = "market_value_eur" in df.columns and df["market_value_eur"].notna().any()
    max_market_value = None
    if include_economic:
        target_values = df.loc[df["Team"].eq(target_team), "market_value_eur"].dropna() / 1_000_000
        suggested_limit = float(target_values.quantile(0.75)) if len(target_values) >= 4 else float(target_values.median()) if len(target_values) else 10.0
        with st.expander("Ajustes económicos del encaje", expanded=True):
            use_economic = st.checkbox(
                "Incluir viabilidad económica en el cálculo de encaje",
                value=True,
                help="Si está activo, el modelo penaliza candidatos caros y descarta los que superan el límite definido.",
            )
            max_market_value = st.slider(
                "Presupuesto/valor objetivo del fichaje (€M)",
                min_value=0.5,
                max_value=float(max(df["market_value_eur"].dropna().max() / 1_000_000, 1.0)),
                value=max(0.5, round(suggested_limit, 1)),
                step=0.5,
                help="Penaliza jugadores cuyo valor supera este umbral. Además, se descartan automáticamente los que superan el 150% del presupuesto.",
            )
            discard_ratio = st.slider(
                "Descartar si valor >= x veces presupuesto",
                min_value=1.0,
                max_value=4.0,
                value=1.5,
                step=0.25,
                help="Con 1.5, un jugador queda descartado si su valor de mercado alcanza o supera el 150% del presupuesto indicado.",
            )
    else:
        use_economic = False
        discard_ratio = 1.5

    include_age = "age" in df.columns and df["age"].notna().any()
    if include_age:
        max_age_available = int(np.ceil(pd.to_numeric(df["age"], errors="coerce").dropna().max()))
        with st.expander("Ajustes de edad y proyecto", expanded=True):
            max_candidate_age = st.slider(
                "Edad máxima del candidato",
                min_value=16,
                max_value=max(16, max_age_available),
                value=max(16, max_age_available),
                step=1,
                help="Filtro duro: solo se recomendarán jugadores con esta edad o menos.",
            )
            use_age = st.checkbox(
                "Incluir edad en el cálculo de encaje",
                value=True,
                help="Si está activo, se favorecen perfiles con mayor recorrido de proyecto y se penalizan edades más altas.",
            )
            ideal_max_age = st.slider(
                "Edad ideal máxima",
                min_value=19,
                max_value=34,
                value=27,
                step=1,
                help="Hasta esta edad el jugador no recibe penalización por edad.",
            )
            hard_max_age = st.slider(
                "Edad con penalización máxima",
                min_value=ideal_max_age + 1,
                max_value=40,
                value=min(34, ideal_max_age + 7),
                step=1,
                help="A partir de esta edad, el encaje por edad se acerca a 0.",
            )
    else:
        use_age = False
        ideal_max_age = 27
        hard_max_age = 34
        max_candidate_age = None

    athletic_policy_active = normalize_name(target_team) == normalize_name("Athletic Club")
    fit = player_team_fit(
        df,
        target_team=target_team,
        role=role,
        cluster_label=selected_profile,
        top_n=top_n,
        include_economic=use_economic,
        max_market_value_million=max_market_value if use_economic else None,
        budget_million=max_market_value if use_economic else None,
        discard_if_value_budget_ratio=discard_ratio,
        include_age=use_age,
        ideal_max_age=ideal_max_age,
        hard_max_age=hard_max_age,
        max_candidate_age=max_candidate_age,
        require_athletic_eligible=athletic_policy_active,
    )
    if athletic_policy_active:
        st.caption(
            "Política Athletic activa: las recomendaciones se limitan a jugadores marcados como elegibles "
            "en la lista externa de nacidos en País Vasco o Navarra."
        )
    if use_economic:
        st.caption(
            f"Regla económica activa: se penaliza por encima de €{max_market_value:.1f}M "
            f"y se descarta automáticamente si el valor es >= {discard_ratio:.2f}x ese presupuesto."
        )
    if use_age:
        st.caption(
            f"Regla de edad activa: sin penalización hasta {ideal_max_age} años; "
            f"penalización máxima cerca de {hard_max_age} años."
        )
    if max_candidate_age is not None:
        st.caption(f"Filtro de edad activo: solo se recomiendan jugadores de {max_candidate_age} años o menos.")
    render_top_fit_recommendations(fit)
    render_fit_breakdown_radar(fit)
    render_fit_recommendations_table(fit, show_athletic_origin=athletic_policy_active)


def missing_data_screen() -> None:
    st.error("No encuentro datos procesados y tampoco hay credenciales configuradas en Streamlit secrets.")
    st.markdown(
        """
        Para verlo en local:

        ```python
        player_stats, team_stats, competition_info = load_statsbomb_laliga()
        master_df = build_master_dataset(player_stats, team_stats)
        pca_results = fit_all_role_pcas(master_df)
        players_modeled, cluster_results = fit_all_clusters(master_df, pca_results)
        players_scored = calculate_impact_scores(players_modeled)
        players_scored.to_parquet(PROCESSED_DIR / "players_scored.parquet", index=False)
        ```

        Para desplegar en Streamlit Cloud, añade `SB_USERNAME` y `SB_PASSWORD` como secrets.
        """
    )


NAV_SECTIONS = ["Inicio", "Clasificaciones", "Jugador", "Equipos", "Encaje", "Comparaciones", "Listas"]


def apply_pending_navigation(df: pd.DataFrame) -> None:
    pending_section = st.session_state.pop("pending_nav_section", None)
    if pending_section in NAV_SECTIONS:
        st.session_state["main_section"] = pending_section

    pending_player = st.session_state.pop("pending_player_profile", None)
    if pending_player is not None and "Name" in df.columns:
        player_names = set(df["Name"].dropna().astype(str))
        if str(pending_player) in player_names:
            st.session_state["player_profile_selected"] = str(pending_player)

    pending_players = st.session_state.pop("pending_comparison_players", None)
    if pending_players:
        players = [str(player) for player in pending_players if str(player) in set(df["Name"].dropna().astype(str))]
        if len(players) >= 2:
            count = max(2, min(5, len(players)))
            selected = players[:count]
            st.session_state[COMPARE_VISIBLE_COUNT_KEY] = count
            st.session_state[COMPARE_REQUESTED_COUNT_KEY] = count
            for idx, player in enumerate(selected):
                st.session_state[COMPARE_PLAYER_KEYS[idx]] = player
            if st.session_state.pop("pending_comparison_confirmed", False):
                st.session_state["comparison_confirmed_players"] = tuple(selected)


def main() -> None:
    inject_css()
    if not login_screen():
        return

    df = get_data(processed_data_version())
    if df is None:
        missing_data_screen()
        return

    if "impacto_global" not in df.columns:
        df = calculate_impact_scores(df)

    apply_pending_navigation(df)

    with st.sidebar:
        st.markdown(laliga_logo_html("sidebar-logo"), unsafe_allow_html=True)
        st.title("TFG StatsBomb")
        st.caption("Impacto, perfiles y encaje")
        if st.session_state.get("auth_user"):
            st.caption(f"Sesión: {st.session_state['auth_user']}")
        if st.button("Cerrar sesión"):
            st.session_state["authenticated"] = False
            st.session_state.pop("auth_user", None)
            st.rerun()

    filtered = filter_players(df)
    if st.session_state.get("main_section") not in NAV_SECTIONS:
        st.session_state["main_section"] = NAV_SECTIONS[0]
    selected_section = st.radio(
        "Apartado",
        NAV_SECTIONS,
        key="main_section",
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_section == "Inicio":
        overview_tab(filtered)
    elif selected_section == "Clasificaciones":
        rankings_tab(filtered)
    elif selected_section == "Jugador":
        player_profile_tab(filtered)
    elif selected_section == "Equipos":
        teams_tab(df)
    elif selected_section == "Encaje":
        fit_tab(df)
    elif selected_section == "Comparaciones":
        comparison_tab(df)
    elif selected_section == "Listas":
        lists_tab(df)

if __name__ == "__main__":
    main()
