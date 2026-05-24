from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import re
import unicodedata
from difflib import SequenceMatcher

# Evita avisos/fugas conocidas de KMeans con MKL en Windows cuando hay pocos datos.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from statsbombpy import sb
except ImportError:  # pragma: no cover
    sb = None


RANDOM_STATE = 42
MIN_MINUTES = 600
COMPETITION_NAME = "La Liga"
SEASON_NAME = "2024/2025"

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
for directory in (RAW_DIR, EXTERNAL_DIR, PROCESSED_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_column_mapping(path: str | Path = "statsbomb_column_mapping.json") -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


POSITION_TO_ROLE = {
    "Goalkeeper": "POR",
    "Left Centre Back": "DEF",
    "Right Centre Back": "DEF",
    "Centre Back": "DEF",
    "Left Back": "LAT",
    "Right Back": "LAT",
    "Left Wing Back": "LAT",
    "Right Wing Back": "LAT",
    "Left Defensive Midfielder": "MED",
    "Right Defensive Midfielder": "MED",
    "Centre Defensive Midfielder": "MED",
    "Right Centre Midfielder": "MED",
    "Left Centre Midfielder": "MED",
    "Centre Midfielder": "MED",
    "Left Midfielder": "MED",
    "Right Midfielder": "MED",
    "Centre Attacking Midfielder": "MED",
    "Right Wing": "EXT",
    "Left Wing": "EXT",
    "Centre Forward": "DEL",
    "Right Centre Forward": "DEL",
    "Left Centre Forward": "DEL",
}

CLUSTER_LABEL_RULES = {
    "POR": {
        "Portero parador": {"impacto_shot_stopping": 0.85, "impacto_porteria_juego_aereo": 0.15},
        "Portero constructor": {"impacto_porteria_distribucion": 1.0},
    },
    "DEF": {
        "Central constructor": {"impacto_progresion_pase": 0.45, "impacto_circulacion": 0.35, "impacto_asociativo": 0.20},
        "Central dominador de area": {"impacto_defensa_area": 0.55, "impacto_duelos": 0.30, "impacto_anticipacion": 0.15},
        "Central corrector": {"impacto_anticipacion": 0.45, "impacto_duelos": 0.30, "impacto_presion": 0.25},
    },
    "LAT": {
        "Lateral profundo": {"impacto_progresion_pase": 0.30, "impacto_progresion_conduccion": 0.30, "impacto_creacion": 0.25, "impacto_amenaza_area": 0.15},
        "Lateral defensivo": {"impacto_duelos": 0.35, "impacto_anticipacion": 0.30, "impacto_presion": 0.20, "impacto_defensa_area": 0.15},
    },
    "MED": {
        "Mediocentro defensivo": {"impacto_duelos": 0.25, "impacto_anticipacion": 0.30, "impacto_presion": 0.25, "impacto_circulacion": 0.20},
        "Interior llegador": {"impacto_finalizacion": 0.40, "impacto_amenaza_area": 0.30, "impacto_creacion": 0.20, "impacto_progresion_conduccion": 0.10},
    },
    "EXT": {
        "Extremo con tendencia hacia dentro": {"impacto_finalizacion": 0.35, "impacto_amenaza_area": 0.30, "impacto_progresion_conduccion": 0.25, "impacto_creacion": 0.10},
        "Extremo abierto": {"impacto_creacion": 0.35, "impacto_progresion_pase": 0.30, "impacto_circulacion": 0.20, "impacto_amenaza_area": 0.15},
    },
    "DEL": {
        "Delantero finalizador": {"impacto_finalizacion": 0.55, "impacto_amenaza_area": 0.30, "impacto_creacion": 0.15},
        "Delantero asociativo": {"impacto_creacion": 0.40, "impacto_progresion_pase": 0.25, "impacto_circulacion": 0.20, "impacto_amenaza_area": 0.15},
    },
}

ROLE_CLUSTER_K = {
    "POR": 2,
    "DEF": 3,
    "LAT": 2,
    "MED": 2,
    "EXT": 2,
    "DEL": 2,
}

ID_COLUMNS = {
    "account_id",
    "player_id",
    "team_id",
    "competition_id",
    "season_id",
    "country_id",
    "Recent Match",
}
VOLUME_COLUMNS = {"Minutes", "Appearances", "Starting Appearances", "Average Minutes", "Minutes 360"}
NEGATIVE_METRICS = {
    "Dribbled Past",
    "Dispossessed",
    "Turnovers",
    "Errors",
    "Positioning Error",
    "Penalties Conceded",
    "Fouls",
    "Yellow Cards",
    "Red Cards",
    "Failed Dribbles",
    "Goals Conceded",
    "Penalties Conceded",
    "Pass into Pressure%",
    "Pass into Danger%",
}

def _unique_metrics(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(metric for group in groups for metric in group))


# Taxonomía futbolística fina. Los modelos pueden usar subgrupos concretos y,
# a la vez, mantener impactos agregados fáciles de comunicar en el dashboard.
METRIC_SUBGROUPS = {
    # OFENSIVO
    "finalizacion": [
        "xG",
        "NP Goals",
        "All Goals",
        "Shots",
        "xG/Shot",
        "PSxG",
        "Shooting%",
        "Goal Conversion%",
        "Scoring Contribution",
        "xG & xG Assisted",
        "Shots & Key Passes",
        "Shot OBV",
        "O/U Performance",
    ],
    "amenaza_area": [
        "Touches In Box",
        "Shot Touch%",
        "Penalty Wins",
        "Deep Completions",
        "PinTin",
        "PintoB",
        "OP Passes Into Box",
        "Passes Inside Box",
        "Average SRI F3",
        "BRIS F3 2m%",
        "BRIS F3 5m%",
        "BRIS F3 10m%",
        "LBP Received F3",
        "LBP Received 5m in F3",
        "LBP Received 10m in F3",
    ],

    # ASOCIATIVO
    "creacion": [
        "xG Assisted",
        "OP xG Assisted",
        "Key Passes",
        "OP Key Passes",
        "Assists",
        "OP Assists",
        "SP xG Assisted",
        "SP Assists",
        "SP Key Passes",
        "Throughballs",
        "PintoB",
        "OP Passes Into Box",
        "Passes Inside Box",
        "PinTin",
        "Successful Crosses",
        "Crossing%",
        "Successful Box Cross%",
        "Pass into Danger%",
        "Pass OBV",
    ],
    "progresion_pase": [
        "xGChain",
        "OP xGChain",
        "xGBuildup",
        "OP xGBuildup",
        "Pass Forward%",
        "Deep Progressions",
        "Deep Completions",
        "OP F3 Passes",
        "F3 Pass Forward%",
        "LBP",
        "LBP Completed",
        "LBP Completed %",
        "LBP F2",
        "LBP F3",
        "LBP OBV",
        "LBP OBV F2",
        "LBP OBV F3",
        "LBP/Pass%",
        "LBP/Pass F2%",
        "LBP/Pass F3%",
        "Pass OBV",
        "Average Pass X",
    ],
    "progresion_conduccion": [
        "Successful Dribbles",
        "Dribbles",
        "Dribble%",
        "Failed Dribbles",
        "Carries",
        "Carry%",
        "Carry Length",
        "Turnovers",
        "Dispossessed",
        "D&C OBV",
    ],
    "circulacion": [
        "OP Passes",
        "Passing%",
        "Pass Length",
        "Succ. Pass Length",
        "Passes Pressured%",
        "Pr. Pass%",
        "Pr. Pass% Dif.",
        "Pr. Pass Length%",
        "Pr. Pass Length Dif.",
        "Long Ball%",
        "Long Balls",
        "Pass Backward%",
        "Pass Sideways%",
        "Pass Forward%",
        "Turnovers",
        "Dispossessed",
    ],

    # DEFENSIVO
    "duelos": [
        "Tackles",
        "PAdj Tackles",
        "Tack/DP%",
        "Aerial Win%",
        "Aerial Wins",
        "Dribbles Stopped%",
        "Dribbled Past",
        "Fouls",
    ],
    "anticipacion": [
        "Interceptions",
        "Tack&Int",
        "PAdj Interceptions",
        "PAdj Tack&Int",
        "Ball Recoveries",
        "Ball Recov. F2",
        "Defensive Regains",
        "Counterpress Regains",
        "Pressure Regains",
    ],
    "defensa_area": [
        "Blocks/Shot",
        "Clearances",
        "PAdj Clearances",
        "Defensive Actions",
        "DA OBV",
        "Errors",
        "Penalties Conceded",
        "Yellow Cards",
        "Red Cards",
    ],
    "presion": [
        "Pressures",
        "PAdj Pressures",
        "Pressures F2",
        "Pressures F2%",
        "Counterpressures",
        "Counterpressures F2",
        "Counterpressures F2%",
        "Aggressive Actions",
        "Average Pressure X",
        "Average DA X",
        "Pressure Regains",
        "Counterpress Regains",
    ],

    # PORTERIA
    "shot_stopping": [
        "GSAA",
        "Shot Stopping%",
        "xSv%",
        "Save%",
        "PSxG Faced",
        "NPOT PSxG Faced",
        "Shots Faced",
        "All Shots Faced",
        "Shots Faced OT%",
        "Goals Conceded",
    ],
    "porteria_juego_aereo": [
        "Claims%",
        "GK Aggressive Dist.",
        "Penalties Faced",
        "Penalties Conceded",
    ],
    "porteria_distribucion": [
        "Goalkeeper OBV",
        "Positive Outcome%",
        "Positive Outcome",
        "UPr. Long Balls",
        "Pr. Long Balls",
        "Pass Length",
        "Pass into Pressure%",
        "Pass into Danger%",
        "Passing%",
        "Long Ball%",
        "Long Balls",
    ],
}

METRIC_GROUPS = {
    **METRIC_SUBGROUPS,
    "ofensivo": _unique_metrics(
        METRIC_SUBGROUPS["finalizacion"],
        METRIC_SUBGROUPS["amenaza_area"],
    ),
    "asociativo": _unique_metrics(
        METRIC_SUBGROUPS["creacion"],
        METRIC_SUBGROUPS["progresion_pase"],
        METRIC_SUBGROUPS["progresion_conduccion"],
        METRIC_SUBGROUPS["circulacion"],
    ),
    "defensivo": _unique_metrics(
        METRIC_SUBGROUPS["duelos"],
        METRIC_SUBGROUPS["anticipacion"],
        METRIC_SUBGROUPS["defensa_area"],
        METRIC_SUBGROUPS["presion"],
    ),
    "porteria": _unique_metrics(
        METRIC_SUBGROUPS["shot_stopping"],
        METRIC_SUBGROUPS["porteria_juego_aereo"],
        METRIC_SUBGROUPS["porteria_distribucion"],
    ),
}

TEAM_DIRECT_STYLE_FEATURES = [
    # Posesion / circulacion
    "team_season_possession",
    "team_season_passing_ratio",
    "team_season_passes_pg",
    "team_season_op_passes_pg",

    # Verticalidad / ritmo
    "team_season_directness",
    "team_season_pace_towards_goal",
    "team_season_gk_pass_distance",
    "team_season_gk_long_pass_ratio",

    # Progresion
    "team_season_deep_progressions_pg",
    "team_season_deep_completions_pg",
    "team_season_passes_inside_box_pg",

    # Juego exterior
    "team_season_crosses_into_box_pg",
    "team_season_successful_crosses_into_box_pg",
    "team_season_successful_box_cross_ratio",

    # Dribbling / conduccion
    "team_season_completed_dribbles_pg",
    "team_season_dribble_ratio",
    "team_season_total_dribbles_pg",

    # Presion
    "team_season_pressures_pg",
    "team_season_counterpressures_pg",
    "team_season_pressure_regains_pg",
    "team_season_counterpressure_regains_pg",
    "team_season_fhalf_pressures_pg",
    "team_season_fhalf_counterpressures_pg",
    "team_season_fhalf_pressures_ratio",
    "team_season_fhalf_counterpressures_ratio",
    "team_season_ppda",
    "team_season_aggressive_actions_pg",
    "team_season_aggression",

    # Altura defensiva
    "team_season_defensive_distance",
    "team_season_defensive_distance_ppda",

    # Transiciones
    "team_season_counter_attacking_shots_pg",
    "team_season_high_press_shots_pg",

    # Perfil ofensivo
    "team_season_np_shot_distance",
    "team_season_shots_in_clear_pg",

    # Contexto del partido
    "team_season_ball_in_play_time",
]

ROLE_AGGREGATED_IMPACT_WEIGHTS = {
    "POR": {
        "porteria": {"shot_stopping": 0.55, "porteria_juego_aereo": 0.15, "porteria_distribucion": 0.30},
    },
    "DEF": {
        "ofensivo": {"finalizacion": 0.30, "amenaza_area": 0.70},
        "asociativo": {"creacion": 0.2, "progresion_pase": 0.4, "progresion_conduccion": 0.05, "circulacion": 0.35},
        "defensivo": {"duelos": 0.30, "anticipacion": 0.25, "defensa_area": 0.4, "presion": 0.05},
    },
    "LAT": {
        "ofensivo": {"finalizacion": 0.25, "amenaza_area": 0.75},
        "asociativo": {"creacion": 0.25, "progresion_pase": 0.30, "progresion_conduccion": 0.30, "circulacion": 0.15},
        "defensivo": {"duelos": 0.20, "anticipacion": 0.30, "defensa_area": 0.20, "presion": 0.30},
    },
    "MED": {
        "ofensivo": {"finalizacion": 0.45, "amenaza_area": 0.55},
        "asociativo": {"creacion": 0.25, "progresion_pase": 0.30, "progresion_conduccion": 0.20, "circulacion": 0.25},
        "defensivo": {"duelos": 0.25, "anticipacion": 0.25, "defensa_area": 0.10, "presion": 0.40},
    },
    "EXT": {
        "ofensivo": {"finalizacion": 0.6, "amenaza_area": 0.4},
        "asociativo": {"creacion": 0.35, "progresion_pase": 0.15, "progresion_conduccion": 0.40, "circulacion": 0.10},
        "defensivo": {"duelos": 0.10, "anticipacion": 0.15, "defensa_area": 0.05, "presion": 0.70},
    },
    "DEL": {
        "ofensivo": {"finalizacion": 0.70, "amenaza_area": 0.30},
        "asociativo": {"creacion": 0.45, "progresion_pase": 0.20, "progresion_conduccion": 0.25, "circulacion": 0.10},
        "defensivo": {"duelos": 0.15, "anticipacion": 0.10, "defensa_area": 0.05, "presion": 0.70},
    },
}

ROLE_GLOBAL_IMPACT_WEIGHTS = {
    "POR": {"impacto_porteria": 1.00},
    "DEF": {"impacto_defensivo": 0.50, "impacto_asociativo": 0.35, "impacto_ofensivo": 0.15},
    "LAT": {"impacto_defensivo": 0.30, "impacto_asociativo": 0.40, "impacto_ofensivo": 0.30},
    "MED": {"impacto_defensivo": 0.25, "impacto_asociativo": 0.50, "impacto_ofensivo": 0.25},
    "EXT": {"impacto_defensivo": 0.10, "impacto_asociativo": 0.40, "impacto_ofensivo": 0.50},
    "DEL": {"impacto_defensivo": 0.05, "impacto_asociativo": 0.25, "impacto_ofensivo": 0.70},
}

PROFILE_GLOBAL_IMPACT_WEIGHTS = {
    "Central dominador de area": {"impacto_defensivo": 0.60, "impacto_asociativo": 0.30, "impacto_ofensivo": 0.10},
    "Central constructor": {"impacto_defensivo": 0.45, "impacto_asociativo": 0.45, "impacto_ofensivo": 0.10},
    "Central corrector": {"impacto_defensivo": 0.55, "impacto_asociativo": 0.30, "impacto_ofensivo": 0.15},
    "Lateral defensivo": {"impacto_defensivo": 0.50, "impacto_asociativo": 0.35, "impacto_ofensivo": 0.15},
    "Lateral profundo": {"impacto_defensivo": 0.35, "impacto_asociativo": 0.30, "impacto_ofensivo": 0.35},
    "Mediocentro defensivo": {"impacto_defensivo": 0.40, "impacto_asociativo": 0.45, "impacto_ofensivo": 0.15},
    "Interior llegador": {"impacto_defensivo": 0.15, "impacto_asociativo": 0.50, "impacto_ofensivo": 0.35},
    "Extremo con tendencia hacia dentro": {"impacto_defensivo": 0.08, "impacto_asociativo": 0.32, "impacto_ofensivo": 0.60},
    "Extremo abierto": {"impacto_defensivo": 0.10, "impacto_asociativo": 0.50, "impacto_ofensivo": 0.40},
    "Delantero finalizador": {"impacto_defensivo": 0.03, "impacto_asociativo": 0.17, "impacto_ofensivo": 0.80},
    "Delantero asociativo": {"impacto_defensivo": 0.05, "impacto_asociativo": 0.40, "impacto_ofensivo": 0.55},
    "Portero parador": {"impacto_porteria": 1.00},
    "Portero constructor": {"impacto_porteria": 1.00},
}

ROLE_GROUPS = {
    "POR": ["shot_stopping", "porteria_juego_aereo", "porteria_distribucion"],
    "DEF": ["duelos", "anticipacion", "defensa_area", "presion", "progresion_pase", "circulacion"],
    "LAT": ["duelos", "anticipacion", "presion", "creacion", "progresion_pase", "progresion_conduccion", "circulacion"],
    "MED": [
        "duelos",
        "anticipacion",
        "presion",
        "creacion",
        "progresion_pase",
        "progresion_conduccion",
        "circulacion",
        "finalizacion",
        "amenaza_area",
    ],
    "EXT": ["finalizacion", "amenaza_area", "creacion", "progresion_conduccion", "progresion_pase"],
    "DEL": ["finalizacion", "amenaza_area", "creacion", "progresion_conduccion"],
}

ROLE_NEED_IMPACT_COLUMNS = {
    role: [f"impacto_{group}" for group in groups]
    for role, groups in ROLE_GROUPS.items()
}


@dataclass
class RolePCAResult:
    role: str
    features: list[str]
    pipeline: Pipeline
    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: pd.Series


@dataclass
class RoleClusterResult:
    role: str
    k: int
    labels: pd.Series
    silhouette_table: pd.DataFrame
    cluster_profiles_z: pd.DataFrame
    feature_importance: pd.Series
    model: KMeans


@dataclass
class TeamSimilarityResult:
    team_profiles: pd.DataFrame
    scaled_profiles: pd.DataFrame
    cosine_matrix: pd.DataFrame
    pca_map: pd.DataFrame
    pca_model: PCA


def get_statsbomb_credentials() -> dict[str, str]:
    user = os.getenv("SB_USERNAME")
    password = os.getenv("SB_PASSWORD")
    if not user or not password:
        raise RuntimeError(
            "Faltan SB_USERNAME o SB_PASSWORD. Define las variables de entorno antes de descargar datos."
        )
    return {"user": user, "passwd": password}


def _read_or_download(path: Path, download_fn, force_download: bool = False) -> pd.DataFrame:
    if path.exists() and not force_download:
        return pd.read_parquet(path)
    df = download_fn()
    df.to_parquet(path, index=False)
    return df


def load_statsbomb_laliga(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if sb is None:
        raise ImportError("statsbombpy no está instalado. Ejecuta `%pip install statsbombpy`.")

    creds = get_statsbomb_credentials()
    competitions = _read_or_download(
        RAW_DIR / "competitions.parquet",
        lambda: sb.competitions(creds=creds),
        force_download=force_download,
    )
    mask = competitions["competition_name"].eq(COMPETITION_NAME) & competitions["season_name"].eq(SEASON_NAME)
    if not mask.any():
        raise ValueError(f"No encuentro {COMPETITION_NAME} {SEASON_NAME} en competitions.")

    info = competitions.loc[mask].iloc[0]
    comp_id, season_id = int(info["competition_id"]), int(info["season_id"])
    safe_name = f"{COMPETITION_NAME}_{SEASON_NAME}".replace("/", "-").replace(" ", "_")
    players = _read_or_download(
        RAW_DIR / f"player_stats_{safe_name}.parquet",
        lambda: sb.player_season_stats(competition_id=comp_id, season_id=season_id, creds=creds),
        force_download=force_download,
    )
    teams = _read_or_download(
        RAW_DIR / f"team_stats_{safe_name}.parquet",
        lambda: sb.team_season_stats(competition_id=comp_id, season_id=season_id, creds=creds),
        force_download=force_download,
    )
    return players, teams, info


def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()


def existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def build_master_dataset(player_stats: pd.DataFrame, team_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    mapping = load_column_mapping()
    players = deduplicate_columns(player_stats.rename(columns=mapping))

    if team_stats is not None and not team_stats.empty and "team_id" in players.columns:
        teams = deduplicate_columns(team_stats.rename(columns=mapping))
        team_cols = ["team_id"] + [col for col in teams.columns if col.startswith("team_season_")]
        teams = teams[existing_columns(teams, team_cols)]
        players = players.merge(teams, on="team_id", how="left", suffixes=("", "_team"))

    players = deduplicate_columns(players)
    players["original_role"] = players["Primary Position"].map(POSITION_TO_ROLE)
    players = players.dropna(subset=["Name", "Team", "original_role"]).copy()
    players["model_role"] = players["original_role"]
    players["role"] = players["model_role"]
    if "Minutes" in players.columns:
        players = players.loc[players["Minutes"].fillna(0) >= MIN_MINUTES].copy()

    numeric_cols = players.select_dtypes(include=np.number).columns
    players[numeric_cols] = players[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return players.reset_index(drop=True)


def get_model_feature_columns(
    df: pd.DataFrame,
    groups: Iterable[str] | None = None,
    exclude_team_context: bool = True,
) -> list[str]:
    if groups is None:
        candidates = df.select_dtypes(include=np.number).columns.tolist()
    else:
        candidates = []
        for group in groups:
            candidates.extend(METRIC_GROUPS.get(group, []))
        candidates = list(dict.fromkeys(candidates))

    features = []
    for col in candidates:
        if col not in df.columns or col in ID_COLUMNS or col in VOLUME_COLUMNS:
            continue
        if exclude_team_context and col.startswith("team_season_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) > 1:
            features.append(col)
    return features


def winsorize_frame(df: pd.DataFrame, cols: list[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        lo, hi = out[col].quantile([lower, upper])
        out[col] = out[col].clip(lo, hi)
    return out


def resumen_dataset(df: pd.DataFrame) -> dict[str, object]:
    return {
        "jugadores": len(df),
        "equipos": df["Team"].nunique() if "Team" in df.columns else None,
        "variables": df.shape[1],
        "roles_modelado": df["model_role"].value_counts().to_dict() if "model_role" in df.columns else df["role"].value_counts().to_dict(),
        "minutos": df["Minutes"].describe().to_dict() if "Minutes" in df.columns else None,
    }


def detectar_correlaciones_fuertes(df: pd.DataFrame, cols: list[str], umbral: float = 0.90) -> pd.DataFrame:
    corr = df[cols].fillna(0).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().reset_index()
    pairs.columns = ["variable_1", "variable_2", "correlacion_abs"]
    return pairs.loc[pairs["correlacion_abs"] >= umbral].sort_values(
        "correlacion_abs", ascending=False
    ).reset_index(drop=True)


def fit_role_pca(df: pd.DataFrame, role: str, variance: float = 0.80, max_components: int = 8) -> RolePCAResult | None:
    role_col = "model_role" if "model_role" in df.columns else "role"
    role_df = df.loc[df[role_col].eq(role)].copy()
    features = get_model_feature_columns(role_df, groups=ROLE_GROUPS.get(role), exclude_team_context=True)
    if len(role_df) < 8 or len(features) < 3:
        return None

    X = winsorize_frame(role_df, features)[features]
    max_allowed = min(max_components, len(features), len(role_df) - 1)
    probe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=max_allowed, random_state=RANDOM_STATE)),
        ]
    )
    probe.fit(X)
    cumulative = np.cumsum(probe.named_steps["pca"].explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance) + 1)
    n_components = min(max(2, n_components), max_allowed)

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ]
    )
    transformed = pipe.fit_transform(X)
    pc_cols = [f"{role}_PC{i + 1}" for i in range(n_components)]
    scores = pd.DataFrame(transformed, columns=pc_cols, index=role_df.index)
    scores[["Name", "Team", "role", "Minutes"]] = role_df[["Name", "Team", "role", "Minutes"]]
    if "model_role" in role_df.columns:
        scores["model_role"] = role_df["model_role"]

    pca = pipe.named_steps["pca"]
    loadings = pd.DataFrame(pca.components_.T, index=features, columns=pc_cols)
    loadings = loadings.sort_values(pc_cols[0], key=lambda s: s.abs(), ascending=False)
    explained = pd.Series(pca.explained_variance_ratio_, index=pc_cols, name="explained_variance")
    return RolePCAResult(role, features, pipe, scores, loadings, explained)


def fit_all_role_pcas(df: pd.DataFrame) -> dict[str, RolePCAResult]:
    role_col = "model_role" if "model_role" in df.columns else "role"
    return {
        role: result
        for role in sorted(df[role_col].dropna().unique())
        if (result := fit_role_pca(df, role)) is not None
    }


def choose_k_by_silhouette(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 6,
    min_cluster_size: int | None = None,
) -> tuple[int, pd.DataFrame]:
    rows = []
    min_size = min_cluster_size or max(3, int(np.ceil(X.shape[0] * 0.07)))
    for k in range(k_min, min(k_max, X.shape[0] - 1) + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30)
        labels = model.fit_predict(X)
        cluster_sizes = pd.Series(labels).value_counts()
        valid_size = bool(cluster_sizes.min() >= min_size)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(X, labels),
                "inertia": model.inertia_,
                "min_cluster_size": int(cluster_sizes.min()),
                "valid_size": valid_size,
            }
        )
    table = pd.DataFrame(rows)
    candidates = table.loc[table["valid_size"]].copy()
    if candidates.empty:
        candidates = table.copy()
    best_k = int(candidates.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    return best_k, table


def fit_role_clusters(df: pd.DataFrame, pca_result: RolePCAResult, k_max: int = 6) -> RoleClusterResult | None:
    role = pca_result.role
    role_col = "model_role" if "model_role" in df.columns else "role"
    pc_cols = [col for col in pca_result.scores.columns if col.startswith(f"{role}_PC")]
    X = pca_result.scores[pc_cols].to_numpy()
    if X.shape[0] < 10:
        return None

    suggested_k, silhouette_table = choose_k_by_silhouette(X, k_max=k_max)
    fixed_k = ROLE_CLUSTER_K.get(role)
    k = fixed_k if fixed_k is not None and 2 <= fixed_k < X.shape[0] else suggested_k
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30)
    labels = pd.Series(model.fit_predict(X), index=pca_result.scores.index, name="cluster")

    role_df = df.loc[df[role_col].eq(role)].copy()
    features = pca_result.features
    X_raw = winsorize_frame(role_df, features)[features]
    X_raw = X_raw.fillna(X_raw.median(numeric_only=True)).fillna(0)
    profiles = X_raw.assign(cluster=labels).groupby("cluster").mean()
    profiles_z = (profiles - X_raw.mean()) / X_raw.std(ddof=0).replace(0, np.nan)

    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, class_weight="balanced")
    rf.fit(X_raw, labels.loc[X_raw.index])
    importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    return RoleClusterResult(role, k, labels, silhouette_table, profiles_z, importance, model)


def fit_all_clusters(df: pd.DataFrame, pca_results: dict[str, RolePCAResult]) -> tuple[pd.DataFrame, dict[str, RoleClusterResult]]:
    out = df.copy()
    out["profile_cluster"] = pd.NA
    results = {}
    for role, pca_result in pca_results.items():
        cluster_result = fit_role_clusters(out, pca_result)
        if cluster_result is None:
            continue
        results[role] = cluster_result
        out.loc[cluster_result.labels.index, "profile_cluster"] = (
            role + "_" + cluster_result.labels.astype(str)
        )
    return out, results


def percentile_by_role(df: pd.DataFrame, columns: list[str], role_col: str = "role") -> pd.DataFrame:
    pct = pd.DataFrame(index=df.index)
    for col in columns:
        values = df.groupby(role_col)[col].rank(pct=True) * 100
        if col in NEGATIVE_METRICS:
            values = 100 - values
        pct[col] = values
    return pct.fillna(50)


def ensure_model_role(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "role" in out.columns and "model_role" not in out.columns:
        out["model_role"] = out["role"]
    if "model_role" in out.columns:
        out["role"] = out["model_role"]
    return out


def assign_cluster_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna nombres futbolisticos interpretables a clusters tecnicos.

    La etiqueta se decide comparando el perfil medio del cluster con plantillas
    de subimpactos. El codigo del cluster se mantiene para trazabilidad.
    """
    out = ensure_model_role(df)
    if "profile_cluster" not in out.columns:
        return out

    impact_cols = [col for col in out.columns if col.startswith("impacto_")]
    if not impact_cols:
        return out

    out["cluster_label"] = out["profile_cluster"]
    cluster_means = out.dropna(subset=["profile_cluster"]).groupby(["role", "profile_cluster"])[impact_cols].mean()
    label_map: dict[str, str] = {}

    for role, role_means in cluster_means.groupby(level=0):
        rules = CLUSTER_LABEL_RULES.get(role)
        if not rules:
            continue
        means = role_means.droplevel(0)
        z_means = (means - means.mean()) / means.std(ddof=0).replace(0, np.nan)
        z_means = z_means.fillna(0)

        used_labels: set[str] = set()
        for cluster_id, row in z_means.iterrows():
            scores = {}
            for label, weights in rules.items():
                valid = {col: weight for col, weight in weights.items() if col in row.index}
                if not valid:
                    continue
                scores[label] = sum(row[col] * weight for col, weight in valid.items()) / sum(valid.values())
            if not scores:
                continue
            ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            chosen = next((label for label, _ in ordered if label not in used_labels), ordered[0][0])
            used_labels.add(chosen)
            label_map[cluster_id] = chosen

    out["cluster_label"] = out["profile_cluster"].map(label_map).fillna(out["profile_cluster"])
    return out


def calculate_impact_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_model_role(df)
    for group in METRIC_SUBGROUPS:
        cols = get_model_feature_columns(out, groups=[group], exclude_team_context=True)
        out[f"impacto_{group}"] = percentile_by_role(out, cols).mean(axis=1).round(2) if cols else np.nan

    aggregate_groups = sorted(
        {group for role_weights in ROLE_AGGREGATED_IMPACT_WEIGHTS.values() for group in role_weights}
    )
    for group in aggregate_groups:
        out[f"impacto_{group}"] = np.nan
        for role in out["role"].dropna().unique():
            role_mask = out["role"].eq(role)
            role_weights = ROLE_AGGREGATED_IMPACT_WEIGHTS.get(role, {}).get(group)
            if not role_weights:
                continue
            weighted_sum = pd.Series(0.0, index=out.index)
            valid_weight_sum = pd.Series(0.0, index=out.index)
            for subgroup, weight in role_weights.items():
                col = f"impacto_{subgroup}"
                if col not in out.columns:
                    continue
                values = out[col]
                valid = role_mask & values.notna()
                weighted_sum.loc[valid] += values.loc[valid] * weight
                valid_weight_sum.loc[valid] += weight
            out.loc[role_mask, f"impacto_{group}"] = (
                weighted_sum.loc[role_mask] / valid_weight_sum.loc[role_mask].replace(0, np.nan)
            ).round(2)

    goalkeeper_cols = [
        "impacto_shot_stopping",
        "impacto_porteria_juego_aereo",
        "impacto_porteria_distribucion",
        "impacto_porteria",
    ]
    field_cols = [
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
        "impacto_ofensivo",
        "impacto_asociativo",
        "impacto_defensivo",
    ]
    is_goalkeeper = out["role"].eq("POR")
    out.loc[is_goalkeeper, existing_columns(out, field_cols)] = np.nan
    out.loc[~is_goalkeeper, existing_columns(out, goalkeeper_cols)] = np.nan

    out = assign_cluster_labels(out)

    def score(row: pd.Series) -> float:
        weights = PROFILE_GLOBAL_IMPACT_WEIGHTS.get(
            row.get("cluster_label"),
            ROLE_GLOBAL_IMPACT_WEIGHTS.get(row["role"], ROLE_GLOBAL_IMPACT_WEIGHTS["MED"]),
        )
        valid = [(row[col], weight) for col, weight in weights.items() if col in row and pd.notna(row[col])]
        return round(sum(value * weight for value, weight in valid) / sum(weight for _, weight in valid), 2)

    out["impacto_global"] = out.apply(score, axis=1)
    return out


def weighted_mean(group: pd.DataFrame, cols: list[str], weight_col: str = "Minutes") -> pd.Series:
    weights = group[weight_col].fillna(0).to_numpy() if weight_col in group.columns else np.ones(len(group))
    if weights.sum() == 0:
        weights = np.ones(len(group))
    values = group[cols].fillna(0).to_numpy()
    return pd.Series(np.average(values, weights=weights, axis=0), index=cols)


def build_team_profiles(
    df: pd.DataFrame,
    style_features: list[str] | None = None,
) -> pd.DataFrame:
    """Perfil de estilo colectivo usando solo estadísticas directas de equipo.

    Esta matriz se usa para similitud entre equipos y compatibilidad de contexto.
    No incluye métricas puras de rendimiento como goles, xG diferencial o puntos.
    """
    features = style_features or TEAM_DIRECT_STYLE_FEATURES
    team_rows = df.sort_values("Minutes", ascending=False).drop_duplicates("Team").set_index("Team")
    cols = [col for col in features if col in team_rows.columns and pd.api.types.is_numeric_dtype(team_rows[col])]
    if len(cols) < 3:
        raise ValueError("No hay suficientes estadísticas directas de estilo de equipo.")
    return team_rows[cols].apply(pd.to_numeric, errors="coerce").sort_index()


def build_role_profiles_by_team(df: pd.DataFrame, role: str, feature_cols: list[str]) -> pd.DataFrame:
    role_col = "model_role" if "model_role" in df.columns and role in set(df["model_role"].dropna()) else "role"
    role_df = df.loc[df[role_col].eq(role)].copy()
    cols = [col for col in feature_cols if col in role_df.columns]
    if not cols:
        return pd.DataFrame()
    return role_df.groupby("Team", group_keys=False).apply(lambda group: weighted_mean(group, cols)).sort_index()


def calculate_team_similarity(df: pd.DataFrame, n_components: int = 2) -> TeamSimilarityResult:
    profiles = build_team_profiles(df)
    scaled = pd.DataFrame(
        Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]).fit_transform(profiles),
        index=profiles.index,
        columns=profiles.columns,
    )
    cosine = pd.DataFrame(cosine_similarity(scaled), index=profiles.index, columns=profiles.index)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    coords = pca.fit_transform(scaled)
    pca_map = pd.DataFrame(coords, index=profiles.index, columns=[f"Team_PC{i + 1}" for i in range(n_components)])
    pca_map["cluster"] = KMeans(n_clusters=min(4, len(pca_map)), random_state=RANDOM_STATE, n_init=30).fit_predict(scaled)
    return TeamSimilarityResult(profiles, scaled, cosine, pca_map, pca)


def similar_teams(team_similarity: TeamSimilarityResult, team: str, n: int = 5) -> pd.Series:
    return team_similarity.cosine_matrix.loc[team].drop(team).sort_values(ascending=False).head(n)


def player_team_fit(
    df: pd.DataFrame,
    target_team: str,
    role: str | None = None,
    cluster_label: str | None = None,
    exclude_current_team: bool = True,
    top_n: int = 25,
    include_economic: bool = True,
    max_market_value_million: float | None = None,
    budget_million: float | None = None,
    discard_if_value_budget_ratio: float = 1.5,
    include_age: bool = True,
    ideal_max_age: float = 27.0,
    hard_max_age: float | None = None,
    max_candidate_age: float | None = None,
    require_athletic_eligible: bool = False,
) -> pd.DataFrame:
    if "impacto_global" not in df.columns:
        df = calculate_impact_scores(df)
    df = ensure_model_role(df)

    team_profiles = build_team_profiles(df)
    team_scaled = pd.DataFrame(
        Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]).fit_transform(team_profiles),
        index=team_profiles.index,
        columns=team_profiles.columns,
    )
    team_cosine = pd.DataFrame(cosine_similarity(team_scaled), index=team_profiles.index, columns=team_profiles.index)

    candidates = df.copy()
    if exclude_current_team:
        candidates = candidates.loc[~candidates["Team"].eq(target_team)].copy()
    if role:
        candidates = candidates.loc[candidates["role"].eq(role)].copy()
    if cluster_label and "cluster_label" in candidates.columns:
        candidates = candidates.loc[candidates["cluster_label"].eq(cluster_label)].copy()
    if max_candidate_age is not None and "age" in candidates.columns:
        candidate_age = pd.to_numeric(candidates["age"], errors="coerce")
        candidates = candidates.loc[candidate_age.le(float(max_candidate_age))].copy()
    if require_athletic_eligible:
        if "athletic_eligible" not in candidates.columns:
            candidates = candidates.head(0).copy()
        else:
            candidates = candidates.loc[candidates["athletic_eligible"].fillna(False).astype(bool)].copy()
    if candidates.empty:
        return candidates.head(0)

    # 1) Compatibilidad de contexto: similitud de estilo entre equipo origen y destino.
    context_similarity = candidates["Team"].map(lambda team: team_cosine.loc[team, target_team] if team in team_cosine.index else np.nan)
    candidates["context_fit"] = ((context_similarity.fillna(0) + 1) / 2 * 100).round(2)
    # 2) Necesidad del equipo: se calcula sobre los mismos subimpactos usados para perfilar cada rol.
    role_for_model = role or (candidates["role"].mode().iloc[0] if not candidates.empty else "MED")
    need_cols = [col for col in ROLE_NEED_IMPACT_COLUMNS.get(role_for_model, ROLE_NEED_IMPACT_COLUMNS["MED"]) if col in df.columns]
    if need_cols:
        team_role_means = df.loc[df["role"].eq(role_for_model)].groupby("Team")[need_cols].mean()
        if target_team in team_role_means.index:
            target_role_profile = team_role_means.loc[target_team]
        else:
            target_role_profile = team_role_means.mean()
        league_role_mean = team_role_means.mean()
        needs = (league_role_mean - target_role_profile).clip(lower=0)
        needs = needs / needs.sum() if needs.sum() > 0 else pd.Series(1 / len(need_cols), index=need_cols)
        candidates["team_need_fit"] = candidates[need_cols].fillna(50).dot(needs).round(2)
    else:
        candidates["team_need_fit"] = 50.0

    # 3) Ajuste al perfil de rol actual del equipo destino.
    role_feature_cols = get_model_feature_columns(df, groups=ROLE_GROUPS.get(role_for_model, ["asociativo"]), exclude_team_context=True)
    model_role_df = df.loc[df["model_role"].eq(role_for_model)].copy()
    role_profiles = build_role_profiles_by_team(model_role_df, role_for_model, role_feature_cols)
    if not role_profiles.empty and target_team in role_profiles.index:
        role_scaler = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        role_scaler.fit(pd.concat([model_role_df[role_feature_cols], role_profiles], axis=0))
        target_role_vec = role_scaler.transform(role_profiles.loc[[target_team], role_feature_cols])
        candidate_role_vecs = role_scaler.transform(candidates[role_feature_cols].fillna(0))
        role_similarity = cosine_similarity(candidate_role_vecs, target_role_vec).ravel()
        candidates["role_fit"] = ((role_similarity + 1) / 2 * 100).round(2)
    else:
        candidates["role_fit"] = 50.0

    has_age = include_age and "age" in candidates.columns and candidates["age"].notna().any()
    if has_age:
        age = candidates["age"]
        hard_age = float(hard_max_age) if hard_max_age is not None else float(ideal_max_age + 7)
        candidates["age_fit"] = np.where(
            age.isna(),
            60,
            np.where(
                age <= ideal_max_age,
                100,
                ((hard_age - age) / max(hard_age - ideal_max_age, 1) * 100).clip(0, 100),
            ),
        ).round(2)

    has_market_values = include_economic and "market_value_eur" in candidates.columns
    if has_market_values:
        candidates["market_value_million_eur"] = candidates["market_value_eur"] / 1_000_000
        target_values = df.loc[df["Team"].eq(target_team), "market_value_eur"] if "market_value_eur" in df.columns else pd.Series(dtype=float)
        target_values = target_values.dropna()
        if max_market_value_million is not None:
            affordability_limit = float(max_market_value_million)
        elif len(target_values) >= 4:
            # Aproximación conservadora: un club suele fichar dentro de su rango salarial/valor actual.
            affordability_limit = max(1.0, target_values.quantile(0.75) / 1_000_000)
        elif len(target_values) > 0:
            affordability_limit = max(1.0, target_values.median() / 1_000_000)
        else:
            affordability_limit = max(1.0, candidates["market_value_million_eur"].median())

        if budget_million is not None and budget_million > 0:
            hard_cap = float(budget_million) * discard_if_value_budget_ratio
        elif max_market_value_million is not None:
            # Si el usuario define un valor máximo objetivo, lo usamos como proxy de presupuesto.
            hard_cap = float(max_market_value_million) * discard_if_value_budget_ratio
        elif "budget_eur" in df.columns:
            budget_values = df.loc[df["Team"].eq(target_team), "budget_eur"].dropna()
            hard_cap = float(budget_values.iloc[0] / 1_000_000 * discard_if_value_budget_ratio) if len(budget_values) else np.inf
        else:
            hard_cap = np.inf

        candidates["economic_discarded"] = candidates["market_value_million_eur"] >= hard_cap
        candidates = candidates.loc[~candidates["economic_discarded"].fillna(False)].copy()

        candidates["affordability_limit_million_eur"] = round(affordability_limit, 2)
        candidates["hard_cap_million_eur"] = round(hard_cap, 2) if np.isfinite(hard_cap) else np.nan
        value = candidates["market_value_million_eur"]
        candidates["economic_fit"] = np.where(
            value.isna(),
            50,
            np.where(value <= affordability_limit, 100, (affordability_limit / value * 100).clip(0, 100)),
        ).round(2)
        if has_age:
            candidates["fit_score"] = (
                0.35 * candidates["impacto_global"]
                + 0.18 * candidates["context_fit"]
                + 0.14 * candidates["team_need_fit"]
                + 0.13 * candidates["role_fit"]
                + 0.12 * candidates["economic_fit"]
                + 0.08 * candidates["age_fit"]
            ).round(2)
        else:
            candidates["fit_score"] = (
                0.40 * candidates["impacto_global"]
                + 0.20 * candidates["context_fit"]
                + 0.15 * candidates["team_need_fit"]
                + 0.15 * candidates["role_fit"]
                + 0.10 * candidates["economic_fit"]
            ).round(2)
    else:
        if has_age:
            candidates["fit_score"] = (
                0.43 * candidates["impacto_global"]
                + 0.22 * candidates["context_fit"]
                + 0.18 * candidates["team_need_fit"]
                + 0.10 * candidates["role_fit"]
                + 0.07 * candidates["age_fit"]
            ).round(2)
        else:
            candidates["fit_score"] = (
                0.50 * candidates["impacto_global"]
                + 0.25 * candidates["context_fit"]
                + 0.15 * candidates["team_need_fit"]
                + 0.10 * candidates["role_fit"]
            ).round(2)

    cols = [
        "Name",
        "Team",
        "player_id_tm",
        "transfermarkt_photo_url",
        "Imagen",
        "nationality",
        "nationality_flag_url",
        "Primary Position",
        "Secondary Position",
        "role",
        "Minutes",
        "profile_cluster",
        "cluster_label",
        "impacto_global",
        "impacto_ofensivo",
        "impacto_asociativo",
        "impacto_defensivo",
        "impacto_porteria",
        "context_fit",
        "team_need_fit",
        "role_fit",
        "economic_fit",
        "age",
        "age_fit",
        "market_value_million_eur",
        "affordability_limit_million_eur",
        "hard_cap_million_eur",
        "athletic_eligible",
        "athletic_birth_place",
        "fit_score",
    ]
    return candidates[existing_columns(candidates, cols)].sort_values("fit_score", ascending=False).head(top_n)


def top_players_by_role(df: pd.DataFrame, role: str, n: int = 10) -> pd.DataFrame:
    impact_cols = (
        ["impacto_porteria"]
        if role == "POR"
        else ["impacto_ofensivo", "impacto_asociativo", "impacto_defensivo"]
    )
    cols = [
        "Name",
        "Team",
        "role",
        "Minutes",
        "profile_cluster",
        "cluster_label",
        "impacto_global",
    ] + impact_cols
    return df.loc[df["role"].eq(role), existing_columns(df, cols)].sort_values("impacto_global", ascending=False).head(n)


def parse_money_to_eur(value) -> float:
    """Convierte formatos tipo '€12.5m', '12,500,000', '1.2bn' a euros."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().lower()
    if text in {"", "-", "nan", "none", "unknown"}:
        return np.nan

    text = (
        text.replace("€", "")
        .replace("eur", "")
        .replace("£", "")
        .replace("$", "")
        .replace(" ", "")
    )
    multiplier = 1.0
    if text.endswith(("bn", "b")):
        multiplier = 1_000_000_000
        text = re.sub(r"(bn|b)$", "", text)
    elif text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]

    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")

    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan



NAME_ALIASES = {
    "nacho": "ignacio",
    "ignasi": "ignacio",
    "paco": "francisco",
    "curro": "francisco",
    "pancho": "francisco",
    "pepe": "jose",
    "josep": "jose",
    "javi": "javier",
    "xavi": "javier",
    "alex": "alejandro",
    "sandro": "alejandro",
    "toni": "antonio",
    "tony": "antonio",
    "antony": "antonio",
    "beto": "alberto",
    "rober": "roberto",
    "robert": "roberto",
    "guille": "guillermo",
    "kike": "enrique",
    "quique": "enrique",
    "lalo": "eduardo",
    "edu": "eduardo",
    "dani": "daniel",
    "nico": "nicolas",
    "fede": "federico",
    "fer": "fernando",
    "rodri": "rodrigo",
    "rafa": "rafael",
    "mike": "michael",
    "mikey": "michael",
    "micki": "michael",
    "micky": "michael",
    "chris": "christian",
    "cris": "cristian",
    "leo": "leonardo",
}

COMMON_NAME_TOKENS = {
    "de",
    "del",
    "da",
    "das",
    "do",
    "dos",
    "di",
    "la",
    "las",
    "los",
    "le",
    "y",
    "i",
    "el",
    "van",
    "von",
    "der",
    "bin",
    "ibn",
    "jr",
    "junior",
}

TEAM_ALIASES = {
    "athletic club bilbao": "athletic club",
    "athletic bilbao": "athletic club",
    "club atletico osasuna": "osasuna",
    "ca osasuna": "osasuna",
    "club atletico de madrid sad": "atletico madrid",
    "atletico de madrid": "atletico madrid",
    "deportivo alaves s a d": "deportivo alaves",
    "alaves": "deportivo alaves",
    "futbol club barcelona": "barcelona",
    "fc barcelona": "barcelona",
    "getafe club de futbol s a d team dubai": "getafe",
    "getafe club de futbol s a d": "getafe",
    "girona futbol club s a d": "girona",
    "girona fc": "girona",
    "rayo vallecano de madrid s a d": "rayo vallecano",
    "real betis balompie sad": "real betis",
    "real betis balompie": "real betis",
    "real club celta de vigo s a d": "celta vigo",
    "celta de vigo": "celta vigo",
    "real club deportivo mallorca sad": "mallorca",
    "rcd mallorca": "mallorca",
    "real madrid club de futbol": "real madrid",
    "real sociedad de futbol sad": "real sociedad",
    "real valladolid cf": "real valladolid",
    "reial club deportiu espanyol de barcelona sad": "espanyol",
    "rcd espanyol barcelona": "espanyol",
    "sevilla futbol club sad": "sevilla",
    "sevilla fc": "sevilla",
    "ud las palmas": "las palmas",
    "union deportiva las palmas": "las palmas",
    "valencia club de futbol s a d": "valencia",
    "valencia cf": "valencia",
    "villarreal club de futbol sad": "villarreal",
    "villarreal cf": "villarreal",
    "cd leganes": "leganes",
}


def normalize_name(value: str) -> str:
    """Normaliza texto para matching: acentos, mayusculas, signos y espacios."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_team_name(value: str) -> str:
    key = normalize_name(value)
    return TEAM_ALIASES.get(key, key)


def canonical_name_token(token: str) -> str:
    return NAME_ALIASES.get(token, token)


def tokenize_player_name(value: str) -> list[str]:
    tokens = []
    for token in normalize_name(value).split():
        if len(token) <= 1 or token in COMMON_NAME_TOKENS:
            continue
        tokens.append(canonical_name_token(token))
    return tokens


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        cur = [i]
        for j, char_right in enumerate(right, 1):
            cost = 0 if char_left == char_right else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _token_coverage(source: list[str], target: list[str]) -> float:
    if not source or not target:
        return 0.0
    hits = 0
    for token_source in source:
        best = 0.0
        for token_target in target:
            ratio = SequenceMatcher(None, token_source, token_target).ratio() * 100
            max_err = 1 if min(len(token_source), len(token_target)) <= 5 else 2
            if _edit_distance(token_source, token_target) <= max_err:
                ratio = max(ratio, 90)
            if token_source == token_target:
                ratio = 100
            best = max(best, ratio)
        if best >= 88:
            hits += 1
    return hits / len(source) * 100


def name_similarity(left: str, right: str) -> float:
    left_key = normalize_name(left)
    right_key = normalize_name(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 100.0

    left_tokens = tokenize_player_name(left_key)
    right_tokens = tokenize_player_name(right_key)
    coverage_left = _token_coverage(left_tokens, right_tokens)
    coverage_right = _token_coverage(right_tokens, left_tokens)
    token_score = max(coverage_left, coverage_right) * 0.70 + ((coverage_left + coverage_right) / 2) * 0.30
    fuzzy_score = max(
        SequenceMatcher(None, left_key, right_key).ratio() * 100,
        SequenceMatcher(None, " ".join(sorted(left_tokens)), " ".join(sorted(right_tokens))).ratio() * 100,
    )
    score = token_score * 0.65 + fuzzy_score * 0.35
    if token_score >= 92:
        score = max(score, 90)
    elif token_score >= 80:
        score = max(score, 82)
    return min(100.0, score)


def team_similarity_score(left: str, right: str) -> float:
    left_key = normalize_team_name(left)
    right_key = normalize_team_name(right)
    if not left_key or not right_key:
        return 50.0
    if left_key == right_key:
        return 100.0
    score = max(
        SequenceMatcher(None, left_key, right_key).ratio() * 100,
        name_similarity(left_key, right_key),
    )
    if score >= 92:
        return 100.0
    if score >= 80:
        return 82.0
    return 45.0


def _date_to_str(value) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "0000-00-00"
    return parsed.strftime("%Y-%m-%d")


def date_similarity_score(left, right) -> tuple[float, bool, str]:
    left_str = _date_to_str(left)
    right_str = _date_to_str(right)
    if left_str == "0000-00-00" or right_str == "0000-00-00":
        return 55.0, True, "fecha_faltante"
    left_parts = left_str.split("-")
    right_parts = right_str.split("-")
    if left_parts[0] != right_parts[0]:
        return 0.0, False, "anio_distinto"
    same_month = left_parts[1] == right_parts[1]
    same_day = left_parts[2] == right_parts[2]
    if same_month and same_day:
        return 100.0, True, "fecha_exacta"
    if same_month or same_day:
        return 88.0, True, "mismo_anio_error_dia_o_mes"
    return 72.0, True, "mismo_anio_dia_y_mes_distintos"


def standardize_player_market_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza CSV/Excel de Transfermarkt y fuentes economicas a columnas estandar."""
    rename_candidates = {
        "idjugador": "player_id_tm",
        "id_jugador": "player_id_tm",
        "player_id_tm": "player_id_tm",
        "player_name": "Name",
        "nombre": "Name",
        "name": "Name",
        "player": "Name",
        "jugador": "Name",
        "club": "Team",
        "club_actual": "Team",
        "squad": "Team",
        "team_name": "Team",
        "team": "Team",
        "current_club_name": "Team",
        "valor_de_mercado": "market_value_eur",
        "valor_mercado": "market_value_eur",
        "market_value": "market_value_eur",
        "market_value_in_eur": "market_value_eur",
        "market_value_eur": "market_value_eur",
        "value": "market_value_eur",
        "vencimiento_contrato": "contract_until",
        "contract_until": "contract_until",
        "contract_expires": "contract_until",
        "fecha_nacimiento": "date_of_birth",
        "date_of_birth": "date_of_birth",
        "birth_date": "date_of_birth",
        "edad": "age",
        "age": "age",
        "temporada": "season_name",
        "season_name": "season_name",
        "competicion": "competition_name",
        "competicion_": "competition_name",
        "nacionalidad": "country_id",
        "posicion": "primary_position",
        "altura": "player_height",
        "wage": "annual_wage_eur",
        "annual_wage": "annual_wage_eur",
        "annual_wage_eur": "annual_wage_eur",
        "salary": "annual_wage_eur",
    }
    normalized_cols = {col: normalize_name(col).replace(" ", "_") for col in df.columns}
    out = df.rename(columns=normalized_cols).copy()
    out = out.rename(columns={col: rename_candidates.get(col, col) for col in out.columns})

    if "Name" not in out.columns:
        raise ValueError("El archivo economico necesita una columna de jugador: Nombre, Name, player_name, player o jugador.")

    if "market_value_eur" in out.columns:
        out["market_value_eur"] = out["market_value_eur"].apply(parse_money_to_eur)
    if "annual_wage_eur" in out.columns:
        out["annual_wage_eur"] = out["annual_wage_eur"].apply(parse_money_to_eur)
    if "date_of_birth" in out.columns:
        out["date_of_birth"] = pd.to_datetime(out["date_of_birth"], errors="coerce")
    if "contract_until" in out.columns:
        out["contract_until"] = pd.to_datetime(out["contract_until"], errors="coerce").dt.date.astype("string")
    if "age" in out.columns:
        out["age"] = pd.to_numeric(out["age"], errors="coerce")

    keep = [
        "player_id_tm",
        "Name",
        "Team",
        "market_value_eur",
        "annual_wage_eur",
        "contract_until",
        "age",
        "date_of_birth",
        "season_name",
        "competition_name",
    ]
    out = out[existing_columns(out, keep)].drop_duplicates().copy()
    if "player_id_tm" in out.columns:
        out = out.drop_duplicates(subset=["player_id_tm"], keep="first")
    out["name_key"] = out["Name"].map(normalize_name)
    if "Team" in out.columns:
        out["team_key"] = out["Team"].map(normalize_team_name)
    if "date_of_birth" in out.columns:
        out["dob_str"] = out["date_of_birth"].map(_date_to_str)
        out["year_block"] = out["dob_str"].str[:4]
    return out.reset_index(drop=True)


def _default_market_value_path() -> Path:
    candidates = [
        EXTERNAL_DIR / "Combined_Transfermarkt (5).xlsx",
        EXTERNAL_DIR / "Combined_Transfermarkt (5).csv",
        EXTERNAL_DIR / "player_market_values.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_master_id_mapping(path: str | Path | None = None) -> dict[int, int]:
    """Lee el diccionario maestro externo y devuelve StatsBomb ID -> Transfermarkt ID."""
    path = Path(path) if path is not None else EXTERNAL_DIR / "diccionario_maestro_ids.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[int, int] = {}
    for row in raw.values():
        sb_id = row.get("player_id_sb") if isinstance(row, dict) else None
        tm_id = row.get("player_id_tm") if isinstance(row, dict) else None
        if pd.notna(sb_id) and pd.notna(tm_id):
            try:
                mapping[int(sb_id)] = int(tm_id)
            except (TypeError, ValueError):
                continue
    return mapping


def read_player_market_values(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else _default_market_value_path()
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el archivo economico: {path}")
    cache_path = PROCESSED_DIR / "transfermarkt_laliga_24_25.parquet"
    if path.suffix.lower() in {".xlsx", ".xls"} and cache_path.exists():
        if cache_path.stat().st_mtime >= path.stat().st_mtime:
            return pd.read_parquet(cache_path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)
    if "temporada" in raw.columns:
        raw = raw[raw["temporada"].astype(str).eq("24/25")].copy()
    standardized = standardize_player_market_values(raw)
    if "competition_name" in standardized.columns:
        competition_key = standardized["competition_name"].map(normalize_name)
        standardized = standardized.loc[competition_key.eq("laliga")].copy()
    standardized = standardized.reset_index(drop=True)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        standardized.to_parquet(cache_path, index=False)
    return standardized


def _player_identity_frame(players: pd.DataFrame) -> pd.DataFrame:
    out = players.copy()
    out["name_key"] = out["Name"].map(normalize_name)
    out["team_key"] = out["Team"].map(normalize_team_name)
    birth_col = "Date of Birth" if "Date of Birth" in out.columns else "birth_date" if "birth_date" in out.columns else None
    out["dob_str"] = out[birth_col].map(_date_to_str) if birth_col else "0000-00-00"
    out["year_block"] = out["dob_str"].str[:4]
    return out


def _candidate_names(row: pd.Series) -> list[str]:
    names = []
    for col in ["Name", "player_name", "player_known_name"]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            names.append(normalize_name(row[col]))
    first = row.get("player_first_name", "")
    last = row.get("player_last_name", "")
    full = normalize_name(f"{first} {last}")
    if full:
        names.append(full)
    return list(dict.fromkeys([name for name in names if name]))


def _best_name_score(left: pd.Series, right: pd.Series) -> float:
    scores = [name_similarity(a, b) for a in _candidate_names(left) for b in _candidate_names(right)]
    return max(scores) if scores else 0.0


def _copy_market_columns(target: pd.DataFrame, source: pd.DataFrame, row_idx, source_idx) -> None:
    for col in [
        "player_id_tm",
        "market_value_eur",
        "annual_wage_eur",
        "contract_until",
        "age",
        "date_of_birth",
        "season_name",
        "competition_name",
        "match_score_tm",
        "match_score_nombre_tm",
        "match_score_fecha_tm",
        "match_score_equipo_tm",
        "match_regla_fecha_tm",
    ]:
        if col in source.columns:
            target.at[row_idx, col] = source.at[source_idx, col]


def merge_player_market_values(
    players: pd.DataFrame,
    market_values: pd.DataFrame,
    min_name_similarity: float = 72.0,
) -> pd.DataFrame:
    """Cruza StatsBomb con Transfermarkt usando nombre, fecha de nacimiento y equipo.

    El anio de nacimiento actua como bloqueo principal cuando existe. El nombre es la
    senal principal, la fecha reduce falsos positivos y el equipo ayuda sin bloquear
    traspasos o nomenclaturas desactualizadas.
    """
    out = _player_identity_frame(players)
    mv = standardize_player_market_values(market_values) if "name_key" not in market_values.columns else market_values.copy()
    if "market_value_eur" not in mv.columns:
        return out.drop(columns=["name_key", "team_key", "dob_str", "year_block"], errors="ignore")

    out["match_score_tm"] = np.nan
    out["match_score_nombre_tm"] = np.nan
    out["match_score_fecha_tm"] = np.nan
    out["match_score_equipo_tm"] = np.nan
    out["match_regla_fecha_tm"] = pd.NA

    assigned_players = set()
    assigned_market = set()
    if "player_id" in out.columns and "player_id_tm" in mv.columns:
        sb_to_tm = read_master_id_mapping()
        mv_by_tm = mv.dropna(subset=["player_id_tm"]).copy()
        mv_by_tm["_source_index"] = mv_by_tm.index
        mv_by_tm["player_id_tm_int"] = pd.to_numeric(mv_by_tm["player_id_tm"], errors="coerce").astype("Int64")
        mv_by_tm = mv_by_tm.dropna(subset=["player_id_tm_int"]).drop_duplicates("player_id_tm_int").set_index("player_id_tm_int")
        for idx, player in out.iterrows():
            try:
                tm_id = sb_to_tm.get(int(player["player_id"]))
            except (TypeError, ValueError):
                tm_id = None
            if tm_id is None or tm_id not in mv_by_tm.index:
                continue
            source_idx = int(mv_by_tm.loc[tm_id, "_source_index"])
            out.at[idx, "match_score_tm"] = 100.0
            out.at[idx, "match_score_nombre_tm"] = 100.0
            out.at[idx, "match_score_fecha_tm"] = 100.0
            out.at[idx, "match_score_equipo_tm"] = 100.0
            out.at[idx, "match_regla_fecha_tm"] = "id_maestro"
            _copy_market_columns(out, mv, idx, source_idx)
            assigned_players.add(idx)
            assigned_market.add(source_idx)

    mv_valid = mv.dropna(subset=["market_value_eur"]).copy()
    if not mv_valid.empty:
        for idx, player in out.drop(index=list(assigned_players), errors="ignore").iterrows():
            same_name_date = mv_valid[
                mv_valid["name_key"].eq(player.get("name_key"))
                & mv_valid.get("dob_str", pd.Series(index=mv_valid.index, dtype=object)).eq(player.get("dob_str"))
                & ~mv_valid.index.isin(assigned_market)
            ]
            if same_name_date.empty:
                continue
            same_team = same_name_date[
                same_name_date.get("team_key", pd.Series(index=same_name_date.index, dtype=object)).eq(player.get("team_key"))
            ]
            if not same_team.empty:
                source_idx = same_team.index[0]
                score_team = 100.0
            elif len(same_name_date) == 1:
                source_idx = same_name_date.index[0]
                score_team = team_similarity_score(player.get("team_key", ""), same_name_date.loc[source_idx].get("team_key", ""))
            else:
                continue
            out.at[idx, "match_score_tm"] = round(90.0 + score_team * 0.10, 2)
            out.at[idx, "match_score_nombre_tm"] = 100.0
            out.at[idx, "match_score_fecha_tm"] = 100.0
            out.at[idx, "match_score_equipo_tm"] = score_team
            out.at[idx, "match_regla_fecha_tm"] = "fecha_exacta"
            _copy_market_columns(out, mv, idx, source_idx)
            assigned_players.add(idx)
            assigned_market.add(source_idx)

    candidate_matches = []
    if assigned_market:
        mv_valid = mv_valid.drop(index=list(assigned_market), errors="ignore")
    for idx, player in out.drop(index=list(assigned_players), errors="ignore").iterrows():
        year = player.get("year_block", "0000")
        if year != "0000" and "year_block" in mv_valid.columns:
            candidates = mv_valid[mv_valid["year_block"].eq(year)]
        else:
            candidates = mv_valid
        if candidates.empty:
            continue

        same_team = candidates[candidates.get("team_key", pd.Series(index=candidates.index, dtype=object)).eq(player.get("team_key"))]
        if not same_team.empty:
            candidates = pd.concat([same_team, candidates]).drop_duplicates()
        player_name_key = player.get("name_key", "")
        if len(candidates) > 25 and player_name_key:
            quick_scores = candidates["name_key"].map(
                lambda name: SequenceMatcher(None, player_name_key, name).ratio()
            )
            candidates = candidates.loc[quick_scores.sort_values(ascending=False).head(25).index]

        best = None
        for mv_idx, candidate in candidates.iterrows():
            score_name = _best_name_score(player, candidate)
            score_date, year_same, date_rule = date_similarity_score(player.get("dob_str"), candidate.get("dob_str", "0000-00-00"))
            score_team = team_similarity_score(player.get("team_key", ""), candidate.get("team_key", ""))

            if not year_same:
                continue
            if score_name < 55:
                continue
            if score_name < 68 and score_date < 88 and score_team < 80:
                continue

            final_score = score_name * 0.62 + score_date * 0.28 + score_team * 0.10
            if final_score >= min_name_similarity and (best is None or final_score > best["score"]):
                best = {
                    "player_idx": idx,
                    "market_idx": mv_idx,
                    "score": final_score,
                    "score_name": score_name,
                    "score_date": score_date,
                    "score_team": score_team,
                    "date_rule": date_rule,
                }
        if best:
            candidate_matches.append(best)

    for match in sorted(candidate_matches, key=lambda item: item["score"], reverse=True):
        if match["player_idx"] in assigned_players or match["market_idx"] in assigned_market:
            continue
        idx = match["player_idx"]
        mv_idx = match["market_idx"]
        out.at[idx, "match_score_tm"] = round(match["score"], 2)
        out.at[idx, "match_score_nombre_tm"] = round(match["score_name"], 2)
        out.at[idx, "match_score_fecha_tm"] = round(match["score_date"], 2)
        out.at[idx, "match_score_equipo_tm"] = round(match["score_team"], 2)
        out.at[idx, "match_regla_fecha_tm"] = match["date_rule"]
        _copy_market_columns(out, mv, idx, mv_idx)
        assigned_players.add(idx)
        assigned_market.add(mv_idx)

    if "market_value_eur" in out.columns:
        out["market_value_million_eur"] = out["market_value_eur"] / 1_000_000
    if "annual_wage_eur" in out.columns:
        out["annual_wage_million_eur"] = out["annual_wage_eur"] / 1_000_000
    if "impacto_global" in out.columns and "market_value_million_eur" in out.columns:
        denominator = out["market_value_million_eur"].replace(0, np.nan)
        out["impact_per_market_million"] = (out["impacto_global"] / denominator).round(2)

    return out.drop(columns=["name_key", "team_key", "dob_str", "year_block"], errors="ignore")

def read_team_budgets(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else EXTERNAL_DIR / "team_budgets.csv"
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el archivo de presupuestos: {path}")
    df = pd.read_csv(path)
    df.columns = [normalize_name(col).replace(" ", "_") for col in df.columns]
    rename = {
        "team": "Team",
        "club": "Team",
        "equipo": "Team",
        "budget": "budget_eur",
        "budget_eur": "budget_eur",
        "presupuesto": "budget_eur",
        "squad_cost": "squad_cost_eur",
    }
    df = df.rename(columns={col: rename.get(col, col) for col in df.columns})
    if "Team" not in df.columns:
        raise ValueError("El CSV de presupuestos necesita columna Team, club o equipo.")
    for col in ["budget_eur", "squad_cost_eur"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_money_to_eur)
    df["team_key"] = df["Team"].map(normalize_team_name)
    return df


def merge_team_budgets(players: pd.DataFrame, budgets: pd.DataFrame) -> pd.DataFrame:
    out = players.copy()
    budget_df = budgets.copy()
    if "team_key" not in budget_df.columns:
        budget_df["team_key"] = budget_df["Team"].map(normalize_name)
    out["team_key"] = out["Team"].map(normalize_team_name)
    budget_cols = ["team_key"] + existing_columns(budget_df, ["budget_eur", "squad_cost_eur"])
    out = out.merge(budget_df[budget_cols].drop_duplicates("team_key"), on="team_key", how="left")
    return out.drop(columns=["team_key"], errors="ignore")


MANUAL_MARKET_VALUES_24_25 = {
    "maroan sannadi harrouch": 10_000_000,
    "maroan sannadi": 10_000_000,
    "fernando lopez gonzalez": 18_000_000,
    "fer lopez": 18_000_000,
    "yoel lago amil": 2_500_000,
    "yoel lago": 2_500_000,
    "facundo tomas garces": 2_000_000,
    "facundo garces": 2_000_000,
    "roberto fernandez jaen": 6_000_000,
    "roberto fernandez": 6_000_000,
    "coba gomes da costa": 3_000_000,
    "coba gomes": 3_000_000,
    "arthur henrique ramos de oliveira melo": 5_000_000,
    "arthur ramos": 5_000_000,
    "juan herzog": 2_500_000,
    "stefan bajcetic maquieira": 9_000_000,
    "bajcetic": 9_000_000,
    "yan diomande": 1_500_000,
    "diomande": 1_500_000,
    "antony matheus dos santos": 35_000_000,
    "antony": 35_000_000,
    "jesus rodriguez caraballo": 30_000_000,
    "jesus rodriguez": 30_000_000,
    "juan camilo hernandez suarez": 18_000_000,
    "juan camilo hernandez": 18_000_000,
    "raul asencio del rosario": 40_000_000,
    "raul asencio": 40_000_000,
    "jon martin vicente": 8_000_000,
    "jon martin": 8_000_000,
    "abulay juma bah": 6_000_000,
    "abdulay juma bah": 6_000_000,
    "juma bah": 6_000_000,
    "adam aznou ben cheikh": 3_000_000,
    "adam cheikh": 3_000_000,
    "antonio candela": 1_800_000,
    "florian grillitsch": 2_000_000,
    "grillitsch": 2_000_000,
    "tamas nikitscher": 1_000_000,
    "nikitscher": 1_000_000,
    "ruben vargas": 7_000_000,
    "pau navarro badenes": 3_000_000,
    "pau navarro": 3_000_000,
}


def apply_manual_market_values(players: pd.DataFrame) -> pd.DataFrame:
    out = players.copy()
    if "Name" not in out.columns:
        return out
    if "market_value_eur" not in out.columns:
        out["market_value_eur"] = np.nan
    if "economic_match_rule" not in out.columns:
        out["economic_match_rule"] = pd.NA

    name_key = out["Name"].map(normalize_name)
    missing_value = out["market_value_eur"].isna()
    manual_values = name_key.map(MANUAL_MARKET_VALUES_24_25)
    mask = missing_value & manual_values.notna()
    out.loc[mask, "market_value_eur"] = manual_values.loc[mask].astype(float)
    out.loc[mask, "economic_match_rule"] = "manual_24_25"
    if "market_value_million_eur" in out.columns or mask.any():
        out["market_value_million_eur"] = out["market_value_eur"] / 1_000_000
    if "impacto_global" in out.columns:
        denominator = out["market_value_million_eur"].replace(0, np.nan)
        out["impact_per_market_million"] = (out["impacto_global"] / denominator).round(2)
    return out


def add_economic_data(
    players: pd.DataFrame,
    player_values_path: str | Path | None = None,
    team_budgets_path: str | Path | None = None,
) -> pd.DataFrame:
    out = players.copy()
    values_path = Path(player_values_path) if player_values_path is not None else _default_market_value_path()
    budgets_path = Path(team_budgets_path) if team_budgets_path is not None else EXTERNAL_DIR / "team_budgets.csv"

    if values_path.exists():
        out = merge_player_market_values(out, read_player_market_values(values_path))
    out = apply_manual_market_values(out)
    if budgets_path.exists():
        out = merge_team_budgets(out, read_team_budgets(budgets_path))
    return out
