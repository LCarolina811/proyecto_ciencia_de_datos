"""Ingeniería de datos: Bronze -> Silver.

Cada transformación es una función pura: recibe un DataFrame y devuelve uno
nuevo. El orden importa y está fijado en build_silver().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE, BRONZE_LABELED_NAME, SILVER, SILVER_FINAL_NAME, SILVER_NAME,
)

log = get_logger(__name__)

EXPECTED_COLUMNS = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
]
DECKS = list("ABCDEFGTX")            # orden FIJO: X = desconocida (reproducible)
TITLE_MAP = {                        # consolidación de títulos (Paso 26)
    "Master": "Mr", "the Countess": "Mrs", "Dona": "Mrs", "Mme": "Mrs",
    "Lady": "Miss", "Mlle": "Miss", "Ms": "Miss",
}
MAIN_TITLES = ["Mr", "Mrs", "Miss"]


def validate_bronze(df: pd.DataFrame) -> pd.DataFrame:
    """Verifica que el DataFrame tenga las columnas esperadas y sin PassengerId duplicados."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en Bronze: {missing}")
    if df["PassengerId"].duplicated().any():
        raise ValueError("PassengerId tiene duplicados en Bronze")
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte Survived, Pclass, SibSp, Parch a int y Age, Fare a float."""
    df = df.copy()
    for col in ["Survived", "Pclass", "SibSp", "Parch"]:
        df[col] = df[col].astype(int)
    for col in ["Age", "Fare"]:
        df[col] = df[col].astype(float)
    return df


def add_person(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Person': 'child' si Age < 16; si no, el valor de Sex.

    Regla para edad desconocida (NaN): se usa Sex. Esto subestima a los niños
    sin edad registrada; el sesgo se cuantifica en el Reto 8.1.
    """
    df = df.copy()
    df["Person"] = np.where(df["Age"] < 16, "child", df["Sex"])
    return df


def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'TCabin' (primera letra de Cabin, 'X' si es nula) y 'HasCabin' (0/1)."""
    df = df.copy()
    df["TCabin"] = df["Cabin"].str[0].fillna("X")
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Familia' = SibSp + Parch y 'Alone' ('with Family' si Familia > 0, 'Alone' si no)."""
    df = df.copy()
    df["Familia"] = df["SibSp"] + df["Parch"]
    df["Alone"] = np.where(df["Familia"] > 0, "with Family", "Alone")
    return df


def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Survivor' mapeando 0 -> 'No' y 1 -> 'Yes' (solo para etiquetar gráficos)."""
    df = df.copy()
    df["Survivor"] = df["Survived"].map({0: "No", 1: "Yes"})
    return df


def add_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae el título del nombre y lo guarda en 'TTitulo'."""
    df = df.copy()
    df["TTitulo"] = df["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    return df


def consolidate_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza títulos raros con el mapa TITLE_MAP y agrupa el resto como 'Otro'."""
    df = df.copy()
    df["TTitulo"] = df["TTitulo"].replace(TITLE_MAP)
    df["TTitulo"] = df["TTitulo"].where(df["TTitulo"].isin(MAIN_TITLES), "Otro")
    return df


def impute_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena los faltantes de 'Embarked' con la moda."""
    df = df.copy()
    moda = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(moda)
    return df


def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa 'Age' con el promedio por 'Sex'.

    Conserva la edad original en 'AgeOriginal' (con NaN) y marca las filas
    imputadas en 'AgeImputada' (bool). Usada para EXPLORAR; el modelo (Parte 11)
    vuelve a imputar solo con datos de entrenamiento.
    """
    df = df.copy()
    df["AgeOriginal"] = df["Age"]
    df["AgeImputada"] = df["Age"].isna()
    promedios = df.groupby("Sex")["Age"].mean()
    df["Age"] = df["Age"].fillna(df["Sex"].map(promedios))
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Crea columnas numéricas NEmbarked, NSex, NCabin y NTitulo."""
    df = df.copy()
    df["NEmbarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df["NSex"] = df["Sex"].map({"female": 0, "male": 1})
    mapa_cabina = {letra: i for i, letra in enumerate(DECKS)}
    df["NCabin"] = df["TCabin"].map(mapa_cabina)
    df["NTitulo"] = df["TTitulo"].map({"Mr": 0, "Mrs": 1, "Miss": 2, "Otro": 3})
    return df


def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una COPIA sin Name, Sex, Embarked y Cabin."""
    return df.drop(columns=["Name", "Sex", "Embarked", "Cabin"])


def validate_silver(df: pd.DataFrame) -> pd.DataFrame:
    """Verifica que las columnas clave de Silver no tengan nulos. Devuelve df si es válido."""
    required_no_nulls = [
        "Survived", "Pclass", "Age", "Fare", "Embarked", "Person", "Alone",
        "TCabin", "TTitulo", "NSex", "NEmbarked", "NCabin", "NTitulo",
    ]
    for col in required_no_nulls:
        if col in df.columns and df[col].isna().any():
            raise ValueError(f"La columna '{col}' tiene nulos en Silver")
    return df


def build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """Orden deliberado: Person usa la edad ORIGINAL, así que va antes de imputar."""
    return (
        bronze
        .pipe(validate_bronze)
        .pipe(cast_types)
        .pipe(add_person)
        .pipe(add_cabin_features)
        .pipe(add_family_features)
        .pipe(add_survivor_label)
        .pipe(add_title)
        .pipe(consolidate_titles)
        .pipe(impute_embarked)
        .pipe(impute_age)
        .pipe(encode_categoricals)
    )


def run(in_dir: Path = BRONZE, out_dir: Path = SILVER) -> Path:
    bronze = pd.read_csv(Path(in_dir) / BRONZE_LABELED_NAME)
    silver = validate_silver(build_silver(bronze))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out_dir / SILVER_NAME, index=False)
    drop_original_columns(silver).to_parquet(out_dir / SILVER_FINAL_NAME, index=False)
    log.info("Silver listo: %d filas, %d columnas", *silver.shape)
    return out_dir / SILVER_NAME
