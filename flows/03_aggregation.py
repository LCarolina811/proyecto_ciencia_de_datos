"""Agregaciones: Silver -> Gold. Cada función devuelve una tabla lista para un informe."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from flows.common.logging import get_logger
from flows.common.paths import GOLD, SILVER, SILVER_NAME

log = get_logger(__name__)


def group_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Genera cuatro tablas descriptivas agrupadas."""
    return {
        "age_by_survival": df.groupby("Survived")["Age"].describe().reset_index(),
        "survival_by_sex": df.groupby("Sex")["Survived"].agg(["mean", "sum", "count"]).reset_index(),
        "survival_by_pclass": df.groupby("Pclass")["Survived"].agg(["mean", "sum", "count"]).reset_index(),
        "survival_by_alone": df.groupby("Alone")["Survived"].agg(["mean", "sum", "count"]).reset_index(),
    }


def chi2_sex_survival(df: pd.DataFrame) -> dict:
    """Devuelve {'chi2': float, 'p_value': float, 'dof': int}."""
    tabla = pd.crosstab(df["Sex"], df["Survived"])
    chi2, p, gl, _ = stats.chi2_contingency(tabla)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(gl)}


def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia con Pclass en filas y Sex en columnas."""
    out = df.pivot_table(values="Survived", index="Pclass", columns="Sex", aggfunc="mean")
    return out.rename(columns=str).reset_index()


def survival_by_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa y número de pasajeros por puerto de embarque."""
    return df.groupby("Embarked")["Survived"].agg(["mean", "count"]).reset_index()


def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia estratificada por puerto y clase."""
    out = df.pivot_table(values="Survived", index="Embarked", columns="Pclass", aggfunc="mean")
    return out.rename(columns=str).reset_index()


def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'FareQ': cuartiles de tarifa calculados dentro de cada clase."""
    df = df.copy()
    df["FareQ"] = df.groupby("Pclass")["Fare"].transform(
        lambda s: pd.qcut(s, 4, labels=False, duplicates="drop")
    )
    return df


def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia por cuartil de tarifa y clase."""
    df_q = add_fare_quartile(df)
    out = df_q.pivot_table(values="Survived", index="FareQ", columns="Pclass", aggfunc="mean")
    return out.rename(columns=str).reset_index()


def survival_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia y cantidad de pasajeros por tamaño de familia."""
    return df.groupby("Familia")["Survived"].agg(["mean", "count"]).reset_index()


def outlier_mask_iqr(s: pd.Series) -> pd.Series:
    """Devuelve una máscara booleana: True donde el valor es atípico según la regla IQR."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta atípicos en Fare, Age (usa AgeOriginal si existe) y Familia.
    Añade tasa de supervivencia de los atípicos de Fare frente al resto.
    """
    rows = []
    age_col = "AgeOriginal" if "AgeOriginal" in df.columns else "Age"

    for col, src in [("Fare", "Fare"), ("Age", age_col), ("Familia", "Familia")]:
        if src not in df.columns:
            continue
        s = df[src].dropna()
        # Crear máscara alineada con el índice original del df
        mask = pd.Series(False, index=df.index)
        mask.loc[s.index] = outlier_mask_iqr(s)

        row: dict = {"variable": col, "n_outliers": int(mask.sum())}
        if col == "Fare" and "Survived" in df.columns:
            row["survival_outliers"] = round(float(df.loc[mask, "Survived"].mean()), 3)
            row["survival_rest"] = round(float(df.loc[~mask, "Survived"].mean()), 3)
        rows.append(row)

    return pd.DataFrame(rows)


def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    df = pd.read_parquet((in_dir or SILVER) / SILVER_NAME)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        **group_stats(df),
        "survival_by_class_sex": survival_by_class_sex(df),
        "survival_by_embarked": survival_by_embarked(df),
        "survival_by_embarked_class": survival_by_embarked_class(df),
        "survival_by_fare_quartile_class": survival_by_fare_quartile_class(df),
        "survival_by_family": survival_by_family(df),
        "outlier_summary": outlier_summary(df),
    }
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "chi2_sex_survival.json").write_text(json.dumps(chi2_sex_survival(df), indent=2))
    log.info("Gold listo: %d tablas en %s", len(tables) + 1, out_dir)
    return out_dir
