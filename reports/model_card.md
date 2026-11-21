# Ficha del Modelo — Titanic Survival Classifier v1

**Autora:** Luz Carolina Hernandez Vega  
**Nombre del modelo:** `titanic_model_v1`  
**Versión:** 1.0  
**Fecha:** 2026-11

---

## Propósito y uso previsto

**Propósito:** predecir la probabilidad de supervivencia de un pasajero del Titanic dado un conjunto de características conocidas al momento de embarcar.

**Uso previsto:** ejercicio académico de clasificación binaria. Sirve para comprender patrones históricos del naufragio del Titanic (1912) como contexto educativo en análisis de riesgo.

**Usos NO recomendados:**
- Tomar decisiones actuales de evacuación, seguros o trato diferencial basadas en género o clase socioeconómica.
- Generalizar a otros accidentes o contextos distintos al Titanic.
- Aplicar como herramienta de predicción en entornos reales de alto impacto.

---

## Datos de entrenamiento

| Aspecto | Detalle |
|---|---|
| **Fuente** | Dataset público del Titanic (URL: `https://paste.c-net.org/MarvelJinxed`) |
| **Tamaño** | 891 pasajeros etiquetados (entrenamiento + prueba) |
| **Periodo** | Datos históricos del 15 de abril de 1912 |
| **Huella SHA-256** | Registrada en `titanic_model_v1.metadata.json` al momento del entrenamiento |
| **División** | 80 % entrenamiento (≈712 pasajeros), 20 % prueba (≈179 pasajeros), estratificada por `Survived` |

---

## Variables del modelo

**Variables incluidas (entrenamiento):**

| Variable | Tipo | Notas |
|---|---|---|
| `Age` (original, con NaN) | Numérica | Imputada por grupo Sex+Pclass dentro del Pipeline |
| `SibSp` | Numérica | Hermanos/cónyuge a bordo |
| `Parch` | Numérica | Padres/hijos a bordo |
| `Fare` | Numérica | Tarifa pagada |
| `Familia` | Numérica | SibSp + Parch |
| `Pclass` | Categórica | 1, 2, 3 (one-hot) |
| `Sex` | Categórica | male/female (one-hot) |
| `Embarked` | Categórica | S/C/Q (one-hot) |
| `TTitulo` | Categórica | Mr/Mrs/Miss/Otro (one-hot) |
| `Alone` | Categórica | Alone/with Family (one-hot) |

**Variables excluidas:** `PassengerId`, `Ticket`, `Name`, `Cabin`, `TCabin`, `HasCabin`, `Survived` (objetivo), `NSex`, `NEmbarked`, `NCabin`, `NTitulo`, `Person`, `Age` imputada, `AgeOriginal`, `AgeImputada`, `Survivor`.

---

## Arquitectura del modelo

- **Estimador base:** `HistGradientBoostingClassifier`
- **Preprocesamiento:**
  1. `GroupAgeImputer`: imputa `Age` con promedio por grupo (Sex, Pclass)
  2. `ColumnTransformer`: numéricas → mediana + StandardScaler; categóricas → imputación + OneHotEncoder
- **Hiperparámetros finales:** `learning_rate=0.05`, `max_depth=3`, `max_iter=200`, `l2_regularization=1.0`
- **Umbral de decisión:** 0.5

---

## Métricas de rendimiento

### Validación cruzada (entrenamiento, k=5, estratificada)

| Métrica | Media | ± Desv. |
|---|---|---|
| ROC-AUC | ≈ 0.87 | ± 0.03 |
| F1 | ≈ 0.74 | ± 0.04 |
| Exactitud | ≈ 0.82 | ± 0.03 |

### Conjunto de prueba (evaluación única)

| Métrica | Valor |
|---|---|
| ROC-AUC | registrado en `.metadata.json` |
| F1 | registrado en `.metadata.json` |
| Exactitud | registrado en `.metadata.json` |
| Tamaño prueba (n) | ≈ 179 pasajeros |

> Los valores exactos de prueba se encuentran en `models/titanic_model_v1.metadata.json` bajo la clave `"metrics"`.

---

## Desempeño por subgrupos

| Subgrupo | Observación |
|---|---|
| Mujeres | Alta tasa de supervivencia real (74 %); el modelo captura este patrón bien. |
| Hombres | Baja tasa real (19 %); recall más bajo para predecir sobrevivientes masculinos. |
| 3ª clase | Mayor dificultad predictiva por heterogeneidad del grupo. |
| Niños (Person='child') | Grupo pequeño (n ≈ 69); estimaciones menos estables. |

---

## Reproducción

```bash
git clone <repositorio>
cd proyecto_ciencia_de_datos
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install
make pipeline
```

El archivo `models/titanic_model_v1.joblib` y su metadata quedan en `models/`.

---

## Versiones

| Componente | Versión |
|---|---|
| Python | ≥ 3.11 |
| scikit-learn | ≥ 1.4 |
| pandas | ≥ 2.2 |
| numpy | ≥ 1.26 |

Las versiones exactas de la ejecución que generó el modelo se registran automáticamente en `titanic_model_v1.metadata.json`.
