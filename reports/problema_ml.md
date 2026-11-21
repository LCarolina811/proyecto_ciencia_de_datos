# Formulación del Problema de Machine Learning — Titanic

**Autora:** Luz Carolina Hernandez Vega

---

## Definición del problema

| Decisión | Respuesta | Justificación |
|---|---|---|
| **Tipo de problema** | Clasificación binaria supervisada | La variable objetivo es categórica binaria (sobrevivió / no sobrevivió) y contamos con etiquetas conocidas. |
| **Variable objetivo** | `Survived` (1 = sobrevivió, 0 = no sobrevivió) | Es la única variable que responde la pregunta central del negocio. |
| **Unidad de análisis y momento de predicción** | Un pasajero, *al embarcar* | Solo se usan variables disponibles antes del evento: edad, clase, género, título, puerto y composición familiar. |
| **Métrica principal** | **ROC-AUC** | Con ~38 % de positivos, la exactitud engaña (un modelo que siempre dice "no sobrevive" acierta ~62 %). ROC-AUC mide el poder discriminativo sin depender del umbral. |
| **Métricas secundarias** | Precisión, recall, F1, exactitud | Se reportan para dar contexto completo. F1 resume el balance entre precisión y recall. |
| **Costo de errores** | Falso negativo (predecir "no sobrevive" a quien sí lo haría) tiene mayor costo de tipo humanitario. Para una aseguradora, un falso positivo (predecir "sobrevive" a quien no) podría subestimar el riesgo del siniestro. | El umbral óptimo depende de la aplicación concreta; aquí se usa 0.5 como valor estándar. |
| **Criterio de éxito** | Superar el baseline de regla de género (exactitud ≈ 79 %, ROC-AUC ≈ 0.85) por un margen estadísticamente relevante. | El baseline de género ya incorpora la variable más informativa; superarlo demuestra que el modelo extrae valor adicional. |

---

## Auditoría de variables

| Columna | ¿Existe antes del evento? | ¿Riesgo de fuga? | Decisión |
|---|---|---|---|
| `Survived`, `Survivor` | No (es el objetivo) | **Fuga directa** | Excluir |
| `PassengerId`, `Ticket`, `Name` | Sí | Identificadores: memorizan, no generalizan | Excluir |
| `NSex`, `NEmbarked`, `NCabin`, `NTitulo` | — | Duplican a `Sex`, etc. con orden arbitrario | Excluir (usar *one-hot* desde los originales) |
| `Age` imputada | — | Calculada con todos los datos (fuga) | Usar `AgeOriginal` en su lugar |
| `AgeOriginal`, `AgeImputada` | — | AgeOriginal es segura; AgeImputada es redundante | Usar `AgeOriginal`; excluir `AgeImputada` |
| `Cabin`, `TCabin`, `HasCabin` | **Dudoso** | ¿Se registró la cabina *después* del naufragio? | Evaluar con y sin ella (ablación, Paso 61) |
| `Person` | Derivada de `Age`/`Sex` | Redundante con ambas variables originales | Excluir del modelo base |
| Variables incluidas | `Age` (orig.), `SibSp`, `Parch`, `Fare`, `Familia`, `Pclass`, `Sex`, `Embarked`, `TTitulo`, `Alone` | Ninguna con riesgo de fuga | **Incluir** |

---

## Baselines

| Modelo | Exactitud | Precisión | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy (most_frequent) | ≈ 0.62 | — | — | — | ≈ 0.50 |
| Regla de género (female=1) | ≈ 0.79 | ≈ 0.74 | ≈ 0.74 | ≈ 0.74 | ≈ 0.84 |

**Criterio de éxito:** ROC-AUC > 0.86 en validación cruzada (superar claramente la regla de género).

---

## Selección de modelos finalistas

Tras validación cruzada estratificada (k=5) sobre el conjunto de entrenamiento:

| Modelo | ROC-AUC (media ± std) | Brecha train-val | Justificación |
|---|---|---|---|
| **HGB** | ≈ 0.87 ± 0.03 | Moderada | Mejor desempeño, robusto a valores atípicos, maneja NaN internamente |
| **Random Forest** | ≈ 0.86 ± 0.03 | Moderada | Desempeño similar, interpretable vía importancia |
| **Logistic Regression** | ≈ 0.84 ± 0.03 | Baja | Interpretable vía coeficientes, buena baseline lineal |

**Modelo elegido: HistGradientBoostingClassifier (HGB)**  
Justificación: mayor ROC-AUC promedio con estabilidad aceptable. Los hiperparámetros ajustados (`learning_rate=0.05`, `max_depth=3`, `max_iter=200`, `l2_regularization=1.0`) reducen la brecha entrenamiento-validación sin sacrificar desempeño.

---

## Limitaciones

1. **Tamaño de muestra pequeño:** 891 pasajeros etiquetados. Los intervalos de confianza son amplios (~±0.03–0.05 en ROC-AUC).
2. **Un solo evento histórico:** no se puede generalizar a otros accidentes marítimos.
3. **Variables faltantes:** 20 % de edades y 77 % de cabinas ausentes introducen incertidumbre.
4. **Variables sensibles:** `Sex` y `Pclass` reflejan discriminación histórica de 1912. Este modelo **no** debería usarse para decisiones actuales de seguros o evacuación basadas en género o clase.
5. **Intervalo de confianza en prueba:** con ~180 pasajeros de prueba, una diferencia de 0.02 en ROC-AUC no es estadísticamente significativa.
6. **Artefacto de cabina:** `HasCabin` puede mejorar la métrica sin aportar conocimiento real (ver Sección de auditoría). Se decidió excluirla (`USE_CABIN_FEATURES = False`).
