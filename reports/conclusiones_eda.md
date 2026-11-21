# Conclusiones del Análisis Exploratorio — Titanic

**Autora:** Luz Carolina Hernandez Vega  
**Dataset:** 891 pasajeros etiquetados (Titanic, 1912)

---

## 1. Hallazgos principales

### H1. El sexo fue el factor de mayor influencia sobre la supervivencia
- Tasa de supervivencia: mujeres **74 %** (n = 314), hombres **19 %** (n = 577).
- Correlación de Pearson con `Survived`: −0.54.
- Prueba chi-cuadrado: χ² ≈ 363.7, gl = 1, p < 0.001. Se rechaza la hipótesis de independencia.
- Regla "women and children first" fue aplicada de forma sistemática.

### H2. La clase social determinó el acceso a los botes
- Tasas de supervivencia: 1ª clase **63 %** (n = 216), 2ª clase **47 %** (n = 184), 3ª clase **24 %** (n = 491).
- Correlación con `Survived`: −0.34.
- Dentro de la 3ª clase, las mujeres sobrevivieron más que los hombres (41 % vs 15 %), pero mucho menos que las mujeres de 1ª clase (97 %).

### H3. Los niños tuvieron mejor oportunidad relativa, pero pocos datos
- Variable `Person = 'child'` (Age < 16): 69 pasajeros conocidos.
- Tasa de supervivencia de niños: **54 %**, superior a la de hombres adultos (19 %).
- **Sesgo**: 177 pasajeros no tienen edad registrada y fueron clasificados como adultos (ver Sección 2).

### H4. El tamaño de la familia muestra una relación no lineal
- Viajar solo: tasa de supervivencia 30 % (n ≈ 537).
- Familias pequeñas (1–3 miembros): tasas superiores al 50 %.
- Familias grandes (6–10 miembros): tasas muy bajas (<20 %), pero con n pequeño — las barras de error en las visualizaciones son amplias para estos grupos.

### H5. La tarifa tiene atípicos con alta supervivencia
- Se identificaron pasajeros con tarifas muy elevadas (>300 USD) mediante la regla IQR.
- Tasa de supervivencia de los atípicos de `Fare`: superior al promedio general, posiblemente relacionado con la clase 1.

---

## 2. Decisiones de limpieza y sesgo introducido

| Decisión | Qué se hizo | Sesgo potencial |
|---|---|---|
| **Edad imputada** | Se reemplazaron los 177 nulos de `Age` con el promedio por género (mujeres ≈ 27, hombres ≈ 30). Se conserva `AgeOriginal`. | La desviación estándar baja tras la imputación; la distribución se concentra artificialmente en las medias. No se usa para el modelo (Parte 11). |
| **`Master` → `Mr`** | El título "Master" (niño varón) se fusionó con "Mr" en `TTitulo`. | Se mezclan niños con hombres adultos en la categoría `Mr`. La variable `Person` compensa parcialmente esto al identificar menores de 16 años por edad. |
| **Cabina desconocida = 'X'** | Pasajeros sin cabina registrada obtienen `TCabin = 'X'` y `HasCabin = 0`. | Tener cabina registrada podría ser un artefacto: los sobrevivientes pudieron haber aportado esa información con más facilidad. Ver Paso 40 del módulo ML. |
| **`Person` sin edad** | Pasajeros con `Age = NaN` se quedan con su sexo en lugar de 'child'. | Subestima la cantidad de niños. Con un 20 % de edades faltantes, si la proporción de menores fuera igual a la de los que sí tienen edad (≈8 %), estimamos ≈14 niños adicionales no clasificados. |

---

## 3. Confusiones detectadas

### Puerto de embarque vs. clase social
- Southampton (S) tiene la tasa de supervivencia más baja.
- Sin embargo, estratificando por clase: dentro de cada clase, las diferencias por puerto desaparecen casi completamente.
- **Conclusión**: el puerto no "causa" menor supervivencia; refleja que más pasajeros de 3ª clase embarcaron en Southampton.

---

## 4. Qué NO podemos afirmar

1. **Asociación ≠ causalidad.** Que las mujeres sobrevivieran más se debe a decisiones humanas ("mujeres y niños primero"), no a características biológicas. El modelo predictivo captura ese patrón histórico, no una causalidad universal.

2. **Un solo evento histórico.** El Titanic es un caso único con condiciones específicas (hora, temperatura del agua, número de botes). No podemos generalizar estos patrones a otros naufragios.

3. **Datos incompletos.** El 20 % de edades y el 77 % de cabinas faltan. Los patrones observados describen a los pasajeros *con datos*, no necesariamente a todos.

4. **Intervalos de confianza amplios.** Con grupos pequeños (familias grandes, cubiertas poco frecuentes), los estadísticos son poco fiables.
