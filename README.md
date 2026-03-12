# 🌡️ Control y Estimación del Confort Térmico en Smart Homes mediante IoT y ML

> Proyecto académico desarrollado en la Universidad Técnica Federico Santa María  
> Autores: Alex Aravena Tapia, Diego Maldonado Martínez, **Benjamín Mena Ardura**

---

## 📌 Descripción

Sistema inteligente para estimar la temperatura de aire óptima (T_a) que mantiene el confort térmico de un ocupante dentro del rango recomendado por la norma ASHRAE 55 (PMV ∈ [-0.5, 0.5]), utilizando únicamente variables ambientales medibles y estados de actuadores del hogar.

El proyecto aborda un **gap tecnológico real**: los sistemas HVAC actuales ignoran las variables personales del ocupante (ropa y metabolismo), lo que produce un control impreciso del confort. La solución propuesta elimina esa dependencia mediante un modelo de Machine Learning entrenado con variables 100% medibles.

---

## 🏗️ Arquitectura del Sistema

El sistema se divide en 4 capas:

| Capa | Función | Tecnología |
|------|---------|-----------|
| 1 - Percepción | Captura de variables ambientales | DHT22, ESP32-C6 |
| 2 - Comunicación | Transporte de datos | Zigbee 3.0 (mesh) |
| 3 - Soporte (ML) | Predicción del setpoint óptimo | Random Forest, XGBoost |
| 4 - Aplicación | Visualización y control | Dashboard Web |

---

## 🤖 Modelos de Machine Learning

Se evaluaron tres variantes en dos escenarios: modelo completo (6 variables) y modelo reducido (4 variables ambientales).

### Benchmark — Predicción de PMV

| Modelo | R² (6 vars) | RMSE (6 vars) | R² (4 vars) | RMSE (4 vars) |
|--------|------------|--------------|------------|--------------|
| Ridge Regression | 0.7510 | 0.3743 | 0.5216 | 0.5188 |
| Random Forest | 0.8877 | 0.2513 | 0.6559 | 0.4400 |
| **XGBoost** | **0.9614** | **0.1474** | 0.6618 | 0.4362 |

### Modelo Final — Predicción directa de T_a (Random Forest)

Dado que las variables personales (`clo`, `met`) no son medibles automáticamente, se redefinió el problema: predecir directamente la temperatura de consigna usando variables operacionales del hogar.

| Métrica | Resultado |
|---------|-----------|
| R² | **0.9865** |
| RMSE | **0.4512 °C** |

> ✅ El error es menor que la histéresis típica de termostatos comerciales (±0.5 °C), validando su uso en sistemas HVAC reales.

---

## 📊 Variables del Modelo Final

**Entradas:**
- `tr` — Temperatura radiante media (°C)
- `rh` — Humedad relativa (%)
- `vel` — Velocidad del aire (m/s)
- `pmv_ce` — PMV ajustado por efecto de enfriamiento
- `blind_curtain` — Estado de persianas (0/1)
- `fan` — Estado del ventilador (On/Off)
- `window` — Estado de ventana (Abierta/Cerrada)
- `door` — Estado de puerta (Abierta/Cerrada)
- `heater` — Estado de calefacción (On/Off)

**Salida:**
- `Ta` — Temperatura de aire óptima (°C)

---

## 📁 Estructura del Repositorio

```
├── Jupyter/               # Dataset ASHRAE Global Thermal Comfort Database and Jupyter notebooks de entrenamiento y análisis
│   ├── pmv_predict.ipynb
│   ├── t_out_predict.ipynb
├── apps/             # Methods
├── imgs/             # Images
├── papers/           # State of Art and Paper (Equipo3_Hito2.pdf)
└── README.md
```

---

## 🛠️ Tecnologías Utilizadas

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![IoT](https://img.shields.io/badge/IoT-ESP32%20%7C%20Zigbee-lightgrey)

- **ML:** Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib
- **Hardware:** ESP32-C6, DHT22, Zigbee 3.0
- **Dataset:** ASHRAE Global Thermal Comfort Database II

---

## 📚 Referencias principales

- Boutahri & Tilioua (2024) — ML predictivo para confort térmico en smart buildings
- Sung & Hsiao (2020) — Control difuso basado en IoT para regulación de PMV
- Rajuroy (2024) — Control adaptativo con Reinforcement Learning

---

## 👤 Contribución de Benjamín Mena Ardura

- Desarrollo y comparación de los tres modelos de ML (Ridge, RF, XGBoost)
- Formulación matemática del modelo inverso para estimación directa de T_a
- Análisis de sensibilidad ante actuadores y variables ambientales
- Evaluación de métricas estadísticas (R², RMSE) y validación operativa
