import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Smart Home Control (XGBoost)", layout="wide")

# 1. Cargar el modelo XGBoost
@st.cache_resource
def load_model():
    return joblib.load('modelo_control_hvac_xgb_interno.pkl')

try:
    modelo = load_model()
except:
    st.error("⚠️ No se encontró 'modelo_control_hvac_xgb_interno.pkl'. Ejecuta el entrenamiento primero.")
    st.stop()

# --- INTERFAZ ---
st.title("🏠 Control Térmico Interno (XGBoost)")
st.markdown("""
Sistema de control inteligente basado en **XGBoost** que determina el Setpoint ($T_a$) óptimo.
*Utiliza únicamente las condiciones internas y el estado de la vivienda (sin variables personales).*
""")

st.divider()

# 2. Controles (Inputs)
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("🌡️ Condiciones Internas")
    tr = st.slider("🧱 Temp. Paredes (Tr)", 10.0, 35.0, 24.0, step=0.5, help="Temperatura radiante media.")
    rh = st.slider("💧 Humedad Interior (%)", 20.0, 90.0, 50.0, step=5.0)
    vel = st.slider("💨 Viento Interior (m/s)", 0.0, 1.0, 0.1, step=0.05)

with col2:
    st.subheader("🎛️ Estado de la Casa")
    pmv = st.slider("🎯 Objetivo (PMV)", -0.5, 0.5, 0.0, step=0.1, help="Rango de confort.")
    
    st.write("---")
    st.caption("Aberturas (1=Cerrado, 0=Abierto)")
    blind = st.checkbox("Cortinas Cerradas", value=True)
    window = st.checkbox("Ventanas Cerradas", value=True)
    door = st.checkbox("Puertas Cerradas", value=True)
    fan_on = st.checkbox("Ventilador Encendido", value=False)
    
    # Conversión para el modelo
    val_blind = 1 if blind else 0
    val_win = 1 if window else 0
    val_door = 1 if door else 0
    val_fan = 1 if fan_on else 0
    val_heat = 0 # Asumimos calefactor auxiliar apagado para cálculo base

# 3. Predicción
# Orden exacto: ['tr', 'rh', 'vel', 'pmv_ce', 'blind_curtain', 'fan', 'window', 'door', 'heater']
input_data = pd.DataFrame([[tr, rh, vel, pmv, val_blind, val_fan, val_win, val_door, val_heat]], 
                          columns=['tr', 'rh', 'vel', 'pmv_ce', 'blind_curtain', 'fan', 'window', 'door', 'heater'])

prediccion = modelo.predict(input_data)[0]

# 4. Visualización
with col3:
    st.subheader("✅ Setpoint Calculado")
    st.metric(label="Temperatura Aire Interior (Ta)", value=f"{prediccion:.1f} °C")

    # Lógica explicativa
    if prediccion < 20:
        st.info(f"❄️ **Estrategia:** El sistema sugiere bajar la temperatura. Si las paredes están frías (<18°C), esto mantiene la eficiencia.")
    elif prediccion > 26:
        st.warning(f"🔥 **Estrategia:** El sistema sugiere subir la temperatura. Si las paredes están calientes (>26°C), esto evita el choque térmico.")
    else:
        st.success(f"🍃 **Estrategia:** Temperatura estándar de confort.")

st.divider()

# 5. Mapa de Calor (Paredes vs Humedad)
st.subheader("🗺️ Mapa de Decisión XGBoost (Paredes vs Humedad)")

plot_col, text_col = st.columns([3, 1])

with plot_col:
    # Grilla
    tr_range = np.linspace(10, 35, 50)
    rh_range = np.linspace(20, 90, 50)
    tr_grid, rh_grid = np.meshgrid(tr_range, rh_range)
    
    num_puntos = tr_grid.size
    
    # Matriz constante
    sim_matrix = pd.DataFrame({
        'tr': tr_grid.ravel(),
        'rh': rh_grid.ravel(),
        'vel': np.full(num_puntos, vel),
        'pmv_ce': np.full(num_puntos, pmv),
        'blind_curtain': np.full(num_puntos, val_blind),
        'fan': np.full(num_puntos, val_fan),
        'window': np.full(num_puntos, val_win),
        'door': np.full(num_puntos, val_door),
        'heater': np.full(num_puntos, val_heat)
    })
    sim_matrix = sim_matrix[['tr', 'rh', 'vel', 'pmv_ce', 'blind_curtain', 'fan', 'window', 'door', 'heater']]

    # Predecir
    temp_flat = modelo.predict(sim_matrix)
    temp_grid = temp_flat.reshape(tr_grid.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    contour = ax.contourf(tr_grid, rh_grid, temp_grid, levels=20, cmap='coolwarm')
    cbar = plt.colorbar(contour)
    cbar.set_label('Setpoint (Ta) [°C]', rotation=270, labelpad=15)

    # Punto actual
    ax.scatter(tr, rh, color='yellow', s=150, edgecolors='black', label='Condición Actual', zorder=10)
    
    ax.set_xlabel('Temp. Paredes (Tr) [°C]')
    ax.set_ylabel('Humedad (%)')
    ax.set_title(f'Setpoint XGBoost (Control Interno)')
    ax.legend()
    
    st.pyplot(fig)

with text_col:
    st.markdown("### Análisis")
    st.write("Este gráfico muestra cómo XGBoost ajusta la temperatura del aire basándose en la radiación de las paredes y la humedad, ignorando factores externos.")