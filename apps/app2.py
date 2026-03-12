import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Smart Home Control - Equipo 3", layout="wide")

# 1. Cargar el modelo (El modelo interno SIN t_out y SIN clo/met)
@st.cache_resource
def load_model():
    return joblib.load('modelo_controlador_interno.pkl')

try:
    modelo = load_model()
except:
    st.error("⚠️ No se encontró 'modelo_controlador_interno.pkl'. Asegúrate de haber ejecutado el último código de entrenamiento.")
    st.stop()

# --- INTERFAZ DE USUARIO ---
st.title("🏠 Control Térmico Interno (Equipo 3)")
st.markdown("""
Este sistema determina el **Setpoint de Temperatura (Ta)** ideal basándose **exclusivamente en las condiciones internas** y el estado de la vivienda.
*Se omiten variables personales y temperatura exterior para simplificar el control.*
""")

st.divider()

# 2. Columnas para los controles (Inputs)
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("🌡️ Condiciones Internas")
    # Nota: Quitamos t_out porque el modelo no lo usa
    tr = st.slider("🧱 Temp. Paredes (Tr)", 15.0, 35.0, 24.0, step=0.5, help="Temperatura radiante media de las superficies.")
    rh = st.slider("💧 Humedad Interior (%)", 20.0, 90.0, 50.0, step=5.0)
    vel = st.slider("💨 Viento Interior (m/s)", 0.0, 1.0, 0.1, step=0.05)

with col2:
    st.subheader("🎛️ Control y Estado")
    # LIMITACIÓN: PMV solo en rango de confort
    pmv = st.slider("🎯 Objetivo Confort (PMV)", -0.5, 0.5, 0.0, step=0.1, help="0.0 es Neutro (Ideal).")
    
    st.write("---")
    st.caption("Estado de Aberturas (1=Cerrado, 0=Abierto)")
    blind = st.checkbox("Cortinas Cerradas", value=True)
    window = st.checkbox("Ventanas Cerradas", value=True)
    door = st.checkbox("Puertas Cerradas", value=True)
    
    # Asumimos fan/heater en 0 para el cálculo base, o puedes agregar checkboxes si quieres
    # Para este ejemplo simple los dejamos apagados o agregamos checkbox opcional
    fan_on = st.checkbox("Ventilador Encendido", value=False)
    
    # Convertir a 1/0
    val_blind = 1 if blind else 0
    val_win = 1 if window else 0
    val_door = 1 if door else 0
    val_fan = 1 if fan_on else 0
    val_heat = 0 # Asumimos calefactor auxiliar apagado para calcular el setpoint base

# 3. Predicción
# El orden EXACTO de entrenamiento fue:
# ['tr', 'rh', 'vel', 'pmv_ce', 'blind_curtain', 'fan', 'window', 'door', 'heater']
input_data = pd.DataFrame([[tr, rh, vel, pmv, val_blind, val_fan, val_win, val_door, val_heat]], 
                          columns=['tr', 'rh', 'vel', 'pmv_ce', 'blind_curtain', 'fan', 'window', 'door', 'heater'])

prediccion = modelo.predict(input_data)[0]

# 4. Visualización del Resultado (CORREGIDO)
with col3:
    st.subheader("✅ Setpoint Calculado")
    
    # Mostrar termostato grande
    st.metric(label="Temperatura Aire Interior (Ta)", value=f"{prediccion:.1f} °C")

    # Lógica de mensajes INTELIGENTE (Analiza Tr y Ta)
    if prediccion < 20:
        if tr < 20:
            st.info(f"❄️ **Estado:** Setpoint bajo ({prediccion:.1f}°C). El modelo sigue la tendencia térmica del edificio (paredes frías a {tr}°C).")
        else:
            st.info(f"❄️ **Estrategia:** El modelo sugiere enfriar el aire para compensar el calor de las paredes ({tr}°C).")
            
    elif prediccion > 25:
        if tr > 25:
            st.warning(f"🔥 **Estado:** Setpoint alto ({prediccion:.1f}°C). El modelo sigue la inercia térmica de las paredes calientes ({tr}°C).")
        else:
            st.warning(f"🔥 **Estrategia:** El modelo sugiere calentar el aire para compensar paredes frías ({tr}°C).")
            
    else:
        st.success("🍃 **Confort:** Temperatura estándar equilibrada.")

st.divider()

# 5. Mapa de Calor (Paredes vs Humedad)
st.subheader(f"🗺️ Mapa de Decisión (Paredes vs Humedad)")

plot_col, text_col = st.columns([3, 1])

with plot_col:
    # Crear grilla (Eje X: Tr, Eje Y: Humedad)
    tr_range = np.linspace(15, 35, 50)
    rh_range = np.linspace(20, 90, 50)
    tr_grid, rh_grid = np.meshgrid(tr_range, rh_range)

    num_puntos = tr_grid.size
    
    # Preparamos matriz manteniendo constantes las otras variables
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
    
    # Ordenar columnas
    sim_matrix = sim_matrix[['tr', 'rh', 'vel', 'pmv_ce', 'blind_curtain', 'fan', 'window', 'door', 'heater']]

    # Predecir
    temp_flat = modelo.predict(sim_matrix)
    temp_grid = temp_flat.reshape(tr_grid.shape)

    # Graficar
    fig, ax = plt.subplots(figsize=(10, 5))
    contour = ax.contourf(tr_grid, rh_grid, temp_grid, levels=20, cmap='coolwarm')
    cbar = plt.colorbar(contour)
    cbar.set_label('Setpoint Interior (Ta) [°C]', rotation=270, labelpad=15)

    # Punto actual
    ax.scatter(tr, rh, color='yellow', s=150, edgecolors='black', label='Condición Actual', zorder=10)
    
    ax.set_xlabel('Temperatura Paredes (Tr) [°C]')
    ax.set_ylabel('Humedad Interior (%)')
    ax.set_title('Setpoint Ideal según Radiación y Humedad')
    ax.legend()
    
    st.pyplot(fig)

with text_col:
    st.markdown("### Interpretación")
    st.write("""
    Este gráfico muestra cómo el sistema ajusta la temperatura del aire (`Ta`) para compensar la temperatura de las paredes (`Tr`) y la humedad.
    """)