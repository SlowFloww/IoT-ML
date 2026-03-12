import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Smart Home AI (XGBoost)", layout="wide")

# 1. Cargar el modelo XGBoost
@st.cache_resource
def load_model():
    # Asegúrate de que este archivo exista (generado en el paso anterior)
    return joblib.load('modelo_control_hvac_xgb.pkl')

try:
    modelo = load_model()
except:
    st.error("⚠️ No se encontró 'modelo_control_hvac_xgb.pkl'. Ejecuta el entrenamiento con XGBoost primero.")
    st.stop()

# --- INTERFAZ ---
st.title("🏠 Control Térmico Personalizado (XGBoost)")
st.markdown("""
Este sistema utiliza un modelo de **Gradient Boosting (XGBoost)** para calcular el setpoint de temperatura ideal ($T_a$).
*El sistema integra variables ambientales y personales (`clo`, `met`) para un ajuste preciso.*
""")

st.divider()

# 2. Controles
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("👤 Variables Personales")
    clo = st.slider("👔 Ropa (clo)", 0.3, 1.5, 0.6, step=0.1, help="0.3: Shorts - 1.5: Abrigo")
    met = st.slider("🏃 Actividad (met)", 0.8, 4.0, 1.2, step=0.1, help="1.0: Sentado - 3.0: Ejercicio")

with col2:
    st.subheader("🌡️ Variables Ambientales")
    tr = st.slider("🧱 Temp. Paredes (Tr)", 10.0, 35.0, 25.0, step=0.5)
    rh = st.slider("💧 Humedad (%)", 20.0, 80.0, 50.0, step=5.0)
    vel = st.slider("💨 Viento (m/s)", 0.0, 1.0, 0.1, step=0.05)
    
    st.write("---")
    # Restricción de Alex: PMV solo en rango de confort
    pmv = st.slider("🎯 Objetivo (PMV)", -0.5, 0.5, 0.0, step=0.1, help="Rango de confort objetivo.")

# 3. Predicción (Orden exacto del entrenamiento XGBoost)
# Features: ["tr", "rh", "vel", "met", "clo", "pmv_ce"]
input_data = pd.DataFrame([[tr, rh, vel, met, clo, pmv]], 
                          columns=["tr", "rh", "vel", "met", "clo", "pmv_ce"])

prediccion = modelo.predict(input_data)[0]

# 4. Visualización y Lógica Explicativa
with col3:
    st.subheader("✅ Setpoint Calculado")
    st.metric(label="Temperatura Aire Interior (Ta)", value=f"{prediccion:.1f} °C")

    # Lógica explicativa para la defensa
    if prediccion < 20:
        st.info(f"❄️ **Estrategia: Enfriamiento / Conservación**")
        if clo > 1.0:
            st.markdown("- **Causa:** Usuario muy abrigado. Bajar la temperatura evita sudoración.")
        elif tr < 18:
            st.markdown("- **Contexto:** Paredes frías. El modelo sugiere una temperatura eficiente para invierno.")
        elif met > 2.0:
             st.markdown("- **Causa:** Alta actividad física. Se requiere enfriamiento activo.")
            
    elif prediccion > 25:
        st.warning(f"🔥 **Estrategia: Confort Adaptativo**")
        if tr > 28:
            st.markdown("- **Fenómeno:** Paredes muy calientes. El modelo sube el setpoint para reducir el choque térmico (aclimatación al calor).")
        elif clo < 0.5:
            st.markdown("- **Causa:** Usuario con ropa ligera, tolera temperaturas más altas.")
    
    else:
        st.success(f"🍃 **Estrategia: Confort Estándar**")
        st.markdown("- Condiciones equilibradas para confort térmico.")

st.divider()

# 5. Mapa de Calor (Actividad vs Ropa - Personalización)
st.subheader("🗺️ Mapa de Decisión (Personalización XGBoost)")

plot_col, text_col = st.columns([3, 1])

with plot_col:
    # Grilla para Heatmap
    met_range = np.linspace(0.8, 4.0, 30)
    clo_range = np.linspace(0.3, 1.5, 30)
    met_grid, clo_grid = np.meshgrid(met_range, clo_range)
    
    num_puntos = met_grid.size
    
    # Matriz manteniendo constantes las ambientales actuales
    sim_matrix = pd.DataFrame({
        'tr': np.full(num_puntos, tr),
        'rh': np.full(num_puntos, rh),
        'vel': np.full(num_puntos, vel),
        'met': met_grid.ravel(),
        'clo': clo_grid.ravel(),
        'pmv_ce': np.full(num_puntos, pmv)
    })
    sim_matrix = sim_matrix[["tr", "rh", "vel", "met", "clo", "pmv_ce"]]

    # Predecir
    temp_flat = modelo.predict(sim_matrix)
    temp_grid = temp_flat.reshape(met_grid.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    contour = ax.contourf(met_grid, clo_grid, temp_grid, levels=20, cmap='coolwarm')
    cbar = plt.colorbar(contour)
    cbar.set_label('Setpoint (Ta) [°C]', rotation=270, labelpad=15)

    # Punto actual
    ax.scatter(met, clo, color='yellow', s=150, edgecolors='black', label='Usuario Actual', zorder=10)
    
    ax.set_xlabel('Actividad (met)')
    ax.set_ylabel('Ropa (clo)')
    ax.set_title(f'Setpoint según Usuario (con Paredes a {tr}°C)')
    ax.legend()
    
    st.pyplot(fig)

with text_col:
    st.markdown("### Análisis del Modelo")
    st.write("""
    Este gráfico muestra cómo el modelo XGBoost ajusta la temperatura para **cualquier** combinación de usuario, manteniendo fijas las condiciones de tu casa.
    """)
    st.info("""
    **Zona Roja:** Temperaturas más altas (ropa ligera/reposo).
    **Zona Azul:** Temperaturas más bajas (abrigo/ejercicio).
    """)