import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import find_peaks, welch
import joblib
import os

# --- CONFIGURACION ---
# Si tienes el sensor, cambias MODO_SIMULACION a False
MODO_SIMULACION = True 

print("INICIANDO MONITOR CARDIACO CON IA...")

# 1. CARGAR CEREBRO IA
if not os.path.exists('cerebro_ecg.joblib'):
    print("ERROR: No encuentro 'cerebro_ecg.joblib'. Ejecuta entrenar_modelo.py primero.")
    exit()
modelo_ia = joblib.load('cerebro_ecg.joblib')
print("IA Cargada OK.")

# 2. PREPARAR DATOS (Solo para simulacion)
fs = 360
if MODO_SIMULACION:
    try:
        from scipy.datasets import electrocardiogram
    except ImportError:
        from scipy.misc import electrocardiogram
    
    print("Descargando señal de prueba...")
    ecg_completo = electrocardiogram()
    # Usamos un tramo largo para simular medicion continua
    buffer_simulacion = ecg_completo[0:fs*300] 

# Variables globales para graficar
ventana_seg = 3
muestras_ventana = ventana_seg * fs
datos_display = np.zeros(muestras_ventana)

# --- FIGURA Y GRAFICAS ---
fig = plt.figure(figsize=(10, 8), facecolor='#f0f0f0')
fig.suptitle('MONITOR CARDIACO INTELIGENTE (Raspberry Pi 4)', fontsize=16)

# Grafica 1: ECG en tiempo real
ax1 = fig.add_subplot(2, 1, 1)
linea_ecg, = ax1.plot([], [], 'b-', lw=1.5)
ax1.set_xlim(0, ventana_seg)
ax1.set_ylim(-2, 3)
ax1.grid(True, which='both', linestyle='--', alpha=0.7)
ax1.set_ylabel('Voltaje (mV)')
texto_diag = ax1.text(0.5, 0.9, "Esperando datos...", transform=ax1.transAxes, 
                     ha="center", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Grafica 2: Analisis de Estres (Fourier)
ax2 = fig.add_subplot(2, 1, 2)
linea_fft, = ax2.plot([], [], 'k-', lw=1)
poly_lf = ax2.fill_between([], [], color='red', alpha=0.3, label='LF (Estres)')
poly_hf = ax2.fill_between([], [], color='green', alpha=0.3, label='HF (Relax)')
ax2.set_xlim(0, 0.5)
ax2.set_ylim(0, 500)
ax2.legend(loc='upper right')
ax2.grid(True)
ax2.set_xlabel('Frecuencia (Hz)')
titulo_estres = ax2.set_title("Nivel de Estrés: Calculando...", fontsize=10)

# --- FUNCION PRINCIPAL (SE EJECUTA CADA CUADRO) ---
idx_simulacion = 0

def update(frame):
    global idx_simulacion, datos_display
    
    # 1. ADQUIRIR DATOS (Simulado vs Real)
    nuevos_datos = []
    
    if MODO_SIMULACION:
        ### ZONA DE CAMBIO HARDWARE: AQUI LEERIAS EL ADS1115 ###
        # Simula que leemos 10 muestras nuevas del sensor
        chunk = 10
        inicio = idx_simulacion
        fin = idx_simulacion + chunk
        if fin >= len(buffer_simulacion): # Reiniciar si se acaba
            idx_simulacion = 0
            inicio = 0
            fin = chunk
        
        nuevos_datos = buffer_simulacion[inicio:fin]
        idx_simulacion += chunk
    else:
        # AQUI IRA TU CODIGO: val = adc.read_volts()
        pass

    # Efecto de "Scroll": Borramos lo viejo, metemos lo nuevo
    datos_display = np.roll(datos_display, -len(nuevos_datos))
    datos_display[-len(nuevos_datos):] = nuevos_datos
    
    # 2. PROCESAMIENTO (Cada cierto tiempo para no saturar)
    # Actualizamos visualmente siempre
    eje_x = np.linspace(0, ventana_seg, len(datos_display))
    linea_ecg.set_data(eje_x, datos_display)
    
    # Analisis matematico (solo si hay picos claros)
    picos, props = find_peaks(datos_display, height=0.5, distance=100)
    
    bpm = 0
    voltaje_pico = 0
    estado_ia = "Esperando..."
    color_aviso = "white"
    
    if len(picos) > 1:
        # Calcular caracteristicas
        rr = np.diff(picos) / fs
        bpm = 60 / np.mean(rr)
        voltaje_pico = np.mean(props['peak_heights'])
        
        # --- CONSULTA A LA IA ---
        # La IA espera [[BPM, VOLTAJE]]
        prediccion = modelo_ia.predict([[bpm, voltaje_pico]])[0]
        
        if prediccion == 0:
            estado_ia = "RITMO NORMAL"
            color_aviso = "#ccffcc" # Verde claro
        else:
            estado_ia = "!!! ARRITMIA DETECTADA !!!"
            color_aviso = "#ffcccc" # Rojo claro
            
    # Actualizar texto en pantalla
    texto_diag.set_text(f"BPM: {bpm:.0f} | IA Diagnóstico: {estado_ia}")
    texto_diag.set_bbox(dict(facecolor=color_aviso, alpha=0.8))

    # 3. ANALISIS DE ESTRES (Cada 50 cuadros)
    if frame % 50 == 0 and len(datos_display) > 0:
        f, p = welch(datos_display, fs=fs, nperseg=len(datos_display))
        linea_fft.set_data(f, p)
        
        # Actualizar colores de estres
        ax2.collections.clear()
        ax2.fill_between(f, p, where=(f>=0.04)&(f<0.15), color='red', alpha=0.3)
        ax2.fill_between(f, p, where=(f>=0.15)&(f<0.40), color='green', alpha=0.3)
        
        # Ratio
        lf = np.trapz(p[(f>=0.04) & (f<0.15)], f[(f>=0.04) & (f<0.15)])
        hf = np.trapz(p[(f>=0.15) & (f<0.40)], f[(f>=0.15) & (f<0.40)])
        ratio = lf/hf if hf > 0 else 0
        diag_estres = "ALTO" if ratio > 2.0 else "Normal"
        titulo_estres.set_text(f"Analisis VFC (Fourier) - Estrés: {diag_estres} (Ratio: {ratio:.2f})")

    return linea_ecg, linea_fft, texto_diag

# Lanzar animacion
ani = animation.FuncAnimation(fig, update, interval=30, blit=False)
plt.show()
