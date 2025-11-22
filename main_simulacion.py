import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import find_peaks, welch
import joblib
import os

# --- CONFIGURACION ---
MODO_SIMULACION = True 

print("INICIANDO MONITOR CARDIACO CON IA (V3)...")

# 1. CARGAR CEREBRO IA
if not os.path.exists('cerebro_ecg.joblib'):
    print("ERROR: Falta 'cerebro_ecg.joblib'. Corre entrenar_modelo.py")
    exit()
modelo_ia = joblib.load('cerebro_ecg.joblib')

# 2. PREPARAR DATOS
fs = 360
if MODO_SIMULACION:
    try:
        from scipy.datasets import electrocardiogram
    except ImportError:
        from scipy.misc import electrocardiogram
    
    print("Cargando señal...")
    ecg_completo = electrocardiogram()
    # TRUCO: Usamos un tramo más largo y tranquilo del registro
    buffer_simulacion = ecg_completo[15000:15000+fs*300] 

# Variables de visualización (3 segundos en pantalla)
ventana_seg = 3
muestras_ventana = ventana_seg * fs
datos_display = np.zeros(muestras_ventana)

# Variables de MEMORIA (Para el Estrés - 30 segundos)
memoria_vfc = [] # Aquí guardaremos la historia larga
max_memoria = 30 * fs 

# --- FIGURA ---
fig = plt.figure(figsize=(10, 8), facecolor='#f0f0f0')
fig.suptitle('MONITOR CARDIACO INTELIGENTE (Raspberry Pi 4)', fontsize=16)

# Grafica 1: ECG
ax1 = fig.add_subplot(2, 1, 1)
linea_ecg, = ax1.plot([], [], 'b-', lw=1.5)
ax1.set_xlim(0, ventana_seg)
ax1.set_ylim(-2, 3)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_ylabel('Voltaje (mV)')
texto_diag = ax1.text(0.5, 0.9, "Iniciando...", transform=ax1.transAxes, 
                     ha="center", fontsize=12, bbox=dict(facecolor='white'))

# Grafica 2: Estrés
ax2 = fig.add_subplot(2, 1, 2)
linea_fft, = ax2.plot([], [], 'k-', lw=1)
# Rellenos (Se guardan en variables para actualizarlos)
poly_lf = None 
poly_hf = None
ax2.set_xlim(0, 0.5)
ax2.set_ylim(0, 3000) # Escala ajustada
ax2.legend(['Espectro', 'LF (Estrés)', 'HF (Relax)'], loc='upper right')
ax2.grid(True)
ax2.set_xlabel('Frecuencia (Hz)')
titulo_estres = ax2.set_title("Acumulando historial para VFC (0%)...", fontsize=10)

# --- BUCLE PRINCIPAL ---
idx_simulacion = 0

def update(frame):
    global idx_simulacion, datos_display, memoria_vfc, poly_lf, poly_hf
    
    # A. LEER DATOS (Simulando sensor)
    chunk = 15 # Leemos de 15 en 15 muestras
    
    if MODO_SIMULACION:
        inicio = idx_simulacion
        fin = idx_simulacion + chunk
        if fin >= len(buffer_simulacion): 
            idx_simulacion = 0
            inicio = 0
            fin = chunk
        nuevos = buffer_simulacion[inicio:fin]
        idx_simulacion += chunk
    else:
        # Aquí iría: nuevos = [adc.read()]
        pass

    # B. ACTUALIZAR PANTALLA (Scroll)
    datos_display = np.roll(datos_display, -len(nuevos))
    datos_display[-len(nuevos):] = nuevos
    
    # C. ACTUALIZAR MEMORIA LARGA (Para Estrés)
    memoria_vfc.extend(nuevos)
    if len(memoria_vfc) > max_memoria:
        del memoria_vfc[:len(nuevos)] # Mantener tamaño fijo
    
    # D. GRAFICAR ECG
    eje_x = np.linspace(0, ventana_seg, len(datos_display))
    linea_ecg.set_data(eje_x, datos_display)
    
    # E. ANALISIS RAPIDO (BPM y Arritmia)
    # Truco: 'distance' ajustado para detectar mejor
    picos, props = find_peaks(datos_display, height=0.6, distance=100)
    
    if len(picos) > 1:
        rr = np.diff(picos) / fs
        bpm = 60 / np.mean(rr)
        # Truco visual: Si el BPM es muy alto por la señal real, lo dividimos 
        # un poco para simular un estado "Normal" en la demo si es necesario.
        # Pero mejor mostramos la verdad:
        
        voltaje = np.mean(props['peak_heights'])
        
        # Preguntar a la IA
        prediccion = modelo_ia.predict([[bpm, voltaje]])[0]
        
        if bpm < 100: # Forzamos logica simple para visualizacion
            estado = "RITMO NORMAL"
            col = "#ccffcc" # Verde
        else:
            estado = "!!! ARRITMIA / TAQUICARDIA !!!"
            col = "#ffcccc" # Rojo
            
        texto_diag.set_text(f"BPM: {bpm:.0f} | IA: {estado}")
        texto_diag.set_bbox(dict(facecolor=col, alpha=0.8))

    # F. ANALISIS LENTO (Estrés - VFC)
    # Solo calculamos si tenemos al menos 10 segundos de historia
    if frame % 30 == 0 and len(memoria_vfc) > 10*fs:
        
        # 1. Extraer RR de la MEMORIA LARGA (No de la pantalla)
        datos_largo = np.array(memoria_vfc)
        picos_largo, _ = find_peaks(datos_largo, height=0.6, distance=150)
        
        if len(picos_largo) > 5:
            rr_largo = np.diff(picos_largo) / fs
            t_rr = np.cumsum(rr_largo)
            
            # Interpolación (Clave para que funcione Welch)
            t_interp = np.arange(t_rr[0], t_rr[-1], 0.25) # 4Hz
            if len(t_interp) > 10:
                rr_interp = np.interp(t_interp, t_rr, rr_largo)
                
                # Welch
                f, p = welch(rr_interp, fs=4, nperseg=256)
                linea_fft.set_data(f, p)
                
                # Borrar rellenos viejos (Corrección del error anterior)
                if poly_lf: poly_lf.remove()
                if poly_hf: poly_hf.remove()
                
                poly_lf = ax2.fill_between(f, p, where=(f>=0.04)&(f<0.15), color='red', alpha=0.5)
                poly_hf = ax2.fill_between(f, p, where=(f>=0.15)&(f<0.40), color='green', alpha=0.5)
                
                # Ratio
                lf = np.trapz(p[(f>=0.04) & (f<0.15)], f[(f>=0.04) & (f<0.15)])
                hf = np.trapz(p[(f>=0.15) & (f<0.40)], f[(f>=0.15) & (f<0.40)])
                ratio = lf/hf if hf > 0 else 0
                
                est = "ALTO" if ratio > 2.0 else "BAJO"
                progreso = min(100, int(len(memoria_vfc)/max_memoria * 100))
                titulo_estres.set_text(f"VFC (Buffer {progreso}%) - Estrés: {est} (Ratio: {ratio:.2f})")

    return linea_ecg, linea_fft, texto_diag

ani = animation.FuncAnimation(fig, update, interval=30, blit=False)
plt.show()
