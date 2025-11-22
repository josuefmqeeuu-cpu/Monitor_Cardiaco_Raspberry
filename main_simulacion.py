import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import find_peaks, welch
import joblib
import os

# --- CONFIGURACION ---
MODO_SIMULACION = True 

print("INICIANDO MONITOR CARDIACO CON IA (V6 - Estabilizado)...")

# 1. CARGAR CEREBRO IA
if not os.path.exists('cerebro_ecg.joblib'):
    print("ERROR: Falta 'cerebro_ecg.joblib'.")
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
    buffer_simulacion = ecg_completo[15000:15000+fs*300] 

    # --- FILTRO DE LIMPIEZA ---
    print("Aplicando filtros...")
    filtro_suave = np.ones(15)/15 
    ecg_limpia = np.convolve(buffer_simulacion, filtro_suave, mode='same')
    filtro_lento = np.ones(300)/300 
    ola_respiracion = np.convolve(buffer_simulacion, filtro_lento, mode='same')
    buffer_simulacion = (ecg_limpia - ola_respiracion) * 2.0

# Variables de visualización
ventana_seg = 3
muestras_ventana = ventana_seg * fs
datos_display = np.zeros(muestras_ventana)

# Memorias
memoria_vfc = [] 
max_memoria = 30 * fs 

# --- NUEVO: MEMORIA PARA ESTABILIZAR BPM ---
historial_bpm = [] 
max_historial_bpm = 20 # Promediaremos los ultimos 20 cuadros

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
ax2.set_xlabel('Frecuencia (Hz)')
ax2.grid(True)

# --- BUCLE PRINCIPAL ---
idx_simulacion = 0

def update(frame):
    global idx_simulacion, datos_display, memoria_vfc, historial_bpm
    
    # A. LEER DATOS
    chunk = 40 # Velocidad rapida
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
        pass

    # B. ACTUALIZAR PANTALLA
    datos_display = np.roll(datos_display, -len(nuevos))
    datos_display[-len(nuevos):] = nuevos
    
    # C. MEMORIA VFC
    memoria_vfc.extend(nuevos)
    if len(memoria_vfc) > max_memoria:
        del memoria_vfc[:len(nuevos)]
    
    # D. GRAFICAR ECG
    eje_x = np.linspace(0, ventana_seg, len(datos_display))
    linea_ecg.set_data(eje_x, datos_display)
    
    # E. ANALISIS DE BPM (ESTABILIZADO)
    picos, props = find_peaks(datos_display, height=0.6, distance=100)
    
    if len(picos) > 1:
        # Calculo instantaneo
        rr = np.diff(picos) / fs
        bpm_inst = 60 / np.mean(rr)
        
        # --- ESTABILIZACION (COLA DE PROMEDIO) ---
        historial_bpm.append(bpm_inst)
        if len(historial_bpm) > max_historial_bpm:
            historial_bpm.pop(0) # Borrar el mas viejo
            
        # Usamos el PROMEDIO, no el instantaneo
        bpm_suave = np.mean(historial_bpm)
        
        # Diagnostico con el valor suave
        voltaje = np.mean(props['peak_heights'])
        
        if bpm_suave < 100: 
            estado = "RITMO NORMAL"
            col = "#ccffcc" 
        else:
            estado = "!!! ARRITMIA DETECTADA !!!"
            col = "#ffcccc" 
            
        texto_diag.set_text(f"BPM Promedio: {bpm_suave:.0f} | IA: {estado}")
        texto_diag.set_bbox(dict(facecolor=col, alpha=0.8))

    # F. ANALISIS ESTRÉS (Cada 30 cuadros)
    if frame % 30 == 0 and len(memoria_vfc) > 10*fs:
        
        datos_largo = np.array(memoria_vfc)
        picos_largo, _ = find_peaks(datos_largo, height=0.6, distance=150)
        
        if len(picos_largo) > 5:
            rr_largo = np.diff(picos_largo) / fs
            t_rr = np.cumsum(rr_largo)
            
            t_interp = np.arange(t_rr[0], t_rr[-1], 0.25) 
            if len(t_interp) > 10:
                rr_interp = np.interp(t_interp, t_rr, rr_largo)
                n_seg = min(256, len(rr_interp))
                f, p = welch(rr_interp, fs=4, nperseg=n_seg)
                
                ax2.clear() 
                ax2.grid(True)
                ax2.set_xlabel('Frecuencia (Hz)')
                ax2.set_xlim(0, 0.5)
                
                ax2.plot(f, p, 'k-', lw=1)
                ax2.fill_between(f, p, where=(f>=0.04)&(f<0.15), color='red', alpha=0.5, label='LF (Estrés)')
                ax2.fill_between(f, p, where=(f>=0.15)&(f<0.40), color='green', alpha=0.5, label='HF (Relax)')
                ax2.legend(loc='upper right')

                lf = np.trapz(p[(f>=0.04) & (f<0.15)], f[(f>=0.04) & (f<0.15)])
                hf = np.trapz(p[(f>=0.15) & (f<0.40)], f[(f>=0.15) & (f<0.40)])
                ratio = lf/hf if hf > 0 else 0
                
                est = "ALTO" if ratio > 2.0 else "BAJO"
                progreso = min(100, int(len(memoria_vfc)/max_memoria * 100))
                ax2.set_title(f"VFC (Buffer {progreso}%) - Estrés: {est} (Ratio: {ratio:.2f})")

    return linea_ecg, texto_diag

ani = animation.FuncAnimation(fig, update, interval=30, blit=False)
plt.show()
