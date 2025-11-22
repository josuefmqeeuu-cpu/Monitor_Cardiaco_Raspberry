import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import find_peaks, welch
import joblib
import os

# --- CONFIGURACION ---
MODO_SIMULACION = True 

print("INICIANDO MONITOR CARDIACO CON IA (V4 - AutoZoom)...")

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
    # Usamos un tramo con arritmia clara para probar
    buffer_simulacion = ecg_completo[15000:15000+fs*300] 

# Variables de visualización
ventana_seg = 3
muestras_ventana = ventana_seg * fs
datos_display = np.zeros(muestras_ventana)

# Memoria para VFC
memoria_vfc = [] 
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

# Grafica 2: Estrés (Configuracion inicial)
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlabel('Frecuencia (Hz)')
ax2.grid(True)

# --- BUCLE PRINCIPAL ---
idx_simulacion = 0

def update(frame):
    global idx_simulacion, datos_display, memoria_vfc
    
    # A. LEER DATOS
    chunk = 15 
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
    
    # C. MEMORIA
    memoria_vfc.extend(nuevos)
    if len(memoria_vfc) > max_memoria:
        del memoria_vfc[:len(nuevos)]
    
    # D. GRAFICAR ECG
    eje_x = np.linspace(0, ventana_seg, len(datos_display))
    linea_ecg.set_data(eje_x, datos_display)
    
    # E. ANALISIS RAPIDO (IA)
    picos, props = find_peaks(datos_display, height=0.6, distance=100)
    if len(picos) > 1:
        rr = np.diff(picos) / fs
        bpm = 60 / np.mean(rr)
        voltaje = np.mean(props['peak_heights'])
        prediccion = modelo_ia.predict([[bpm, voltaje]])[0]
        
        if bpm < 100: 
            estado = "RITMO NORMAL"
            col = "#ccffcc" 
        else:
            estado = "!!! ARRITMIA DETECTADA !!!"
            col = "#ffcccc" 
            
        texto_diag.set_text(f"BPM: {bpm:.0f} | IA: {estado}")
        texto_diag.set_bbox(dict(facecolor=col, alpha=0.8))

    # F. ANALISIS LENTO (ESTRÉS - VFC)
    # Solo entramos aqui cada 30 cuadros Y si tenemos datos suficientes
    if frame % 30 == 0 and len(memoria_vfc) > 10*fs:
        
        datos_largo = np.array(memoria_vfc)
        picos_largo, _ = find_peaks(datos_largo, height=0.6, distance=150)
        
        if len(picos_largo) > 5:
            rr_largo = np.diff(picos_largo) / fs
            t_rr = np.cumsum(rr_largo)
            
            t_interp = np.arange(t_rr[0], t_rr[-1], 0.25) 
            if len(t_interp) > 10:
                rr_interp = np.interp(t_interp, t_rr, rr_largo)
                
                # --- CORRECCION DEL ERROR nperseg ---
                # Calculamos nperseg dinamicamente para que nunca falle
                n_seg = min(256, len(rr_interp))
                f, p = welch(rr_interp, fs=4, nperseg=n_seg)
                
                # --- CORRECCION VISUAL (REDIBUJAR TODO) ---
                ax2.clear() # Borramos lo viejo
                ax2.grid(True)
                ax2.set_xlabel('Frecuencia (Hz)')
                ax2.set_xlim(0, 0.5)
                # NO FIJAMOS EL LIMITE Y, dejamos que matplotlib haga auto-zoom
                
                ax2.plot(f, p, 'k-', lw=1)
                ax2.fill_between(f, p, where=(f>=0.04)&(f<0.15), color='red', alpha=0.5, label='LF (Estrés)')
                ax2.fill_between(f, p, where=(f>=0.15)&(f<0.40), color='green', alpha=0.5, label='HF (Relax)')
                ax2.legend(loc='upper right')

                # Ratio
                lf = np.trapz(p[(f>=0.04) & (f<0.15)], f[(f>=0.04) & (f<0.15)])
                hf = np.trapz(p[(f>=0.15) & (f<0.40)], f[(f>=0.15) & (f<0.40)])
                ratio = lf/hf if hf > 0 else 0
                
                est = "ALTO" if ratio > 2.0 else "BAJO"
                progreso = min(100, int(len(memoria_vfc)/max_memoria * 100))
                ax2.set_title(f"VFC (Buffer {progreso}%) - Estrés: {est} (Ratio: {ratio:.2f})")

    return linea_ecg, texto_diag

ani = animation.FuncAnimation(fig, update, interval=30, blit=False)
plt.show()
