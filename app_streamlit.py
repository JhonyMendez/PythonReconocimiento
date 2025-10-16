import time
import numpy as np
import cv2
import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from database import Database

# Configuración de página
st.set_page_config(page_title="Sistema de Reconocimiento con IA", page_icon="🤖", layout="wide")

# Inicializar base de datos
db = Database()

st.title("🤖 Sistema de Reconocimiento con IA")
st.caption("Detección automática en tiempo real con registro por persona detectada")

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

@st.cache_resource
def load_model_cached(model_path: str):
    return load_model(model_path, compile=False)

@st.cache_data
def load_labels(labels_path: str):
    with open(labels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

# Cargar recursos
try:
    model = load_model_cached(MODEL_PATH)
    labels = load_labels(LABELS_PATH)
    st.success(f"✅ Modelo y {len(labels)} etiquetas cargados correctamente")
except Exception as e:
    st.error(f"❌ Error al cargar: {e}")
    st.stop()

# Configuración STUN para WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Clase transformadora CON guardado automático por persona detectada
class VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        self.latest = {"class": None, "confidence": 0.0}
        self.model = model
        self.labels = labels
        self.frame_count = 0
        self.frames_to_save = 30  # Guardar cada 30 frames (aprox 1 segundo)
        self.last_saved_class = None  # Para evitar duplicados consecutivos

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32).reshape(1, 224, 224, 3)
        x = (x / 127.5) - 1.0

        # Predicción
        pred = self.model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        label = self.labels[idx] if idx < len(self.labels) else f"Clase {idx}"
        conf = float(pred[0][idx])

        self.latest = {"class": label, "confidence": conf}

        # GUARDAR AUTOMÁTICAMENTE según persona detectada
        self.frame_count += 1
        umbral = st.session_state.get('umbral_confianza', 0.95)
        
        if self.frame_count >= self.frames_to_save and conf >= umbral:
            # Solo guardar si cambió la persona o es la primera detección
            if self.last_saved_class != label:
                try:
                    # CORREGIDO: Ahora llama con solo 2 parámetros
                    db.registrar_deteccion(label, conf)
                    self.last_saved_class = label
                except Exception as e:
                    print(f"❌ Error al guardar: {e}")
            self.frame_count = 0

        # Overlay en video
        overlay = img.copy()
        text = f"{label} | {conf*100:.1f}%"
        cv2.rectangle(overlay, (5, 5), (5 + 8*len(text), 45), (0, 0, 0), -1)
        cv2.putText(overlay, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        return overlay

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Configuración del Sistema")
    
    # Ajustes de cámara
    st.subheader("📹 Ajustes de Cámara")
    facing = st.selectbox(
        "Tipo de cámara", 
        ["auto (por defecto)", "user (frontal)", "environment (trasera)"],
        index=0
    )
    quality = st.selectbox("Calidad de video", ["640x480", "1280x720", "1920x1080"], index=1)
    
    st.divider()
    
    # Configuración de guardado
    st.subheader("💾 Configuración de Guardado")
    umbral_confianza = st.slider(
        "Confianza mínima para guardar (%)",
        min_value=70,
        max_value=100,
        value=95,
        step=5,
        help="Solo se guardarán detecciones con este nivel de confianza o superior"
    )
    st.session_state.umbral_confianza = umbral_confianza / 100.0
    
    st.info("ℹ️ El sistema guarda automáticamente cada vez que detecta una persona con confianza suficiente")
    
    st.divider()
    
    # Top personas detectadas
    st.subheader("📋 Top 5 Personas Detectadas")
    personas_df = db.obtener_todas_personas()
    if not personas_df.empty:
        top5 = personas_df[['nombre', 'total_detecciones']].head(5)
        st.dataframe(top5, hide_index=True, use_container_width=True)
    else:
        st.info("Aún no hay detecciones registradas")

# Media constraints
w, h = map(int, quality.split("x"))
video_constraints = {"width": w, "height": h}
if facing != "auto (por defecto)":
    video_constraints["facingMode"] = facing.split(" ")[0]
media_constraints = {"video": video_constraints, "audio": False}

# ==================== TABS PRINCIPALES ====================
tab1, tab2, tab3, tab4 = st.tabs(["📹 Detección en Vivo", "📊 Estadísticas por Persona", "📈 Análisis Global", "💾 Exportar Datos"])

# ==================== TAB 1: DETECCIÓN EN VIVO ====================
with tab1:
    st.header("Detección Automática en Tiempo Real")
    
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.subheader("Cámara en vivo")
        webrtc_ctx = webrtc_streamer(
            key="keras-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints=media_constraints,
            video_transformer_factory=VideoTransformer,
            async_processing=True,
        )
        st.info("💡 Concede permisos de cámara. Las detecciones se guardan automáticamente.", icon="ℹ️")
    
    with col2:
        st.subheader("📊 Última Detección")
        result_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        # Actualización periódica
        if webrtc_ctx and webrtc_ctx.state.playing:
            for _ in range(300000):
                if not webrtc_ctx.state.playing:
                    break
                vt = webrtc_ctx.video_transformer
                if vt is not None and vt.latest["class"] is not None:
                    cls = vt.latest["class"]
                    conf = vt.latest["confidence"]
                    
                    with result_placeholder.container():
                        st.metric("Persona Detectada", cls, f"{conf*100:.1f}%")
                        if conf >= st.session_state.get('umbral_confianza', 0.95):
                            st.success("✅ Guardando en BD")
                        else:
                            st.warning("⚠️ Confianza baja")
                    progress_placeholder.progress(min(max(conf, 0.0), 1.0))
                time.sleep(0.2)
        else:
            result_placeholder.write("🎥 Activa la cámara para comenzar")
    
    # Modo alternativo con foto
    st.markdown("---")
    with st.expander("📸 Modo Alternativo (Captura por Foto)"):
        st.write("Si tu red bloquea WebRTC, usa este modo para predecir con una foto.")
        snap = st.camera_input("Captura una imagen")
        if snap is not None:
            file_bytes = np.asarray(bytearray(snap.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            x = resized.astype(np.float32).reshape(1, 224, 224, 3)
            x = (x / 127.5) - 1.0
            pred = model.predict(x, verbose=0)
            idx = int(np.argmax(pred))
            label = labels[idx] if idx < len(labels) else f"Clase {idx}"
            conf = float(pred[0][idx])
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption=f"{label} | {conf*100:.2f}%", channels="BGR")
            with col2:
                st.success(f"**Predicción:** {label}")
                st.metric("Confianza", f"{conf*100:.2f}%")
                
                # Guardar predicción automáticamente
                if st.button("💾 Guardar esta detección"):
                    # CORREGIDO: Ahora llama con solo 2 parámetros
                    db.registrar_deteccion(label, conf)
                    st.success("✅ Detección guardada en la base de datos")
                    st.rerun()

# ==================== TAB 2: ESTADÍSTICAS POR PERSONA ====================
with tab2:
    st.header("📊 Estadísticas por Persona Detectada")
    
    # Selector de persona
    personas_df = db.obtener_todas_personas()
    if not personas_df.empty:
        persona_seleccionada = st.selectbox(
            "Selecciona una persona:",
            options=personas_df['nombre'].tolist()
        )
        
        if persona_seleccionada:
            stats = db.obtener_estadisticas_persona(persona_seleccionada)
            datos = db.obtener_datos_persona(persona_seleccionada)
            
            if not stats.empty:
                # Métricas principales
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Total Detecciones", datos['total_visitas'])
                with col2:
                    st.metric("📅 Primera Detección", datos['primera_deteccion'][:10] if datos['primera_deteccion'] else 'N/A')
                with col3:
                    promedio_conf = stats['confianza_promedio'].mean()
                    st.metric("📈 Confianza Promedio", f"{promedio_conf*100:.1f}%")
                
                st.caption(f"Última detección: {datos['ultima_deteccion'][:16] if datos['ultima_deteccion'] else 'N/A'}")
                
                st.markdown("---")
                
                # Gráfica de confianza en el tiempo
                st.subheader("📈 Historial de Detecciones")
                fig_timeline = px.line(
                    stats,
                    x='ultima_deteccion',
                    y='confianza_promedio',
                    title=f'Evolución de Confianza - {persona_seleccionada}',
                    labels={'ultima_deteccion': 'Fecha', 'confianza_promedio': 'Confianza Promedio'},
                    markers=True
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Tabla detallada
                st.subheader("📋 Detalle de Estadísticas")
                stats_display = stats.copy()
                stats_display['confianza_promedio'] = stats_display['confianza_promedio'].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(stats_display, use_container_width=True, hide_index=True)
            else:
                st.info(f"No hay estadísticas disponibles para {persona_seleccionada}")
    else:
        st.info("📭 Aún no hay personas detectadas. ¡Activa la cámara en la pestaña 'Detección en Vivo'!")

# ==================== TAB 3: ANÁLISIS GLOBAL ====================
with tab3:
    st.header("📈 Análisis Global del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Ranking de Personas Detectadas")
        personas_df = db.obtener_todas_personas()
        if not personas_df.empty:
            fig_ranking = px.bar(
                personas_df.head(10),
                x='nombre',
                y='total_detecciones',
                title='Top 10 Personas Más Detectadas',
                labels={'nombre': 'Persona', 'total_detecciones': 'Total Detecciones'},
                color='total_detecciones',
                color_continuous_scale='Blues',
                text='total_detecciones'
            )
            fig_ranking.update_traces(textposition='outside')
            fig_ranking.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_ranking, use_container_width=True)
            
            # Tabla de ranking
            st.dataframe(
                personas_df[['nombre', 'total_detecciones', 'ultima_visita']].head(10),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No hay datos globales disponibles aún")
    
    with col2:
        st.subheader("📅 Actividad por Fecha")
        detecciones_fecha = db.obtener_detecciones_por_fecha()
        if not detecciones_fecha.empty:
            fig_linea = px.line(
                detecciones_fecha,
                x='fecha',
                y='total',
                color='etiqueta',  # CORREGIDO: Ahora usa 'etiqueta' en vez de 'persona'
                title='Detecciones por Día',
                labels={'fecha': 'Fecha', 'total': 'Total', 'etiqueta': 'Etiqueta'},
                markers=True
            )
            st.plotly_chart(fig_linea, use_container_width=True)
        else:
            st.info("No hay datos de actividad por fecha")
    
    # Métricas generales
    st.divider()
    st.subheader("📊 Métricas Generales del Sistema")
    
    if not personas_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total de Personas", len(personas_df))
        
        with col2:
            total_detecciones = int(personas_df['total_detecciones'].sum())
            st.metric("🔍 Total Detecciones", total_detecciones)
        
        with col3:
            promedio = personas_df['total_detecciones'].mean()
            st.metric("📊 Promedio por Persona", f"{promedio:.1f}")
        
        with col4:
            persona_top = personas_df.iloc[0]['nombre'] if len(personas_df) > 0 else "N/A"
            st.metric("🏆 Persona Más Detectada", persona_top)

# ==================== TAB 4: EXPORTAR DATOS ====================
with tab4:
    st.header("💾 Exportar Datos del Sistema")
    
    st.write("Descarga los datos almacenados en la base de datos en formato CSV.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Exportar Personas")
        personas_df = db.obtener_todas_personas()
        if not personas_df.empty:
            csv_personas = personas_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar Lista de Personas (CSV)",
                data=csv_personas,
                file_name=f"personas_detectadas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.dataframe(personas_df, use_container_width=True, hide_index=True)
        else:
            st.info("No hay personas para exportar")
    
    with col2:
        st.subheader("📊 Exportar Detecciones por Fecha")
        detecciones_df = db.obtener_detecciones_por_fecha()
        if not detecciones_df.empty:
            csv_detecciones = detecciones_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar Detecciones (CSV)",
                data=csv_detecciones,
                file_name=f"detecciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.dataframe(detecciones_df, use_container_width=True, hide_index=True)
        else:
            st.info("No hay detecciones para exportar")

# Footer
st.markdown("---")
st.caption("🤖 Sistema de Reconocimiento Automático con IA | Desarrollado con Streamlit, TensorFlow y SQLite")