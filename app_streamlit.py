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
import io
import zipfile

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

# Clase transformadora con guardado automático
class VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        self.latest = {"class": None, "confidence": 0.0}
        self.model = model
        self.labels = labels
        self.frame_count = 0
        self.frames_to_save = 30
        self.last_saved_class = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32).reshape(1, 224, 224, 3)
        x = (x / 127.5) - 1.0

        pred = self.model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        label = self.labels[idx] if idx < len(self.labels) else f"Clase {idx}"
        conf = float(pred[0][idx])

        self.latest = {"class": label, "confidence": conf}

        self.frame_count += 1
        
        # Obtener umbral individual si existe
        persona_data = db.obtener_persona(label)
        umbral = persona_data['umbral_individual'] if persona_data else st.session_state.get('umbral_confianza', 0.95)
        
        if self.frame_count >= self.frames_to_save and conf >= umbral:
            if self.last_saved_class != label:
                try:
                    db.registrar_deteccion(label, conf, fuente='camara')
                    self.last_saved_class = label
                except Exception as e:
                    print(f"❌ Error al guardar: {e}")
            self.frame_count = 0

        overlay = img.copy()
        text = f"{label} | {conf*100:.1f}%"
        cv2.rectangle(overlay, (5, 5), (5 + 8*len(text), 45), (0, 0, 0), -1)
        cv2.putText(overlay, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        return overlay

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Configuración del Sistema")
    
    st.subheader("📹 Ajustes de Cámara")
    facing = st.selectbox(
        "Tipo de cámara", 
        ["auto (por defecto)", "user (frontal)", "environment (trasera)"],
        index=0
    )
    quality = st.selectbox("Calidad de video", ["640x480", "1280x720", "1920x1080"], index=1)
    
    st.divider()
    
    st.subheader("💾 Configuración de Guardado")
    umbral_confianza = st.slider(
        "Confianza mínima global (%)",
        min_value=70,
        max_value=100,
        value=95,
        step=5,
        help="Umbral por defecto (puede ser personalizado por persona)"
    )
    st.session_state.umbral_confianza = umbral_confianza / 100.0
    
    st.info("ℹ️ El sistema guarda automáticamente cada detección con confianza suficiente")
    
    st.divider()
    
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📹 En Vivo", 
    "👥 Administración", 
    "📊 Estadísticas por Persona", 
    "📈 Analítica", 
    "💾 Exportar Datos"
])

# ==================== TAB 1: EN VIVO ====================
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
        
        if webrtc_ctx and webrtc_ctx.state.playing:
            for _ in range(300000):
                if not webrtc_ctx.state.playing:
                    break
                vt = webrtc_ctx.video_transformer
                if vt is not None and vt.latest["class"] is not None:
                    cls = vt.latest["class"]
                    conf = vt.latest["confidence"]
                    
                    persona_data = db.obtener_persona(cls)
                    umbral = persona_data['umbral_individual'] if persona_data else st.session_state.get('umbral_confianza', 0.95)
                    
                    with result_placeholder.container():
                        st.metric("Persona Detectada", cls, f"{conf*100:.1f}%")
                        if conf >= umbral:
                            st.success("✅ Guardando en BD")
                        else:
                            st.warning("⚠️ Confianza baja")
                    progress_placeholder.progress(min(max(conf, 0.0), 1.0))
                time.sleep(0.2)
        else:
            result_placeholder.write("🎥 Activa la cámara para comenzar")
    
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
                
                if st.button("💾 Guardar esta detección"):
                    db.registrar_deteccion(label, conf, fuente='imagen')
                    st.success("✅ Detección guardada en la base de datos")
                    st.rerun()

# ==================== TAB 2: ADMINISTRACIÓN ====================
with tab2:
    st.header("👥 Administración de Personas")
    
    subtab1, subtab2 = st.tabs(["➕ Agregar/Editar", "📋 Lista de Personas"])
    
    with subtab1:
        st.subheader("Gestión de Personas")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### ➕ Nueva Persona / ✏️ Editar Existente")
            
            personas_existentes = db.obtener_todas_personas()['nombre'].tolist() if not db.obtener_todas_personas().empty else []
            
            modo = st.radio("Modo", ["Agregar Nueva", "Editar Existente"], horizontal=True)
            
            if modo == "Editar Existente":
                if personas_existentes:
                    persona_editar = st.selectbox("Selecciona persona a editar", personas_existentes)
                    persona_data = db.obtener_persona(persona_editar)
                    
                    nombre = st.text_input("Nombre", value=persona_data['nombre'])
                    correo = st.text_input("Correo electrónico", value=persona_data['correo'] or "")
                    rol = st.selectbox(
                        "Rol",
                        ["Empleado", "Visitante", "Administrador", "Contratista", "Otro"],
                        index=["Empleado", "Visitante", "Administrador", "Contratista", "Otro"].index(persona_data['rol']) if persona_data['rol'] in ["Empleado", "Visitante", "Administrador", "Contratista", "Otro"] else 0
                    )
                    umbral_individual = st.slider(
                        "Umbral de confianza individual (%)",
                        min_value=70,
                        max_value=100,
                        value=int(persona_data['umbral_individual'] * 100),
                        step=5
                    )
                    notas = st.text_area("Notas", value=persona_data['notas'] or "")
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("💾 Actualizar Persona", use_container_width=True):
                            if nombre.strip():
                                success = db.actualizar_persona(
                                    persona_editar, nombre.strip(), correo.strip() or None,
                                    rol, umbral_individual/100, notas.strip() or None
                                )
                                if success:
                                    st.success("✅ Persona actualizada correctamente")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("❌ Error: El nombre ya existe")
                            else:
                                st.error("❌ El nombre es obligatorio")
                    
                    with col_btn2:
                        if st.button("🗑️ Eliminar Persona", use_container_width=True, type="secondary"):
                            db.eliminar_persona(persona_editar)
                            st.success("✅ Persona eliminada")
                            time.sleep(1)
                            st.rerun()
                else:
                    st.info("No hay personas registradas aún")
            
            else:  # Agregar Nueva
                nombre = st.text_input("Nombre *", placeholder="Ej: Juan Pérez")
                correo = st.text_input("Correo electrónico", placeholder="juan@ejemplo.com")
                rol = st.selectbox(
                    "Rol",
                    ["Empleado", "Visitante", "Administrador", "Contratista", "Otro"]
                )
                umbral_individual = st.slider(
                    "Umbral de confianza individual (%)",
                    min_value=70,
                    max_value=100,
                    value=95,
                    step=5,
                    help="Confianza mínima para guardar detecciones de esta persona"
                )
                notas = st.text_area("Notas", placeholder="Información adicional...")
                
                if st.button("➕ Agregar Persona", use_container_width=True, type="primary"):
                    if nombre.strip():
                        try:
                            db.agregar_persona(
                                nombre.strip(),
                                correo.strip() or None,
                                rol,
                                umbral_individual/100,
                                notas.strip() or None
                            )
                            st.success(f"✅ Persona '{nombre}' agregada correctamente")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                    else:
                        st.error("❌ El nombre es obligatorio")
        
        with col2:
            st.markdown("#### ℹ️ Información")
            st.info("""
            **Campos:**
            - **Nombre**: Identificador único (obligatorio)
            - **Correo**: Para notificaciones o contacto
            - **Rol**: Clasificación de la persona
            - **Umbral individual**: Nivel de confianza personalizado
            - **Notas**: Información adicional relevante
            """)
            
            st.success("""
            **Consejos:**
            - Usa nombres descriptivos y únicos
            - El umbral individual sobrescribe el global
            - Las notas son útiles para contexto adicional
            """)
    
    with subtab2:
        st.subheader("📋 Todas las Personas Registradas")
        
        personas_df = db.obtener_todas_personas()
        
        if not personas_df.empty:
            st.dataframe(
                personas_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "nombre": "Nombre",
                    "correo": "Correo",
                    "rol": "Rol",
                    "umbral_individual": st.column_config.NumberColumn(
                        "Umbral (%)",
                        format="%.0f%%",
                    ),
                    "total_detecciones": "Total Detecciones",
                    "ultima_visita": "Última Visita"
                }
            )
            
            st.caption(f"**Total de personas registradas:** {len(personas_df)}")
        else:
            st.info("📭 No hay personas registradas. Agrega la primera persona en la pestaña 'Agregar/Editar'")

# ==================== TAB 3: ESTADÍSTICAS POR PERSONA ====================
with tab3:
    st.header("📊 Estadísticas por Persona Detectada")
    
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
                
                st.subheader("📋 Detalle de Estadísticas")
                stats_display = stats.copy()
                stats_display['confianza_promedio'] = stats_display['confianza_promedio'].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(stats_display, use_container_width=True, hide_index=True)
            else:
                st.info(f"No hay estadísticas disponibles para {persona_seleccionada}")
    else:
        st.info("📭 Aún no hay personas detectadas. ¡Activa la cámara en la pestaña 'En Vivo'!")

# ==================== TAB 4: ANALÍTICA (5 GRÁFICAS) ====================
with tab4:
    st.header("📈 Análisis Global del Sistema")
    
    # GRÁFICA 1: Top 10 Personas
    st.subheader("1️⃣ Ranking de Personas Detectadas")
    personas_df = db.obtener_todas_personas()
    if not personas_df.empty:
        fig1 = px.bar(
            personas_df.head(10),
            x='nombre',
            y='total_detecciones',
            title='Top 10 Personas Más Detectadas',
            labels={'nombre': 'Persona', 'total_detecciones': 'Total Detecciones'},
            color='total_detecciones',
            color_continuous_scale='Blues',
            text='total_detecciones'
        )
        fig1.update_traces(textposition='outside')
        fig1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True, key="grafica1")
    else:
        st.info("No hay datos disponibles")
    
    st.markdown("---")
    
    # GRÁFICA 2: Actividad por Fecha
    st.subheader("2️⃣ Actividad por Fecha")
    detecciones_fecha = db.obtener_detecciones_por_fecha()
    if not detecciones_fecha.empty:
        fig2 = px.line(
            detecciones_fecha,
            x='fecha',
            y='total',
            color='etiqueta',
            title='Detecciones por Día',
            labels={'fecha': 'Fecha', 'total': 'Total', 'etiqueta': 'Persona'},
            markers=True
        )
        st.plotly_chart(fig2, use_container_width=True, key="grafica2")
    else:
        st.info("No hay datos de actividad por fecha")
    
    st.markdown("---")
    
    # GRÁFICA 3: Distribución de Confianza (Histograma)
    st.subheader("3️⃣ Distribución de Niveles de Confianza")
    distribucion_conf = db.obtener_distribucion_confianza()
    if not distribucion_conf.empty:
        fig3 = px.histogram(
            distribucion_conf,
            x='confianza',
            nbins=20,
            title='Distribución de Confianza en Detecciones',
            labels={'confianza': 'Nivel de Confianza', 'count': 'Frecuencia'},
            color_discrete_sequence=['#636EFA']
        )
        fig3.update_layout(
            xaxis_title="Nivel de Confianza",
            yaxis_title="Cantidad de Detecciones",
            bargap=0.1
        )
        st.plotly_chart(fig3, use_container_width=True, key="grafica3")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Confianza Media", f"{distribucion_conf['confianza'].mean()*100:.1f}%")
        with col2:
            st.metric("📈 Confianza Máxima", f"{distribucion_conf['confianza'].max()*100:.1f}%")
        with col3:
            st.metric("📉 Confianza Mínima", f"{distribucion_conf['confianza'].min()*100:.1f}%")
    else:
        st.info("No hay datos de confianza")
    
    st.markdown("---")
    
    # GRÁFICA 4: Detecciones por Fuente (Cámara vs Imagen)
    st.subheader("4️⃣ Comparativa por Fuente de Detección")
    detecciones_fuente = db.obtener_detecciones_por_fuente()
    if not detecciones_fuente.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig4a = px.pie(
                detecciones_fuente,
                values='total',
                names='fuente',
                title='Proporción de Detecciones por Fuente',
                color_discrete_map={'camara': '#00CC96', 'imagen': '#EF553B'}
            )
            st.plotly_chart(fig4a, use_container_width=True, key="grafica4a")
        
        with col2:
            fig4b = px.bar(
                detecciones_fuente,
                x='fuente',
                y='confianza_promedio',
                title='Confianza Promedio por Fuente',
                labels={'fuente': 'Fuente', 'confianza_promedio': 'Confianza Promedio'},
                color='fuente',
                color_discrete_map={'camara': '#00CC96', 'imagen': '#EF553B'},
                text='confianza_promedio'
            )
            fig4b.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            st.plotly_chart(fig4b, use_container_width=True, key="grafica4b")
    else:
        st.info("No hay datos por fuente")
    
    st.markdown("---")
    
    # GRÁFICA 5: Detecciones por Hora del Día
    st.subheader("5️⃣ Actividad por Hora del Día")
    detecciones_hora = db.obtener_detecciones_por_hora()
    if not detecciones_hora.empty:
        fig5 = px.bar(
            detecciones_hora,
            x='hora',
            y='total',
            title='Detecciones por Hora del Día (24h)',
            labels={'hora': 'Hora del Día', 'total': 'Total Detecciones'},
            color='total',
            color_continuous_scale='Viridis',
            text='total'
        )
        fig5.update_traces(textposition='outside')
        fig5.update_layout(
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1,
                range=[-0.5, 23.5]
            )
        )
        st.plotly_chart(fig5, use_container_width=True, key="grafica5")
        
        hora_pico = detecciones_hora.loc[detecciones_hora['total'].idxmax(), 'hora'] if len(detecciones_hora) > 0 else "N/A"
        st.info(f"🕐 **Hora pico de actividad:** {hora_pico}:00 hrs")
    else:
        st.info("No hay datos por hora")
    
    st.markdown("---")
    
    # Métricas Generales
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

# ==================== TAB 5: EXPORTAR DATOS ====================
with tab5:
    st.header("💾 Exportar Datos del Sistema")
    
    st.write("Descarga los datos almacenados en la base de datos en formato CSV o exporta las gráficas en ZIP.")
    
    # Exportar CSVs
    st.subheader("📄 Exportar Datos en CSV")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👥 Personas")
        personas_df = db.obtener_todas_personas()
        if not personas_df.empty:
            csv_personas = personas_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar Lista de Personas (CSV)",
                data=csv_personas,
                file_name=f"personas_detectadas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption(f"Total: {len(personas_df)} personas")
        else:
            st.info("No hay personas para exportar")
    
    with col2:
        st.markdown("#### 🔍 Detecciones Completas")
        detecciones_completas = db.obtener_todas_detecciones()
        if not detecciones_completas.empty:
            csv_detecciones = detecciones_completas.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar Todas las Detecciones (CSV)",
                data=csv_detecciones,
                file_name=f"detecciones_completas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption(f"Total: {len(detecciones_completas)} detecciones")
        else:
            st.info("No hay detecciones para exportar")
    
    st.markdown("---")
    
    # Exportar Gráficas en ZIP
    st.subheader("📊 Exportar Gráficas en ZIP")
    
    st.info("⚠️ Asegúrate de haber visitado la pestaña 'Analítica' para generar las gráficas antes de exportar")
    
    if st.button("📦 Generar ZIP con Gráficas", type="primary", use_container_width=True):
        with st.spinner("Generando gráficas y comprimiendo..."):
            try:
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    
                    # Gráfica 1: Top 10 Personas
                    if not personas_df.empty:
                        fig1 = px.bar(
                            personas_df.head(10),
                            x='nombre',
                            y='total_detecciones',
                            title='Top 10 Personas Más Detectadas',
                            color='total_detecciones',
                            color_continuous_scale='Blues'
                        )
                        img1 = fig1.to_image(format="png", width=1200, height=800)
                        zip_file.writestr("1_top_personas.png", img1)
                    
                    # Gráfica 2: Actividad por Fecha
                    detecciones_fecha = db.obtener_detecciones_por_fecha()
                    if not detecciones_fecha.empty:
                        fig2 = px.line(
                            detecciones_fecha,
                            x='fecha',
                            y='total',
                            color='etiqueta',
                            title='Detecciones por Día',
                            markers=True
                        )
                        img2 = fig2.to_image(format="png", width=1200, height=800)
                        zip_file.writestr("2_actividad_fecha.png", img2)
                    
                    # Gráfica 3: Distribución de Confianza
                    distribucion_conf = db.obtener_distribucion_confianza()
                    if not distribucion_conf.empty:
                        fig3 = px.histogram(
                            distribucion_conf,
                            x='confianza',
                            nbins=20,
                            title='Distribución de Confianza'
                        )
                        img3 = fig3.to_image(format="png", width=1200, height=800)
                        zip_file.writestr("3_distribucion_confianza.png", img3)
                    
                    # Gráfica 4: Detecciones por Fuente
                    detecciones_fuente = db.obtener_detecciones_por_fuente()
                    if not detecciones_fuente.empty:
                        fig4 = px.pie(
                            detecciones_fuente,
                            values='total',
                            names='fuente',
                            title='Proporción por Fuente'
                        )
                        img4 = fig4.to_image(format="png", width=1200, height=800)
                        zip_file.writestr("4_detecciones_fuente.png", img4)
                    
                    # Gráfica 5: Actividad por Hora
                    detecciones_hora = db.obtener_detecciones_por_hora()
                    if not detecciones_hora.empty:
                        fig5 = px.bar(
                            detecciones_hora,
                            x='hora',
                            y='total',
                            title='Actividad por Hora del Día',
                            color='total'
                        )
                        img5 = fig5.to_image(format="png", width=1200, height=800)
                        zip_file.writestr("5_actividad_hora.png", img5)
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label="⬇️ Descargar ZIP con Gráficas",
                    data=zip_buffer,
                    file_name=f"graficas_sistema_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
                
                st.success("✅ ZIP generado correctamente con 5 gráficas en formato PNG")
                
            except Exception as e:
                st.error(f"❌ Error al generar ZIP: {e}")
                st.info("💡 Nota: Para exportar gráficas necesitas tener instalado 'kaleido'. Instálalo con: pip install kaleido")
    
    st.markdown("---")
    
    # Vista previa de datos
    st.subheader("👁️ Vista Previa de Datos")
    
    with st.expander("Ver todas las detecciones registradas"):
        detecciones_completas = db.obtener_todas_detecciones()
        if not detecciones_completas.empty:
            st.dataframe(
                detecciones_completas,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "fecha_deteccion": "Fecha/Hora",
                    "fuente": "Fuente",
                    "etiqueta": "Persona",
                    "confianza": st.column_config.NumberColumn(
                        "Confianza",
                        format="%.2f%%",
                    ),
                    "correo": "Correo",
                    "rol": "Rol"
                }
            )
        else:
            st.info("No hay detecciones registradas")

# Footer
st.markdown("---")
st.caption("🤖 Sistema de Reconocimiento Automático con IA | Desarrollado con Streamlit, TensorFlow y SQLite")