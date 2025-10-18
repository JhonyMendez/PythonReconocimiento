import sqlite3
from datetime import datetime
import pandas as pd

class Database:
    def __init__(self, db_name='reconocimiento.db'):
        self.db_name = db_name
        self._migrated = False  # Flag para evitar migraciones múltiples
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Tabla personas con nuevos campos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS personas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL UNIQUE,
                correo TEXT,
                rol TEXT,
                umbral_individual REAL DEFAULT 0.95,
                notas TEXT,
                fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla detecciones con campo fuente
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detecciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                persona_id INTEGER,
                etiqueta TEXT NOT NULL,
                confianza REAL NOT NULL,
                fuente TEXT NOT NULL DEFAULT 'camara',
                fecha_deteccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (persona_id) REFERENCES personas(id)
            )
        ''')
        
        conn.commit()
        
        # MIGRACIÓN: Solo ejecutar una vez por instancia
        if not self._migrated:
            self._migrate_database(cursor)
            self._migrated = True
            conn.commit()
        
        conn.close()
    
    def _migrate_database(self, cursor):
        """Migra la base de datos agregando columnas faltantes"""
        try:
            # Verificar y agregar columnas en tabla personas
            cursor.execute("PRAGMA table_info(personas)")
            columnas_personas = [col[1] for col in cursor.fetchall()]
            
            columnas_a_agregar = {
                
                'rol': ('TEXT', None),
                'umbral_individual': ('REAL', 0.95),
                'notas': ('TEXT', None)
            }
            
            for columna, (tipo, default) in columnas_a_agregar.items():
                if columna not in columnas_personas:
                    try:
                        if default is not None:
                            sql = f'ALTER TABLE personas ADD COLUMN {columna} {tipo} DEFAULT {default}'
                        else:
                            sql = f'ALTER TABLE personas ADD COLUMN {columna} {tipo}'
                        cursor.execute(sql)
                        print(f"✅ Columna '{columna}' agregada a personas")
                    except sqlite3.OperationalError as e:
                        if "duplicate column" not in str(e).lower():
                            print(f"⚠️ Error al agregar '{columna}': {e}")
            
            # Verificar y agregar columnas en tabla detecciones
            cursor.execute("PRAGMA table_info(detecciones)")
            columnas_detecciones = [col[1] for col in cursor.fetchall()]
            
            if 'fuente' not in columnas_detecciones:
                try:
                    cursor.execute("ALTER TABLE detecciones ADD COLUMN fuente TEXT NOT NULL DEFAULT 'camara'")
                    print("✅ Columna 'fuente' agregada a detecciones")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        print(f"⚠️ Error al agregar 'fuente': {e}")
                    
        except Exception as e:
            # Silenciar errores de columnas duplicadas
            if "duplicate column" not in str(e).lower():
                print(f"⚠️ Error general en migración: {e}")
    
    def agregar_persona(self, nombre, correo=None, rol=None, umbral=0.95, notas=None):
        """Agregar o actualizar persona con todos los campos"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO personas (nombre, correo, rol, umbral_individual, notas) 
                VALUES (?, ?, ?, ?, ?)
            ''', (nombre, correo, rol, umbral, notas))
            conn.commit()
            persona_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            # Si ya existe, actualizar datos
            cursor.execute('''
                UPDATE personas 
                SET correo=?, rol=?, umbral_individual=?, notas=?
                WHERE nombre=?
            ''', (correo, rol, umbral, notas, nombre))
            conn.commit()
            cursor.execute('SELECT id FROM personas WHERE nombre = ?', (nombre,))
            persona_id = cursor.fetchone()[0]
        conn.close()
        return persona_id
    
    def registrar_deteccion(self, persona_nombre, confianza, fuente='camara'):
        """Registrar detección con fuente (camara o imagen)"""
        persona_id = self.agregar_persona(persona_nombre)
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO detecciones (persona_id, etiqueta, confianza, fuente) 
            VALUES (?, ?, ?, ?)
        ''', (persona_id, persona_nombre, confianza, fuente))
        conn.commit()
        conn.close()
        print(f"✅ Detección guardada: {persona_nombre} - Confianza: {confianza*100:.1f}% - Fuente: {fuente}")
    
    def obtener_persona(self, nombre):
        """Obtener datos completos de una persona"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, nombre, correo, rol, umbral_individual, notas, fecha_registro
            FROM personas WHERE nombre = ?
        ''', (nombre,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return {
                'id': result[0],
                'nombre': result[1],
                'correo': result[2],
                'rol': result[3],
                'umbral_individual': result[4],
                'notas': result[5],
                'fecha_registro': result[6]
            }
        return None
    
    def actualizar_persona(self, nombre_original, nombre_nuevo, correo, rol, umbral, notas):
        """Actualizar datos de una persona"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE personas 
                SET nombre=?, correo=?, rol=?, umbral_individual=?, notas=?
                WHERE nombre=?
            ''', (nombre_nuevo, correo, rol, umbral, notas, nombre_original))
            conn.commit()
            success = cursor.rowcount > 0
        except sqlite3.IntegrityError:
            success = False
        conn.close()
        return success
    
    def eliminar_persona(self, nombre):
        """Eliminar persona y sus detecciones"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM personas WHERE nombre = ?', (nombre,))
        result = cursor.fetchone()
        if result:
            persona_id = result[0]
            cursor.execute('DELETE FROM detecciones WHERE persona_id = ?', (persona_id,))
            cursor.execute('DELETE FROM personas WHERE id = ?', (persona_id,))
            conn.commit()
        conn.close()
    
    def obtener_todas_personas(self):
        """Obtener todas las personas con sus datos completos"""
        conn = sqlite3.connect(self.db_name)
        query = '''
            SELECT p.nombre, p.correo, p.rol, p.umbral_individual, 
                   COUNT(d.id) as total_detecciones,
                   MAX(d.fecha_deteccion) as ultima_visita
            FROM personas p LEFT JOIN detecciones d ON p.id = d.persona_id
            GROUP BY p.nombre ORDER BY total_detecciones DESC
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def obtener_estadisticas_persona(self, persona_nombre):
        conn = sqlite3.connect(self.db_name)
        query = '''
            SELECT d.etiqueta, COUNT(*) as total_detecciones,
                   AVG(d.confianza) as confianza_promedio,
                   MAX(d.fecha_deteccion) as ultima_deteccion
            FROM detecciones d JOIN personas p ON d.persona_id = p.id
            WHERE p.nombre = ?
            GROUP BY d.etiqueta ORDER BY total_detecciones DESC
        '''
        df = pd.read_sql_query(query, conn, params=(persona_nombre,))
        conn.close()
        return df
    
    def obtener_detecciones_por_fecha(self):
        """Obtener detecciones agrupadas por fecha"""
        conn = sqlite3.connect(self.db_name)
        query = '''
            SELECT DATE(d.fecha_deteccion) as fecha, p.nombre as etiqueta, COUNT(*) as total
            FROM detecciones d
            JOIN personas p ON d.persona_id = p.id
            GROUP BY DATE(d.fecha_deteccion), p.nombre
            ORDER BY fecha DESC
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def obtener_detecciones_por_fuente(self):
        """Obtener detecciones agrupadas por fuente"""
        conn = sqlite3.connect(self.db_name)
        query = '''
            SELECT fuente, COUNT(*) as total, AVG(confianza) as confianza_promedio
            FROM detecciones
            GROUP BY fuente
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def obtener_detecciones_por_hora(self):
        """Obtener detecciones agrupadas por hora del día"""
        conn = sqlite3.connect(self.db_name)
        query = '''
            SELECT CAST(strftime('%H', fecha_deteccion) AS INTEGER) as hora, 
                   COUNT(*) as total
            FROM detecciones
            GROUP BY hora
            ORDER BY hora
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def obtener_distribucion_confianza(self):
        """Obtener distribución de niveles de confianza"""
        conn = sqlite3.connect(self.db_name)
        query = 'SELECT confianza FROM detecciones'
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def obtener_datos_persona(self, persona_nombre):
        conn = sqlite3.connect(self.db_name)
        query = '''
            SELECT p.nombre, p.fecha_registro, COUNT(d.id) as total_visitas,
                   MIN(d.fecha_deteccion) as primera_deteccion,
                   MAX(d.fecha_deteccion) as ultima_deteccion
            FROM personas p LEFT JOIN detecciones d ON p.id = d.persona_id
            WHERE p.nombre = ? GROUP BY p.id
        '''
        cursor = conn.cursor()
        cursor.execute(query, (persona_nombre,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return {
                'nombre': result[0],
                'fecha_registro': result[1],
                'total_visitas': result[2],
                'primera_deteccion': result[3],
                'ultima_deteccion': result[4]
            }
        return None
    
    def obtener_todas_detecciones(self):
        """Obtener todas las detecciones para exportar"""
        conn = sqlite3.connect(self.db_name)
        query = '''
            SELECT d.fecha_deteccion, d.fuente, p.nombre as etiqueta, 
                   d.confianza, p.correo, p.rol
            FROM detecciones d
            JOIN personas p ON d.persona_id = p.id
            ORDER BY d.fecha_deteccion DESC
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df