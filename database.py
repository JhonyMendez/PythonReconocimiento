import sqlite3
from datetime import datetime
import pandas as pd

class Database:
    def __init__(self, db_name='reconocimiento.db'):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS personas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL UNIQUE,
                fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detecciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                persona_id INTEGER,
                etiqueta TEXT NOT NULL,
                confianza REAL NOT NULL,
                fecha_deteccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (persona_id) REFERENCES personas(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def agregar_persona(self, nombre):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO personas (nombre) VALUES (?)', (nombre,))
            conn.commit()
            persona_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor.execute('SELECT id FROM personas WHERE nombre = ?', (nombre,))
            persona_id = cursor.fetchone()[0]
        conn.close()
        return persona_id
    
    def registrar_deteccion(self, persona_nombre, confianza):
        """CORREGIDO: Ahora acepta solo 2 parámetros"""
        persona_id = self.agregar_persona(persona_nombre)
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO detecciones (persona_id, etiqueta, confianza) VALUES (?, ?, ?)',
                      (persona_id, persona_nombre, confianza))  # etiqueta = persona_nombre
        conn.commit()
        conn.close()
        print(f"✅ Detección guardada: {persona_nombre} - Confianza: {confianza*100:.1f}%")
    
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
    
    def obtener_todas_personas(self):
        conn = sqlite3.connect(self.db_name)
        query = '''
            SELECT p.nombre, COUNT(d.id) as total_detecciones,
                   MAX(d.fecha_deteccion) as ultima_visita
            FROM personas p LEFT JOIN detecciones d ON p.id = d.persona_id
            GROUP BY p.nombre ORDER BY total_detecciones DESC
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def obtener_detecciones_por_fecha(self):
        """CORREGIDO: Ahora une con la tabla personas para obtener el nombre"""
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
            return {'nombre': result[0], 'fecha_registro': result[1], 'total_visitas': result[2],
                    'primera_deteccion': result[3], 'ultima_deteccion': result[4]}
        return None