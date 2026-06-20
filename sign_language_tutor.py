"""
=============================================================================
Tutor de Lenguaje de Señas Mexicano (LSM) - Aplicación Principal
=============================================================================

Sistema completo de reconocimiento y tutoría de Lenguaje de Señas Mexicano.
Incluye:
- Reconocimiento en tiempo real desde cámara
- Sistema de tutoría con lecciones estructuradas para LSM
- Agente conversacional para práctica de roleplay
- Soporte para alfabeto mexicano (incluye Ñ y RR)

OpenCV: 4.13.0 | MediaPipe: 0.10.31

Author: César Adrián Delgado Díaz
LinkedIn: https://www.linkedin.com/in/cesar-delgado-diaz
GitHub: https://github.com/cesar530

License: MIT
=============================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import json
import os
import logging
import time
import threading
import urllib.request

# Importaciones locales
from config import (
    CAMERA, MEDIAPIPE, DETECTION, UI, TUTOR, CONVERSATION,
    ALL_SIGNS, NUM_CLASSES, LESSONS_FILE, MESSAGES,
    PROJECT_ROOT, MODELS_DIR, USER_PROGRESS_DIR,
    get_sign_name, SignCategory, get_signs_by_category
)
from utils import (
    HandLandmarks, HandType, DetectionResult, LessonProgress,
    preprocess_frame, convert_bgr_to_rgb, enhance_contrast,
    extract_landmarks_from_results, landmarks_to_pixel_coords,
    create_feature_vector, draw_hand_landmarks, draw_prediction_info,
    draw_lesson_ui, load_lessons, save_progress, load_progress,
    smooth_predictions, generate_roleplay_response, get_feedback_message
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Hand Detector Class - MediaPipe Tasks API (0.10.31+)
# =============================================================================

# URL del modelo de detección de manos
HAND_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_LANDMARKER_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")


def download_model_if_needed():
    """Descarga el modelo de MediaPipe si no existe."""
    if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
        logger.info(f"Descargando modelo de MediaPipe Hand Landmarker...")
        os.makedirs(MODELS_DIR, exist_ok=True)
        urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, HAND_LANDMARKER_MODEL_PATH)
        logger.info(f"Modelo descargado en: {HAND_LANDMARKER_MODEL_PATH}")
    return HAND_LANDMARKER_MODEL_PATH


def load_model_as_bytes():
    """
    Carga el modelo como bytes para evitar problemas con rutas que contienen espacios.
    MediaPipe Tasks API tiene problemas con rutas que contienen espacios o caracteres especiales.
    """
    model_path = download_model_if_needed()
    with open(model_path, 'rb') as f:
        model_data = f.read()
    logger.info(f"Modelo cargado como bytes ({len(model_data)} bytes)")
    return model_data


class HandDetector:
    """
    Detector de manos usando MediaPipe Tasks API (0.10.31+).
    Maneja la detección y extracción de landmarks.
    """
    
    def __init__(self):
        """Inicializa el detector de MediaPipe usando Tasks API."""
        # Cargar modelo como bytes para evitar problemas con espacios en rutas
        model_data = load_model_as_bytes()
        
        # Configurar opciones del detector usando model_asset_buffer en lugar de path
        base_options = python.BaseOptions(model_asset_buffer=model_data)
        
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=MEDIAPIPE.max_num_hands,
            min_hand_detection_confidence=MEDIAPIPE.min_detection_confidence,
            min_hand_presence_confidence=MEDIAPIPE.min_tracking_confidence,
            min_tracking_confidence=MEDIAPIPE.min_tracking_confidence
        )
        
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_results = None
        
        logger.info("HandDetector (Tasks API) inicializado correctamente")
    
    def detect(self, frame: np.ndarray) -> Tuple[List[HandLandmarks], Any]:
        """
        Detecta manos en un frame.
        
        Args:
            frame: Frame BGR de OpenCV
            
        Returns:
            Tupla (lista de HandLandmarks, resultados raw de MediaPipe)
        """
        # Convertir a RGB para MediaPipe
        rgb_frame = convert_bgr_to_rgb(frame)
        
        # Crear imagen de MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detectar manos
        results = self.detector.detect(mp_image)
        self.last_results = results
        
        # Extraer landmarks usando la nueva estructura
        hands = self._extract_landmarks(results, frame.shape)
        
        return hands, results
    
    def _extract_landmarks(self, results, frame_shape: Tuple[int, int, int]) -> List[HandLandmarks]:
        """
        Extrae landmarks de los resultados de MediaPipe Tasks API.
        
        Args:
            results: Resultados de HandLandmarker
            frame_shape: Shape del frame (alto, ancho, canales)
            
        Returns:
            Lista de HandLandmarks detectados
        """
        hands = []
        
        if results.hand_landmarks:
            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                # Extraer coordenadas
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] 
                    for lm in hand_landmarks
                ])
                
                # Determinar tipo de mano
                if results.handedness and idx < len(results.handedness):
                    handedness = results.handedness[idx][0]
                    hand_label = handedness.category_name
                    confidence = handedness.score
                else:
                    hand_label = "Unknown"
                    confidence = 0.5
                
                hand_type = HandType.LEFT if hand_label == "Left" else HandType.RIGHT
                
                hands.append(HandLandmarks(
                    landmarks=landmarks,
                    hand_type=hand_type,
                    confidence=confidence
                ))
        
        return hands
    
    def draw_landmarks(
        self, 
        frame: np.ndarray, 
        results: Any,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        Dibuja landmarks en el frame usando dibujo manual (compatible con 0.10.31+).
        
        Args:
            frame: Frame BGR
            results: Resultados de MediaPipe Tasks
            draw_connections: Si dibujar conexiones
            
        Returns:
            Frame con landmarks dibujados
        """
        if results and results.hand_landmarks:
            h, w = frame.shape[:2]
            
            # Conexiones de la mano (índices de landmarks)
            HAND_CONNECTIONS = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Pulgar
                (0, 5), (5, 6), (6, 7), (7, 8),  # Índice
                (0, 9), (9, 10), (10, 11), (11, 12),  # Medio
                (0, 13), (13, 14), (14, 15), (15, 16),  # Anular
                (0, 17), (17, 18), (18, 19), (19, 20),  # Meñique
                (5, 9), (9, 13), (13, 17), (0, 17)  # Palma
            ]
            
            for hand_landmarks in results.hand_landmarks:
                # Convertir landmarks a coordenadas de píxel
                points = []
                for lm in hand_landmarks:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    points.append((px, py))
                
                # Dibujar conexiones
                if draw_connections:
                    for connection in HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(points) and end_idx < len(points):
                            cv2.line(frame, points[start_idx], points[end_idx], 
                                    (255, 255, 255), 2)
                
                # Dibujar puntos de landmarks
                for idx, point in enumerate(points):
                    # Color diferente para puntas de dedos
                    if idx in [4, 8, 12, 16, 20]:
                        color = (0, 255, 255)  # Amarillo para puntas
                        radius = 6
                    elif idx == 0:
                        color = (255, 0, 0)  # Azul para muñeca
                        radius = 8
                    else:
                        color = (0, 255, 0)  # Verde para el resto
                        radius = 4
                    
                    cv2.circle(frame, point, radius, color, -1)
                    cv2.circle(frame, point, radius, (0, 0, 0), 1)
        
        return frame
    
    def close(self):
        """Libera recursos."""
        if hasattr(self, 'detector'):
            self.detector.close()


# =============================================================================
# Sign Classifier Class
# =============================================================================

class SignClassifier:
    """
    Clasificador de señas usando características extraídas de landmarks.
    Implementa un clasificador simple basado en distancias y ángulos.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el clasificador.
        
        Args:
            model_path: Ruta al modelo entrenado (opcional)
        """
        self.model = None
        self.is_trained = False
        
        # Buffer de predicciones para suavizado
        self.prediction_buffer: deque = deque(maxlen=DETECTION.smoothing_window)
        self.confidence_buffer: deque = deque(maxlen=DETECTION.smoothing_window)
        
        # Intentar cargar modelo si existe
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            logger.info("Usando clasificador basado en reglas (sin modelo entrenado)")
            self._init_rule_based_classifier()
    
    def _init_rule_based_classifier(self):
        """Inicializa clasificador basado en reglas para demo."""
        # Patrones de referencia simplificados para algunas señas
        # En producción, estos serían aprendidos de datos reales
        self.reference_patterns = {
            "A": {"fingers_closed": True, "thumb_side": True},
            "B": {"fingers_extended": True, "thumb_tucked": True},
            "C": {"curved": True},
            "L": {"thumb_index_extended": True},
            "V": {"index_middle_extended": True},
            "Y": {"thumb_pinky_extended": True},
            "HELLO": {"wave_motion": True},
            "THANK_YOU": {"chin_forward": True},
        }
    
    def _load_model(self, path: str):
        """Carga modelo entrenado."""
        try:
            import joblib
            self.model = joblib.load(path)
            self.is_trained = True
            logger.info(f"Modelo cargado desde: {path}")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            self._init_rule_based_classifier()
    
    def predict(self, hand_landmarks: HandLandmarks) -> DetectionResult:
        """
        Predice la seña basándose en los landmarks.
        
        Args:
            hand_landmarks: Landmarks de la mano
            
        Returns:
            DetectionResult con predicción y confianza
        """
        # Extraer características
        features = create_feature_vector(hand_landmarks)
        
        if self.is_trained and self.model:
            # Usar modelo entrenado
            prediction = self.model.predict([features])[0]
            confidence = np.max(self.model.predict_proba([features]))
            sign = get_sign_name(prediction)
        else:
            # Usar clasificador basado en reglas
            sign, confidence = self._rule_based_predict(hand_landmarks)
        
        # Agregar al buffer
        self.prediction_buffer.append(sign)
        self.confidence_buffer.append(confidence)
        
        # Suavizar predicciones si está habilitado
        if DETECTION.prediction_smoothing:
            sign, confidence = smooth_predictions(
                list(self.prediction_buffer),
                list(self.confidence_buffer)
            )
        
        return DetectionResult(
            sign=sign,
            confidence=confidence,
            landmarks=hand_landmarks,
            timestamp=datetime.now()
        )
    
    def _rule_based_predict(
        self, 
        hand_landmarks: HandLandmarks
    ) -> Tuple[str, float]:
        """
        Clasificación basada en reglas para demo.
        
        Args:
            hand_landmarks: Landmarks de la mano
            
        Returns:
            Tupla (seña predicha, confianza)
        """
        landmarks = hand_landmarks.landmarks
        
        # Calcular características básicas
        finger_tips = [4, 8, 12, 16, 20]  # Puntas de dedos
        finger_bases = [2, 5, 9, 13, 17]  # Bases de dedos
        
        # Determinar qué dedos están extendidos
        fingers_extended = []
        for tip, base in zip(finger_tips[1:], finger_bases[1:]):  # Excluir pulgar
            extended = landmarks[tip][1] < landmarks[base][1]  # y menor = más arriba
            fingers_extended.append(extended)
        
        # Pulgar (lógica diferente por orientación)
        thumb_extended = landmarks[4][0] > landmarks[3][0]  # Para mano derecha
        if hand_landmarks.hand_type == HandType.LEFT:
            thumb_extended = landmarks[4][0] < landmarks[3][0]
        
        all_fingers = [thumb_extended] + fingers_extended
        num_extended = sum(all_fingers)
        
        # Clasificación simple por número de dedos
        if num_extended == 0:
            return "A", 0.75  # Puño cerrado
        elif num_extended == 1:
            if all_fingers[1]:  # Índice
                return "D", 0.70
            elif all_fingers[0]:  # Pulgar
                return "A", 0.65
        elif num_extended == 2:
            if all_fingers[1] and all_fingers[2]:  # Índice y medio
                return "V", 0.80
            elif all_fingers[0] and all_fingers[4]:  # Pulgar y meñique
                return "Y", 0.75
            elif all_fingers[0] and all_fingers[1]:  # Pulgar e índice
                return "L", 0.75
        elif num_extended == 3:
            if all_fingers[1] and all_fingers[2] and all_fingers[3]:
                return "W", 0.70
        elif num_extended == 4:
            if not all_fingers[0]:  # Pulgar escondido
                return "B", 0.75
        elif num_extended == 5:
            return "5", 0.80  # Número 5 o HELLO
        
        return "UNKNOWN", 0.50
    
    def reset_buffer(self):
        """Limpia el buffer de predicciones."""
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()


# =============================================================================
# Lesson Manager Class
# =============================================================================

class LessonManager:
    """
    Gestor de lecciones y progreso del usuario.
    """
    
    def __init__(self, lessons_path: str = None):
        """
        Inicializa el gestor de lecciones.
        
        Args:
            lessons_path: Ruta al archivo de lecciones (si no se especifica,
                          usa el archivo según el idioma activo)
        """
        # Importar configuración de idioma
        from config import ACTIVE_SIGN_LANGUAGE, SignLanguage, LESSONS_FILE_ASL, LESSONS_FILE_LSM
        
        # Seleccionar archivo de lecciones según idioma activo
        if lessons_path:
            self.lessons_path = lessons_path
        elif ACTIVE_SIGN_LANGUAGE == SignLanguage.LSM:
            self.lessons_path = str(LESSONS_FILE_LSM)
        else:
            self.lessons_path = str(LESSONS_FILE_ASL)
        
        self.lessons_data = load_lessons(self.lessons_path)
        
        self.current_lesson = None
        self.current_sign_index = 0
        self.score = 0
        self.attempts = 0
        self.correct_signs = []
        
        logger.info(f"LessonManager inicializado con {self._count_lessons()} lecciones")
    
    def _count_lessons(self) -> int:
        """Cuenta el total de lecciones."""
        total = 0
        for level in self.lessons_data.get("levels", []):
            total += len(level.get("lessons", []))
        return total
    
    def get_available_lessons(self) -> List[Dict]:
        """Retorna lista de lecciones disponibles."""
        lessons = []
        for level in self.lessons_data.get("levels", []):
            for lesson in level.get("lessons", []):
                lessons.append({
                    "id": lesson["id"],
                    "title": lesson["title"],
                    "level": level["name"],
                    "difficulty": lesson.get("difficulty", "unknown"),
                    "signs_count": len(lesson.get("signs", []))
                })
        return lessons
    
    def start_lesson(self, lesson_id: str) -> bool:
        """
        Inicia una lección específica.
        
        Args:
            lesson_id: ID de la lección
            
        Returns:
            True si se inició correctamente
        """
        # Buscar lección
        for level in self.lessons_data.get("levels", []):
            for lesson in level.get("lessons", []):
                if lesson["id"] == lesson_id:
                    self.current_lesson = lesson
                    self.current_sign_index = 0
                    self.score = 0
                    self.attempts = 0
                    self.correct_signs = []
                    logger.info(f"Lección iniciada: {lesson['title']}")
                    return True
        
        logger.warning(f"Lección no encontrada: {lesson_id}")
        return False
    
    def get_current_target(self) -> Optional[str]:
        """Obtiene la seña objetivo actual."""
        if not self.current_lesson:
            return None
        
        signs = self.current_lesson.get("signs", [])
        if self.current_sign_index < len(signs):
            return signs[self.current_sign_index]
        return None
    
    def check_sign(self, detected_sign: str, confidence: float) -> Dict:
        """
        Verifica si la seña detectada es correcta.
        
        Args:
            detected_sign: Seña detectada
            confidence: Nivel de confianza
            
        Returns:
            Diccionario con resultado de la verificación
        """
        target = self.get_current_target()
        
        if not target:
            return {"status": "no_target", "message": "No hay seña objetivo"}
        
        self.attempts += 1
        is_correct = detected_sign.upper() == target.upper() and confidence >= DETECTION.confidence_threshold
        
        result = {
            "target": target,
            "detected": detected_sign,
            "confidence": confidence,
            "is_correct": is_correct,
            "attempts": self.attempts,
            "feedback": get_feedback_message(is_correct, confidence, target)
        }
        
        if is_correct:
            self.correct_signs.append(target)
            self.score += 1
            
            # Avanzar a la siguiente seña
            self.current_sign_index += 1
            
            # Verificar si completó la lección
            signs = self.current_lesson.get("signs", [])
            if self.current_sign_index >= len(signs):
                result["lesson_complete"] = True
                result["final_score"] = (self.score / len(signs)) * 100
        
        return result
    
    def get_progress(self) -> Dict:
        """Obtiene el progreso actual de la lección."""
        if not self.current_lesson:
            return {"status": "no_lesson"}
        
        signs = self.current_lesson.get("signs", [])
        return {
            "lesson_title": self.current_lesson["title"],
            "current_index": self.current_sign_index,
            "total_signs": len(signs),
            "completed": self.correct_signs,
            "remaining": signs[self.current_sign_index:],
            "score": self.score,
            "attempts": self.attempts,
            "accuracy": (self.score / self.attempts * 100) if self.attempts > 0 else 0
        }


# =============================================================================
# Conversation Agent Class
# =============================================================================

class ConversationAgent:
    """
    Agente conversacional para práctica de roleplay.
    Soporta ASL (inglés) y LSM (español) según el idioma activo.
    """
    
    def __init__(self):
        """Inicializa el agente conversacional según el idioma activo."""
        # Importar configuración de idioma
        from config import ACTIVE_SIGN_LANGUAGE, SignLanguage
        
        self.active_language = ACTIVE_SIGN_LANGUAGE
        self.history: List[Dict] = []
        self.current_scenario: str = CONVERSATION.default_scenario
        self.scenarios = self._load_scenarios()
        
        lang_name = "LSM (Español)" if self.active_language == SignLanguage.LSM else "ASL (English)"
        logger.info(f"ConversationAgent inicializado para {lang_name}")
    
    def _load_scenarios(self) -> Dict:
        """Carga escenarios de conversación según el idioma activo."""
        from config import ACTIVE_SIGN_LANGUAGE, SignLanguage
        
        if ACTIVE_SIGN_LANGUAGE == SignLanguage.LSM:
            return self._load_lsm_scenarios()
        else:
            return self._load_asl_scenarios()
    
    def _load_asl_scenarios(self) -> Dict:
        """Carga escenarios de conversación para ASL (inglés)."""
        return {
            "greeting": {
                "name": "Greetings",
                "context": "You meet someone and want to say hello",
                "expected_signs": ["HELLO", "GOODBYE", "THANK_YOU", "YES", "NO"],
                "responses": {
                    "HELLO": ["Hello! How are you doing today?", "Hi there! Nice to see you!"],
                    "GOODBYE": ["Goodbye! Take care!", "See you later! Have a great day!"],
                    "THANK_YOU": ["You're welcome!", "No problem at all!"],
                    "YES": ["Great! That's wonderful!", "Perfect!"],
                    "NO": ["That's okay, no worries.", "Alright, maybe next time."],
                    "default": ["Interesting! Can you show me a greeting sign?"]
                }
            },
            "introduction": {
                "name": "Introductions",
                "context": "You're meeting someone new",
                "expected_signs": ["HELLO", "FRIEND", "LOVE"],
                "responses": {
                    "HELLO": ["Hello! Nice to meet you! What's your name?"],
                    "FRIEND": ["It's great to make a new friend!"],
                    "LOVE": ["That's so sweet! Spreading love is wonderful."],
                    "default": ["Tell me more about yourself using signs."]
                }
            },
            "asking_help": {
                "name": "Asking for Help",
                "context": "You need assistance at a store",
                "expected_signs": ["HELP", "PLEASE", "THANK_YOU", "SORRY"],
                "responses": {
                    "HELP": ["Of course! What do you need help with?"],
                    "PLEASE": ["Certainly, I'm happy to assist."],
                    "THANK_YOU": ["You're very welcome! Happy to help!"],
                    "SORRY": ["No need to apologize! It's all good."],
                    "default": ["If you need anything, show me the HELP sign."]
                }
            }
        }
    
    def _load_lsm_scenarios(self) -> Dict:
        """Carga escenarios de conversación para LSM (español)."""
        return {
            "greeting": {
                "name": "Saludos",
                "context": "Encuentras a alguien y quieres saludar en México",
                "expected_signs": ["HOLA", "ADIOS", "GRACIAS", "BUENOS_DIAS"],
                "responses": {
                    "HOLA": ["¡Hola! ¿Cómo estás hoy?", "¡Qué tal! Me alegra verte."],
                    "ADIOS": ["¡Hasta luego! Que te vaya bien.", "¡Adiós! Fue un gusto platicar."],
                    "GRACIAS": ["¡De nada! Es un placer.", "No hay de qué, cuando gustes."],
                    "BUENOS_DIAS": ["¡Buenos días! ¿Cómo amaneciste?", "¡Muy buenos días!"],
                    "BUENAS_NOCHES": ["¡Buenas noches! Que descanses."],
                    "default": ["Interesante. ¿Puedes mostrarme un saludo?"]
                }
            },
            "introduction": {
                "name": "Presentaciones",
                "context": "Conoces a alguien nuevo en México",
                "expected_signs": ["HOLA", "ME_LLAMO", "MUCHO_GUSTO"],
                "responses": {
                    "HOLA": ["¡Hola! Mucho gusto. ¿Cómo te llamas?"],
                    "ME_LLAMO": ["¡Qué nombre tan bonito! Yo me llamo Ana."],
                    "MUCHO_GUSTO": ["¡El gusto es mío! ¿De dónde eres?"],
                    "default": ["Cuéntame más sobre ti usando señas."]
                }
            },
            "asking_help": {
                "name": "Pidiendo Ayuda",
                "context": "Necesitas asistencia en una tienda mexicana",
                "expected_signs": ["AYUDA", "POR_FAVOR", "GRACIAS"],
                "responses": {
                    "AYUDA": ["¡Claro! ¿En qué puedo ayudarte?"],
                    "POR_FAVOR": ["Por supuesto, con mucho gusto."],
                    "GRACIAS": ["¡Siempre es un placer ayudar!"],
                    "default": ["Si necesitas algo, muéstrame la seña de AYUDA."]
                }
            },
            "familia": {
                "name": "Familia",
                "context": "Platicas sobre tu familia",
                "expected_signs": ["FAMILIA", "MAMA", "PAPA", "HERMANO", "AMOR"],
                "responses": {
                    "FAMILIA": ["¡Qué bonito! Cuéntame de tu familia."],
                    "MAMA": ["¿Cómo está tu mamá?"],
                    "PAPA": ["¿Y tu papá qué hace?"],
                    "HERMANO": ["¿Tienes hermanos? ¡Qué bien!"],
                    "AMOR": ["Se nota que quieres mucho a tu familia."],
                    "default": ["Cuéntame más de tu familia."]
                }
            },
            "necesidades": {
                "name": "Necesidades Básicas",
                "context": "Expresas necesidades básicas",
                "expected_signs": ["AGUA", "COMIDA", "CASA", "AYUDA"],
                "responses": {
                    "AGUA": ["¿Quieres agua? Aquí hay."],
                    "COMIDA": ["¿Tienes hambre? Vamos a comer."],
                    "CASA": ["¿Vamos a casa?"],
                    "AYUDA": ["¿Necesitas ayuda con algo?"],
                    "default": ["Dime qué necesitas."]
                }
            }
        }
    
    def get_language_info(self) -> Dict:
        """Retorna información del idioma activo del agente."""
        from config import ACTIVE_SIGN_LANGUAGE, SignLanguage
        if ACTIVE_SIGN_LANGUAGE == SignLanguage.LSM:
            return {"code": "LSM", "name": "Lengua de Señas Mexicana", "language": "Español"}
        return {"code": "ASL", "name": "American Sign Language", "language": "English"}
    
    def set_scenario(self, scenario_name: str) -> bool:
        """
        Establece el escenario de conversación.
        
        Args:
            scenario_name: Nombre del escenario
            
        Returns:
            True si se estableció correctamente
        """
        if scenario_name in self.scenarios:
            self.current_scenario = scenario_name
            self.history = []
            logger.info(f"Escenario establecido: {scenario_name}")
            return True
        return False
    
    def get_response(self, detected_sign: str) -> str:
        """
        Genera respuesta basada en la seña detectada.
        
        Args:
            detected_sign: Seña detectada
            
        Returns:
            Respuesta del agente
        """
        scenario = self.scenarios.get(self.current_scenario, {})
        responses = scenario.get("responses", {})
        
        # Buscar respuesta específica o usar default
        sign_responses = responses.get(detected_sign.upper(), responses.get("default", ["..."]))
        
        # Seleccionar respuesta (rotar entre opciones)
        import random
        response = random.choice(sign_responses)
        
        # Agregar al historial
        self.history.append({
            "user_sign": detected_sign,
            "agent_response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limitar historial
        if len(self.history) > CONVERSATION.history_length:
            self.history = self.history[-CONVERSATION.history_length:]
        
        return response
    
    def get_context(self) -> str:
        """Obtiene el contexto del escenario actual."""
        scenario = self.scenarios.get(self.current_scenario, {})
        return scenario.get("context", "Practica libremente")
    
    def get_expected_signs(self) -> List[str]:
        """Obtiene las señas esperadas para el escenario."""
        scenario = self.scenarios.get(self.current_scenario, {})
        return scenario.get("expected_signs", [])


# =============================================================================
# Main Sign Language Tutor Class
# =============================================================================

class SignLanguageTutor:
    """
    Clase principal que integra todos los componentes del tutor.
    """
    
    def __init__(self):
        """Inicializa el tutor de lenguaje de señas."""
        logger.info("=" * 60)
        logger.info("Iniciando Sign Language Tutor")
        logger.info("=" * 60)
        
        # Componentes principales
        self.detector = HandDetector()
        self.classifier = SignClassifier()
        self.lesson_manager = LessonManager()
        self.conversation_agent = ConversationAgent()
        
        # Estado
        self.is_running = False
        self.mode = "recognition"  # recognition, lesson, practice, conversation
        self.cap = None
        
        # Resultados recientes
        self.last_detection: Optional[DetectionResult] = None
        self.frame_count = 0
        
        logger.info("Sign Language Tutor inicializado correctamente")
    
    def _init_camera(self) -> bool:
        """
        Inicializa la cámara.
        
        Returns:
            True si se inicializó correctamente
        """
        self.cap = cv2.VideoCapture(CAMERA.index)
        
        if not self.cap.isOpened():
            logger.error(MESSAGES["camera_error"])
            return False
        
        # Configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.height)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA.fps)
        
        logger.info(f"Cámara inicializada: {CAMERA.width}x{CAMERA.height} @ {CAMERA.fps}fps")
        return True
    
    def start_recognition(self):
        """Inicia el modo de reconocimiento en tiempo real."""
        self.mode = "recognition"
        self._run_main_loop()
    
    def start_lesson(self, lesson_id: str):
        """
        Inicia el modo de lección.
        
        Args:
            lesson_id: ID de la lección a iniciar
        """
        if self.lesson_manager.start_lesson(lesson_id):
            self.mode = "lesson"
            self._run_main_loop()
        else:
            logger.error(f"No se pudo iniciar la lección: {lesson_id}")
    
    def start_practice(self, category: str = "alphabet"):
        """
        Inicia el modo de práctica libre.
        
        Args:
            category: Categoría de señas a practicar
        """
        self.mode = "practice"
        self._run_main_loop()
    
    def start_conversation(self, scenario: str = "greeting"):
        """
        Inicia el modo de conversación.
        
        Args:
            scenario: Escenario de roleplay
        """
        self.conversation_agent.set_scenario(scenario)
        self.mode = "conversation"
        self._run_main_loop()
    
    def _reinit_detector(self):
        """Reinicializa el detector de MediaPipe si fue cerrado."""
        try:
            # Intentar detectar para verificar si el detector está activo
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            self.detector.detect(test_frame)
        except Exception as e:
            logger.info("Reinicializando HandDetector...")
            self.detector = HandDetector()
            logger.info("HandDetector reinicializado correctamente")
    
    def _run_main_loop(self):
        """Ejecuta el loop principal de captura y procesamiento."""
        # Reinicializar detector si fue cerrado previamente
        self._reinit_detector()
        
        if not self._init_camera():
            return
        
        self.is_running = True
        logger.info(f"Iniciando modo: {self.mode}")
        
        # Crear ventana
        window_name = f"Sign Language Tutor - {self.mode.upper()}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        last_detection_time = 0
        cooldown_active = False
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Error leyendo frame de cámara")
                    continue
                
                self.frame_count += 1
                
                # Preprocesar frame
                frame = preprocess_frame(frame, CAMERA.resolution)
                
                # Detectar manos
                hands, results = self.detector.detect(frame)
                
                # Dibujar landmarks
                frame = self.detector.draw_landmarks(frame, results)
                
                # Procesar detección si hay manos
                if hands:
                    current_time = time.time()
                    
                    # Verificar cooldown
                    if current_time - last_detection_time >= DETECTION.cooldown_frames / CAMERA.fps:
                        # Clasificar seña (usar primera mano detectada)
                        detection = self.classifier.predict(hands[0])
                        self.last_detection = detection
                        
                        if detection.confidence >= DETECTION.confidence_threshold:
                            last_detection_time = current_time
                            
                            # Procesar según modo
                            frame = self._process_detection(frame, detection)
                
                # Dibujar UI según modo
                frame = self._draw_mode_ui(frame)
                
                # Mostrar frame
                cv2.imshow(window_name, frame)
                
                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q o ESC
                    break
                elif key == ord('r'):  # Reset
                    self.classifier.reset_buffer()
                elif key == ord('m'):  # Cambiar modo
                    self._cycle_mode()
        
        finally:
            self._cleanup()
    
    def _process_detection(
        self, 
        frame: np.ndarray, 
        detection: DetectionResult
    ) -> np.ndarray:
        """
        Procesa la detección según el modo actual.
        
        Args:
            frame: Frame actual
            detection: Resultado de detección
            
        Returns:
            Frame procesado
        """
        if self.mode == "recognition":
            frame = draw_prediction_info(
                frame,
                detection.sign,
                detection.confidence
            )
            
        elif self.mode == "lesson":
            result = self.lesson_manager.check_sign(
                detection.sign,
                detection.confidence
            )
            
            progress = self.lesson_manager.get_progress()
            target = self.lesson_manager.get_current_target() or "COMPLETADO"
            
            frame = draw_lesson_ui(
                frame,
                detection.sign,
                target,
                progress.get("accuracy", 0),
                progress.get("attempts", 0),
                result.get("is_correct", False)
            )
            
            if result.get("lesson_complete"):
                logger.info(f"¡Lección completada! Score: {result['final_score']:.1f}%")
                
        elif self.mode == "conversation":
            response = self.conversation_agent.get_response(detection.sign)
            
            # Dibujar respuesta del agente
            frame = self._draw_conversation_ui(frame, detection.sign, response)
        
        return frame
    
    def _draw_mode_ui(self, frame: np.ndarray) -> np.ndarray:
        """
        Dibuja elementos de UI específicos del modo.
        
        Args:
            frame: Frame actual
            
        Returns:
            Frame con UI
        """
        h, w = frame.shape[:2]
        
        # Barra inferior con información del modo
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 40), (w, h), UI.color_background, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Texto de modo
        mode_text = f"Modo: {self.mode.upper()} | Q: Salir | R: Reset | M: Cambiar modo"
        cv2.putText(frame, mode_text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI.color_text, 1, cv2.LINE_AA)
        
        # Si no hay detección, mostrar mensaje
        if not self.last_detection:
            cv2.putText(frame, MESSAGES["no_hand_detected"], (w // 2 - 150, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, UI.color_warning, 2, cv2.LINE_AA)
        
        return frame
    
    def _draw_conversation_ui(
        self, 
        frame: np.ndarray, 
        detected_sign: str, 
        response: str
    ) -> np.ndarray:
        """
        Dibuja UI de conversación.
        
        Args:
            frame: Frame actual
            detected_sign: Seña detectada
            response: Respuesta del agente
            
        Returns:
            Frame con UI de conversación
        """
        h, w = frame.shape[:2]
        
        # Panel de conversación
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 150), UI.color_background, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Contexto del escenario
        context = self.conversation_agent.get_context()
        cv2.putText(frame, f"Escenario: {context[:50]}...", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, UI.color_secondary, 1, cv2.LINE_AA)
        
        # Seña del usuario
        cv2.putText(frame, f"Tu seña: {detected_sign}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, UI.color_primary, 2, cv2.LINE_AA)
        
        # Respuesta del agente
        cv2.putText(frame, f"Agente: {response[:40]}...", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI.color_text, 2, cv2.LINE_AA)
        
        # Señas esperadas
        expected = self.conversation_agent.get_expected_signs()
        expected_text = f"Prueba: {', '.join(expected[:3])}"
        cv2.putText(frame, expected_text, (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI.color_warning, 1, cv2.LINE_AA)
        
        return frame
    
    def _cycle_mode(self):
        """Cicla entre modos disponibles."""
        modes = ["recognition", "lesson", "practice", "conversation"]
        current_index = modes.index(self.mode)
        self.mode = modes[(current_index + 1) % len(modes)]
        
        self.classifier.reset_buffer()
        logger.info(f"Modo cambiado a: {self.mode}")
    
    def _cleanup(self):
        """Limpia recursos (cámara y ventanas, pero mantiene el detector reutilizable)."""
        self.is_running = False
        
        # Liberar cámara
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Cerrar ventanas de OpenCV
        cv2.destroyAllWindows()
        
        # Esperar un momento para asegurar que las ventanas se cierren
        cv2.waitKey(1)
        
        # Nota: No cerramos el detector aquí para permitir reiniciar el reconocimiento
        # El detector se reinicializará si es necesario en _reinit_detector()
        
        logger.info("Recursos liberados correctamente")
    
    def get_available_lessons(self) -> List[Dict]:
        """Retorna lecciones disponibles."""
        return self.lesson_manager.get_available_lessons()
    
    def get_available_scenarios(self) -> List[str]:
        """Retorna escenarios de conversación disponibles."""
        return list(self.conversation_agent.scenarios.keys())


# =============================================================================
# Funciones de Demo y Testing
# =============================================================================

def demo_recognition():
    """Demo de reconocimiento básico de LSM."""
    print("\n" + "=" * 60)
    print("🤟 DEMO: Reconocimiento de Lenguaje de Señas Mexicano (LSM)")
    print("=" * 60)
    print("\nControles:")
    print("  Q/ESC - Salir")
    print("  R     - Reset buffer de predicciones")
    print("  M     - Cambiar modo")
    print("\nMuestra tu mano frente a la cámara...")
    print("Alfabeto LSM incluye: A-Z, Ñ, RR")
    print("=" * 60 + "\n")
    
    tutor = SignLanguageTutor()
    tutor.start_recognition()


def demo_lesson():
    """Demo de lección LSM."""
    print("\n" + "=" * 60)
    print("🤟 DEMO: Modo Lección - LSM")
    print("=" * 60)
    
    tutor = SignLanguageTutor()
    
    # Mostrar lecciones disponibles
    lessons = tutor.get_available_lessons()
    print("\nLecciones disponibles de LSM:")
    for i, lesson in enumerate(lessons, 1):
        print(f"  {i}. {lesson['title']} ({lesson['level']})")
    
    # Iniciar primera lección
    if lessons:
        print(f"\nIniciando: {lessons[0]['title']}")
        tutor.start_lesson(lessons[0]['id'])


def demo_conversation():
    """Demo de conversación en LSM."""
    print("\n" + "=" * 60)
    print("🤟 DEMO: Modo Conversación - LSM")
    print("=" * 60)
    
    tutor = SignLanguageTutor()
    
    # Mostrar escenarios
    scenarios = tutor.get_available_scenarios()
    print("\nEscenarios disponibles:")
    for scenario in scenarios:
        print(f"  - {scenario}")
    
    print("\nIniciando escenario: greeting (Saludos)")
    tutor.start_conversation("greeting")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Punto de entrada principal."""
    print("\n" + "=" * 70)
    print("    🤟 TUTOR DE LENGUAJE DE SEÑAS MEXICANO (LSM) 🇲🇽")
    print("=" * 70)
    print("\n    OpenCV: 4.13.0 | MediaPipe: 0.10.31")
    print("    Author: César Adrián Delgado Díaz")
    print("    GitHub: github.com/cesar530")
    print("    License: MIT")
    print("\n" + "=" * 70)
    
    print("\nSelecciona un modo:")
    print("  1. Reconocimiento en tiempo real")
    print("  2. Lección interactiva")
    print("  3. Conversación (Roleplay)")
    print("  4. Salir")
    
    try:
        choice = input("\nOpción (1-4): ").strip()
        
        if choice == "1":
            demo_recognition()
        elif choice == "2":
            demo_lesson()
        elif choice == "3":
            demo_conversation()
        elif choice == "4":
            print("\n¡Hasta luego! 👋")
        else:
            print("Opción no válida. Iniciando reconocimiento por defecto...")
            demo_recognition()
            
    except KeyboardInterrupt:
        print("\n\nPrograma interrumpido por el usuario.")
    except Exception as e:
        logger.error(f"Error en ejecución: {e}")
        raise


if __name__ == "__main__":
    main()
