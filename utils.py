"""
=============================================================================
Tutor de Lenguaje de Señas Mexicano (LSM) - Funciones de Utilidad
=============================================================================

Funciones auxiliares para el reconocimiento de lenguaje de señas,
procesamiento de landmarks y utilidades generales.
Compatible con MediaPipe Tasks API (0.10.31+)

Author: César Adrián Delgado Díaz
LinkedIn: https://www.linkedin.com/in/cesar-delgado-diaz
GitHub: https://github.com/cesar530

License: MIT
=============================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import json
import os
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes y Enums
# =============================================================================

class HandType(Enum):
    """Tipo de mano detectada."""
    LEFT = "Left"
    RIGHT = "Right"
    UNKNOWN = "Unknown"


@dataclass
class HandLandmarks:
    """Estructura para almacenar landmarks de una mano."""
    landmarks: np.ndarray  # Shape: (21, 3) - 21 puntos, 3 coordenadas (x, y, z)
    hand_type: HandType
    confidence: float
    
    def to_flat_array(self) -> np.ndarray:
        """Convierte landmarks a array plano para clasificación."""
        return self.landmarks.flatten()
    
    def normalize(self) -> 'HandLandmarks':
        """Normaliza landmarks respecto a la muñeca (punto 0)."""
        wrist = self.landmarks[0]
        normalized = self.landmarks - wrist
        
        # Escalar por la distancia máxima para invariancia de escala
        max_dist = np.max(np.linalg.norm(normalized, axis=1))
        if max_dist > 0:
            normalized = normalized / max_dist
            
        return HandLandmarks(
            landmarks=normalized,
            hand_type=self.hand_type,
            confidence=self.confidence
        )


@dataclass
class DetectionResult:
    """Resultado de detección de señas."""
    sign: str
    confidence: float
    landmarks: Optional[HandLandmarks]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte resultado a diccionario."""
        return {
            "sign": self.sign,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "hand_type": self.landmarks.hand_type.value if self.landmarks else None
        }


@dataclass
class LessonProgress:
    """Progreso de una lección."""
    lesson_id: str
    completed_signs: List[str]
    accuracy_scores: Dict[str, float]
    attempts: int
    best_score: float
    last_practice: datetime
    
    def calculate_overall_accuracy(self) -> float:
        """Calcula precisión promedio."""
        if not self.accuracy_scores:
            return 0.0
        return np.mean(list(self.accuracy_scores.values()))


# =============================================================================
# Funciones de Procesamiento de Imagen
# =============================================================================

def preprocess_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (640, 480)
) -> np.ndarray:
    """
    Preprocesa un frame para detección de manos.
    
    Args:
        frame: Frame BGR de OpenCV
        target_size: Tamaño objetivo (ancho, alto)
        
    Returns:
        Frame preprocesado
    """
    # Redimensionar si es necesario
    if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
        frame = cv2.resize(frame, target_size)
    
    # Voltear horizontalmente (efecto espejo)
    frame = cv2.flip(frame, 1)
    
    return frame


def convert_bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convierte frame de BGR a RGB para MediaPipe."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def enhance_contrast(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Mejora el contraste usando CLAHE.
    
    Args:
        frame: Frame en formato BGR
        clip_limit: Límite de contraste
        grid_size: Tamaño de grid para CLAHE
        
    Returns:
        Frame con contraste mejorado
    """
    # Convertir a LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplicar CLAHE al canal L
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l)
    
    # Reconstruir imagen
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


# =============================================================================
# Funciones de Extracción de Landmarks
# =============================================================================

def extract_landmarks_from_results(
    results,
    frame_shape: Tuple[int, int, int]
) -> List[HandLandmarks]:
    """
    Extrae landmarks de los resultados de MediaPipe.
    Compatible con MediaPipe Tasks API (0.10.31+) y API legacy.
    
    Args:
        results: Resultados de MediaPipe Hands (Tasks API o Legacy)
        frame_shape: Shape del frame (alto, ancho, canales)
        
    Returns:
        Lista de HandLandmarks detectados
    """
    hands = []
    
    # Verificar si es la nueva Tasks API (tiene hand_landmarks)
    if hasattr(results, 'hand_landmarks') and results.hand_landmarks:
        # Nueva API de MediaPipe Tasks (0.10.31+)
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
    
    # API Legacy (multi_hand_landmarks) - fallback para compatibilidad
    elif hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
        if results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # Extraer coordenadas
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] 
                    for lm in hand_landmarks.landmark
                ])
                
                # Determinar tipo de mano
                hand_label = handedness.classification[0].label
                hand_type = HandType.LEFT if hand_label == "Left" else HandType.RIGHT
                confidence = handedness.classification[0].score
                
                hands.append(HandLandmarks(
                    landmarks=landmarks,
                    hand_type=hand_type,
                    confidence=confidence
                ))
    
    return hands


def landmarks_to_pixel_coords(
    landmarks: np.ndarray,
    frame_width: int,
    frame_height: int
) -> np.ndarray:
    """
    Convierte landmarks normalizados a coordenadas de píxel.
    
    Args:
        landmarks: Array de landmarks (21, 3) con valores 0-1
        frame_width: Ancho del frame
        frame_height: Alto del frame
        
    Returns:
        Array de coordenadas en píxeles (21, 2)
    """
    pixel_coords = np.zeros((21, 2), dtype=np.int32)
    pixel_coords[:, 0] = (landmarks[:, 0] * frame_width).astype(np.int32)
    pixel_coords[:, 1] = (landmarks[:, 1] * frame_height).astype(np.int32)
    return pixel_coords


def calculate_hand_bounding_box(
    landmarks: np.ndarray,
    frame_width: int,
    frame_height: int,
    padding: float = 0.1
) -> Tuple[int, int, int, int]:
    """
    Calcula bounding box de la mano con padding.
    
    Args:
        landmarks: Landmarks normalizados
        frame_width: Ancho del frame
        frame_height: Alto del frame
        padding: Porcentaje de padding adicional
        
    Returns:
        Tuple (x_min, y_min, x_max, y_max)
    """
    x_coords = landmarks[:, 0] * frame_width
    y_coords = landmarks[:, 1] * frame_height
    
    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
    
    # Agregar padding
    width = x_max - x_min
    height = y_max - y_min
    pad_x = int(width * padding)
    pad_y = int(height * padding)
    
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(frame_width, x_max + pad_x)
    y_max = min(frame_height, y_max + pad_y)
    
    return (x_min, y_min, x_max, y_max)


# =============================================================================
# Funciones de Feature Engineering
# =============================================================================

def extract_finger_angles(landmarks: np.ndarray) -> np.ndarray:
    """
    Extrae ángulos entre segmentos de dedos.
    
    Args:
        landmarks: Array de landmarks (21, 3)
        
    Returns:
        Array de ángulos para cada dedo
    """
    # Índices de joints por dedo
    fingers = {
        'thumb': [0, 1, 2, 3, 4],
        'index': [0, 5, 6, 7, 8],
        'middle': [0, 9, 10, 11, 12],
        'ring': [0, 13, 14, 15, 16],
        'pinky': [0, 17, 18, 19, 20]
    }
    
    angles = []
    
    for finger_name, indices in fingers.items():
        for i in range(len(indices) - 2):
            p1 = landmarks[indices[i]]
            p2 = landmarks[indices[i + 1]]
            p3 = landmarks[indices[i + 2]]
            
            # Calcular vectores
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calcular ángulo
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(angle)
    
    return np.array(angles)


def extract_finger_distances(landmarks: np.ndarray) -> np.ndarray:
    """
    Extrae distancias entre puntas de dedos.
    
    Args:
        landmarks: Array de landmarks (21, 3)
        
    Returns:
        Array de distancias entre puntas
    """
    # Índices de puntas de dedos
    tips = [4, 8, 12, 16, 20]
    
    distances = []
    
    # Distancias entre todas las puntas
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            dist = np.linalg.norm(landmarks[tips[i]] - landmarks[tips[j]])
            distances.append(dist)
    
    # Distancias de cada punta a la palma (punto 0)
    for tip in tips:
        dist = np.linalg.norm(landmarks[tip] - landmarks[0])
        distances.append(dist)
    
    return np.array(distances)


def create_feature_vector(hand_landmarks: HandLandmarks) -> np.ndarray:
    """
    Crea vector de características completo para clasificación.
    
    Args:
        hand_landmarks: HandLandmarks normalizado
        
    Returns:
        Vector de características concatenado
    """
    # Normalizar landmarks
    normalized = hand_landmarks.normalize()
    
    # Extraer diferentes características
    flat_landmarks = normalized.to_flat_array()  # 63 features (21 * 3)
    angles = extract_finger_angles(normalized.landmarks)  # 15 features
    distances = extract_finger_distances(normalized.landmarks)  # 15 features
    
    # Concatenar todas las características
    feature_vector = np.concatenate([
        flat_landmarks,
        angles,
        distances
    ])
    
    return feature_vector


# =============================================================================
# Funciones de Visualización
# =============================================================================

def draw_hand_landmarks(
    frame: np.ndarray,
    landmarks: np.ndarray,
    connections: List[Tuple[int, int]] = None,
    landmark_color: Tuple[int, int, int] = (0, 255, 0),
    connection_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Dibuja landmarks de mano en el frame.
    
    Args:
        frame: Frame BGR
        landmarks: Landmarks en coordenadas de píxel (21, 2)
        connections: Lista de conexiones entre landmarks
        landmark_color: Color de puntos (BGR)
        connection_color: Color de líneas (BGR)
        thickness: Grosor de líneas
        
    Returns:
        Frame con landmarks dibujados
    """
    if connections is None:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Pulgar
            (0, 5), (5, 6), (6, 7), (7, 8),  # Índice
            (0, 9), (9, 10), (10, 11), (11, 12),  # Medio
            (0, 13), (13, 14), (14, 15), (15, 16),  # Anular
            (0, 17), (17, 18), (18, 19), (19, 20),  # Meñique
            (5, 9), (9, 13), (13, 17)  # Palma
        ]
    
    frame = frame.copy()
    
    # Dibujar conexiones
    for start, end in connections:
        pt1 = tuple(landmarks[start])
        pt2 = tuple(landmarks[end])
        cv2.line(frame, pt1, pt2, connection_color, thickness)
    
    # Dibujar puntos
    for i, point in enumerate(landmarks):
        cv2.circle(frame, tuple(point), 5, landmark_color, -1)
        cv2.circle(frame, tuple(point), 5, (0, 0, 0), 1)
    
    return frame


def draw_prediction_info(
    frame: np.ndarray,
    prediction: str,
    confidence: float,
    position: Tuple[int, int] = (20, 50),
    font_scale: float = 1.2,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Dibuja información de predicción en el frame.
    
    Args:
        frame: Frame BGR
        prediction: Seña predicha
        confidence: Confianza de la predicción
        position: Posición del texto
        font_scale: Tamaño de fuente
        color: Color del texto (BGR)
        
    Returns:
        Frame con información
    """
    frame = frame.copy()
    
    # Fondo semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Texto de predicción
    text = f"Seña: {prediction}"
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, 2, cv2.LINE_AA)
    
    # Barra de confianza
    bar_width = int(250 * confidence)
    bar_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 0, 255)
    cv2.rectangle(frame, (20, 70), (20 + bar_width, 90), bar_color, -1)
    cv2.rectangle(frame, (20, 70), (270, 90), (255, 255, 255), 1)
    
    # Porcentaje
    conf_text = f"{confidence * 100:.1f}%"
    cv2.putText(frame, conf_text, (280, 85), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame


def draw_lesson_ui(
    frame: np.ndarray,
    current_sign: str,
    target_sign: str,
    score: float,
    attempts: int,
    is_correct: bool
) -> np.ndarray:
    """
    Dibuja interfaz de lección en el frame.
    
    Args:
        frame: Frame BGR
        current_sign: Seña detectada actualmente
        target_sign: Seña objetivo de la lección
        score: Puntuación actual
        attempts: Número de intentos
        is_correct: Si la seña es correcta
        
    Returns:
        Frame con UI de lección
    """
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Panel superior
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Seña objetivo
    cv2.putText(frame, f"OBJETIVO: {target_sign}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Seña detectada
    color = (0, 255, 0) if is_correct else (0, 0, 255)
    cv2.putText(frame, f"Detectado: {current_sign}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    
    # Score y intentos
    cv2.putText(frame, f"Score: {score:.1f}%", (w - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Intentos: {attempts}", (w - 200, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Indicador de éxito
    if is_correct:
        cv2.putText(frame, "✓ CORRECTO!", (w // 2 - 100, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
    
    return frame


# =============================================================================
# Funciones de Utilidad General
# =============================================================================

def load_lessons(filepath: str) -> Dict[str, Any]:
    """
    Carga lecciones desde archivo JSON.
    
    Args:
        filepath: Ruta al archivo de lecciones
        
    Returns:
        Diccionario con estructura de lecciones
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Archivo de lecciones no encontrado: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error al parsear JSON: {e}")
        return {}


def save_progress(
    progress: LessonProgress,
    filepath: str
) -> bool:
    """
    Guarda progreso de lección.
    
    Args:
        progress: Objeto LessonProgress
        filepath: Ruta para guardar
        
    Returns:
        True si se guardó correctamente
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            "lesson_id": progress.lesson_id,
            "completed_signs": progress.completed_signs,
            "accuracy_scores": progress.accuracy_scores,
            "attempts": progress.attempts,
            "best_score": progress.best_score,
            "last_practice": progress.last_practice.isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Progreso guardado en: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error al guardar progreso: {e}")
        return False


def load_progress(filepath: str) -> Optional[LessonProgress]:
    """
    Carga progreso de lección.
    
    Args:
        filepath: Ruta del archivo de progreso
        
    Returns:
        LessonProgress o None si no existe
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return LessonProgress(
            lesson_id=data["lesson_id"],
            completed_signs=data["completed_signs"],
            accuracy_scores=data["accuracy_scores"],
            attempts=data["attempts"],
            best_score=data["best_score"],
            last_practice=datetime.fromisoformat(data["last_practice"])
        )
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Error al cargar progreso: {e}")
        return None


def calculate_confidence_threshold(
    predictions: List[float],
    base_threshold: float = 0.7
) -> float:
    """
    Calcula umbral de confianza dinámico basado en predicciones recientes.
    
    Args:
        predictions: Lista de confianzas recientes
        base_threshold: Umbral base
        
    Returns:
        Umbral ajustado
    """
    if not predictions:
        return base_threshold
    
    mean_conf = np.mean(predictions)
    std_conf = np.std(predictions)
    
    # Ajustar umbral basado en variabilidad
    adjusted = base_threshold - (std_conf * 0.1) + (mean_conf - 0.5) * 0.2
    
    return np.clip(adjusted, 0.5, 0.95)


def smooth_predictions(
    predictions: List[str],
    confidences: List[float],
    window_size: int = 5
) -> Tuple[str, float]:
    """
    Suaviza predicciones usando ventana deslizante y votación.
    
    Args:
        predictions: Lista de predicciones recientes
        confidences: Lista de confianzas correspondientes
        window_size: Tamaño de ventana
        
    Returns:
        Tupla (predicción_suavizada, confianza_promedio)
    """
    if not predictions:
        return "", 0.0
    
    recent = predictions[-window_size:]
    recent_conf = confidences[-window_size:]
    
    # Votación ponderada por confianza
    votes = {}
    for pred, conf in zip(recent, recent_conf):
        if pred not in votes:
            votes[pred] = 0
        votes[pred] += conf
    
    # Obtener predicción con más votos
    winner = max(votes, key=votes.get)
    
    # Calcular confianza promedio del ganador
    winner_confs = [c for p, c in zip(recent, recent_conf) if p == winner]
    avg_conf = np.mean(winner_confs) if winner_confs else 0.0
    
    return winner, avg_conf


# =============================================================================
# Funciones de Conversación
# =============================================================================

def generate_roleplay_response(
    detected_sign: str,
    scenario: str,
    conversation_history: List[Dict[str, str]]
) -> str:
    """
    Genera respuesta para roleplay conversacional.
    
    Args:
        detected_sign: Seña detectada
        scenario: Escenario de roleplay
        conversation_history: Historial de conversación
        
    Returns:
        Respuesta del agente
    """
    # Respuestas predefinidas por escenario (LSM - Español Mexicano)
    responses = {
        "greeting": {
            "HOLA": "¡Hola! Me alegra saludarte. ¿Cómo estás hoy?",
            "ADIOS": "¡Hasta luego! Fue un gusto practicar contigo.",
            "GRACIAS": "¡De nada! Es un placer ayudarte a aprender LSM.",
            "SI": "¡Genial! Me alegra que estés de acuerdo.",
            "NO": "Entiendo. ¿Hay algo más que te gustaría practicar?",
            "BUENOS_DIAS": "¡Buenos días! ¿Cómo amaneciste?",
            "BUENAS_NOCHES": "¡Buenas noches! Que descanses bien.",
            # Compatibilidad con señas en inglés
            "HELLO": "¡Hola! Me alegra saludarte. ¿Cómo estás hoy?",
            "GOODBYE": "¡Hasta luego! Fue un gusto practicar contigo.",
            "THANK_YOU": "¡De nada! Es un placer ayudarte a aprender.",
            "YES": "¡Genial! Me alegra que estés de acuerdo.",
            "NO": "Entiendo. ¿Hay algo más que te gustaría practicar?",
            "default": "Interesante seña. ¿Puedes intentar decir 'HOLA'?"
        },
        "introduction": {
            "ME_LLAMO": "Mucho gusto en conocerte. Yo soy tu asistente de LSM.",
            "HOLA": "¡Hola! ¿Podrías presentarte con la seña ME_LLAMO?",
            "MUCHO_GUSTO": "¡El gusto es mío! ¿De dónde eres?",
            # Compatibilidad
            "MY_NAME": "Mucho gusto en conocerte. Yo soy tu asistente de LSM.",
            "HELLO": "¡Hola! ¿Podrías presentarte con la seña de tu nombre?",
            "NICE_TO_MEET": "¡El gusto es mío! ¿De dónde eres?",
            "default": "Intenta presentarte usando las señas que has aprendido."
        },
        "asking_help": {
            "AYUDA": "¡Claro que puedo ayudarte! ¿Qué necesitas?",
            "POR_FAVOR": "Por supuesto, con mucho gusto. ¿En qué puedo asistirte?",
            "GRACIAS": "¡Siempre es un placer ayudar!",
            # Compatibilidad
            "HELP": "¡Claro que puedo ayudarte! ¿Qué necesitas?",
            "PLEASE": "Por supuesto, con mucho gusto. ¿En qué puedo asistirte?",
            "THANK_YOU": "¡Siempre es un placer ayudar!",
            "default": "Si necesitas ayuda, muestra la seña de 'AYUDA'."
        },
        "familia": {
            "FAMILIA": "¡Qué bonito! Cuéntame de tu familia.",
            "MAMA": "¿Cómo está tu mamá?",
            "PAPA": "¿Y tu papá cómo está?",
            "HERMANO": "¿Tienes hermanos? ¡Qué bien!",
            "AMOR": "Se nota que quieres mucho a tu familia.",
            "default": "Cuéntame más de tu familia."
        },
        "necesidades": {
            "AGUA": "¿Quieres agua? Aquí hay.",
            "COMIDA": "¿Tienes hambre? Vamos a comer.",
            "CASA": "¿Vamos a casa?",
            "AYUDA": "¿Necesitas ayuda con algo?",
            "default": "Dime qué necesitas."
        },
        "default_scenario": {
            "default": "Detecté tu seña. ¡Sigue practicando LSM!"
        }
    }
    
    scenario_responses = responses.get(scenario, responses["default_scenario"])
    response = scenario_responses.get(detected_sign.upper(), scenario_responses["default"])
    
    return response


def get_feedback_message(
    is_correct: bool,
    confidence: float,
    target_sign: str
) -> str:
    """
    Genera mensaje de retroalimentación para el usuario.
    
    Args:
        is_correct: Si la seña fue correcta
        confidence: Nivel de confianza
        target_sign: Seña objetivo
        
    Returns:
        Mensaje de retroalimentación
    """
    if is_correct:
        if confidence > 0.95:
            return f"¡Excelente! Perfecta ejecución de '{target_sign}'. 🎉"
        elif confidence > 0.85:
            return f"¡Muy bien! La seña '{target_sign}' está clara. 👍"
        else:
            return f"¡Correcto! Intenta hacer la seña '{target_sign}' con más claridad."
    else:
        return f"Sigue intentando. Muestra la seña '{target_sign}'. 💪"


# =============================================================================
# Test de utilidades
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🤟 Tutor LSM - Utils Test")
    print("=" * 60)
    
    # Test de estructuras de datos
    test_landmarks = np.random.rand(21, 3)
    hand = HandLandmarks(
        landmarks=test_landmarks,
        hand_type=HandType.RIGHT,
        confidence=0.95
    )
    
    print(f"\n✓ HandLandmarks creado: {hand.hand_type.value}")
    print(f"✓ Landmarks shape: {hand.landmarks.shape}")
    
    # Test de normalización
    normalized = hand.normalize()
    print(f"✓ Landmarks normalizados: {normalized.landmarks.shape}")
    
    # Test de extracción de características
    features = create_feature_vector(hand)
    print(f"✓ Vector de características: {features.shape}")
    
    # Test de ángulos y distancias
    angles = extract_finger_angles(test_landmarks)
    distances = extract_finger_distances(test_landmarks)
    print(f"✓ Ángulos extraídos: {angles.shape}")
    print(f"✓ Distancias extraídas: {distances.shape}")
    
    # Test de suavizado
    predictions = ["A", "A", "B", "A", "A"]
    confidences = [0.9, 0.85, 0.6, 0.88, 0.92]
    smoothed, conf = smooth_predictions(predictions, confidences)
    print(f"✓ Predicción suavizada: {smoothed} ({conf:.2f})")
    
    print("\n" + "=" * 60)
    print("Todos los tests pasaron correctamente ✓")
    print("=" * 60)
