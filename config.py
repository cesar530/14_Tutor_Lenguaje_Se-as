"""
=============================================================================
Tutor de Lenguaje de Señas Mexicano (LSM) - Configuración
=============================================================================

Archivo de configuración centralizado para el proyecto.
Contiene todos los parámetros ajustables del sistema.
Adaptado para Lenguaje de Señas Mexicano (LSM).

OpenCV: 4.13.0 | MediaPipe: 0.10.31

Author: César Adrián Delgado Díaz
LinkedIn: https://www.linkedin.com/in/cesar-delgado-diaz
GitHub: https://github.com/cesar530

License: MIT
=============================================================================
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum


# =============================================================================
# Rutas del Proyecto
# =============================================================================

# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.absolute()

# Directorios de datos
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
LOGS_DIR = PROJECT_ROOT / "logs"
USER_PROGRESS_DIR = PROJECT_ROOT / "user_progress"

# Archivos específicos
LESSONS_FILE = PROJECT_ROOT / "lessons.json"  # ASL (default)
LESSONS_FILE_ASL = PROJECT_ROOT / "lessons.json"
LESSONS_FILE_LSM = PROJECT_ROOT / "lessons_lsm.json"
CLASSIFIER_MODEL = MODELS_DIR / "gesture_classifier.pkl"
KERAS_MODEL = MODELS_DIR / "sign_language_model.keras"


# =============================================================================
# Configuración de Cámara
# =============================================================================

@dataclass
class CameraConfig:
    """Configuración de la cámara."""
    index: int = 0  # Índice de cámara (0 = default)
    width: int = 640  # Ancho de captura
    height: int = 480  # Alto de captura
    fps: int = 30  # Frames por segundo objetivo
    flip_horizontal: bool = True  # Efecto espejo
    
    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)


CAMERA = CameraConfig()


# =============================================================================
# Configuración de MediaPipe Tasks API (0.10.31+)
# =============================================================================

@dataclass
class MediaPipeConfig:
    """Configuración de MediaPipe Hand Landmarker (Tasks API)."""
    static_image_mode: bool = False
    max_num_hands: int = 2
    model_complexity: int = 1  # 0 = Lite, 1 = Full (no usado en Tasks API)
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5


MEDIAPIPE = MediaPipeConfig()


# =============================================================================
# Configuración de Detección
# =============================================================================

@dataclass
class DetectionConfig:
    """Configuración del sistema de detección."""
    confidence_threshold: float = 0.7  # Umbral mínimo de confianza
    prediction_smoothing: bool = True  # Suavizar predicciones
    smoothing_window: int = 5  # Ventana de suavizado
    cooldown_frames: int = 10  # Frames entre detecciones
    min_consecutive_frames: int = 3  # Frames consecutivos para confirmar


DETECTION = DetectionConfig()


# =============================================================================
# Configuración de Interfaz
# =============================================================================

@dataclass 
class UIConfig:
    """Configuración de la interfaz visual."""
    # Colores (BGR)
    color_primary: Tuple[int, int, int] = (0, 255, 0)  # Verde
    color_secondary: Tuple[int, int, int] = (255, 255, 0)  # Cyan
    color_warning: Tuple[int, int, int] = (0, 255, 255)  # Amarillo
    color_error: Tuple[int, int, int] = (0, 0, 255)  # Rojo
    color_text: Tuple[int, int, int] = (255, 255, 255)  # Blanco
    color_background: Tuple[int, int, int] = (50, 50, 50)  # Gris oscuro
    
    # Tipografía
    font_scale_large: float = 1.2
    font_scale_medium: float = 0.8
    font_scale_small: float = 0.6
    font_thickness: int = 2
    
    # Landmarks
    landmark_radius: int = 5
    connection_thickness: int = 2
    
    # Paneles
    panel_opacity: float = 0.7
    margin: int = 20


UI = UIConfig()


# =============================================================================
# Configuración de Tutoría
# =============================================================================

@dataclass
class TutorConfig:
    """Configuración del sistema de tutoría."""
    feedback_delay: float = 1.5  # Segundos entre retroalimentación
    min_accuracy_to_pass: float = 0.8  # Precisión mínima para aprobar
    repetitions_per_sign: int = 3  # Repeticiones requeridas por seña
    show_hints: bool = True  # Mostrar pistas
    audio_feedback: bool = True  # Retroalimentación por audio
    celebration_threshold: float = 0.95  # Umbral para celebración


TUTOR = TutorConfig()


# =============================================================================
# Configuración de Conversación
# =============================================================================

@dataclass
class ConversationConfig:
    """Configuración del agente conversacional."""
    response_delay: float = 0.5  # Segundos antes de responder
    history_length: int = 10  # Mensajes a recordar
    default_scenario: str = "greeting"
    available_scenarios: List[str] = field(default_factory=lambda: [
        "greeting",
        "introduction", 
        "asking_help",
        "shopping",
        "directions"
    ])


CONVERSATION = ConversationConfig()


# =============================================================================
# Sistema de Selección de Idioma de Señas
# =============================================================================

class SignLanguage(Enum):
    """Idiomas de señas soportados."""
    ASL = "asl"      # American Sign Language
    LSM = "lsm"      # Lengua de Señas Mexicana


# Variable global para seleccionar el idioma activo
# Cambiar a SignLanguage.ASL para American Sign Language
# Cambiar a SignLanguage.LSM para Lengua de Señas Mexicana
ACTIVE_SIGN_LANGUAGE = SignLanguage.LSM


def set_sign_language(language: SignLanguage):
    """Cambia el idioma de señas activo."""
    global ACTIVE_SIGN_LANGUAGE, ALL_SIGNS, NUM_CLASSES
    global ASL_ALPHABET, ASL_NUMBERS, ASL_WORDS
    
    ACTIVE_SIGN_LANGUAGE = language
    
    if language == SignLanguage.ASL:
        ASL_ALPHABET = ASL_ALFABETO_ORIGINAL
        ASL_NUMBERS = ASL_NUMEROS_ORIGINAL
        ASL_WORDS = ASL_PALABRAS_ORIGINAL
        ALL_SIGNS = {**ASL_ALFABETO_ORIGINAL, **ASL_NUMEROS_ORIGINAL, **ASL_PALABRAS_ORIGINAL}
    else:  # LSM
        ASL_ALPHABET = LSM_ALFABETO
        ASL_NUMBERS = LSM_NUMEROS
        ASL_WORDS = LSM_PALABRAS
        ALL_SIGNS = {**LSM_ALFABETO, **LSM_NUMEROS, **LSM_PALABRAS}
    
    NUM_CLASSES = len(ALL_SIGNS)
    return ALL_SIGNS


def get_sign_language_info():
    """Retorna información del idioma de señas activo."""
    if ACTIVE_SIGN_LANGUAGE == SignLanguage.ASL:
        return {
            "code": "ASL",
            "name": "American Sign Language",
            "alphabet_count": len(ASL_ALFABETO_ORIGINAL),
            "numbers_count": len(ASL_NUMEROS_ORIGINAL),
            "words_count": len(ASL_PALABRAS_ORIGINAL),
            "total": len(ASL_ALFABETO_ORIGINAL) + len(ASL_NUMEROS_ORIGINAL) + len(ASL_PALABRAS_ORIGINAL),
            "lessons_file": str(LESSONS_FILE_ASL)
        }
    else:
        return {
            "code": "LSM",
            "name": "Lengua de Señas Mexicana",
            "alphabet_count": len(LSM_ALFABETO),
            "numbers_count": len(LSM_NUMEROS),
            "words_count": len(LSM_PALABRAS),
            "total": len(LSM_ALFABETO) + len(LSM_NUMEROS) + len(LSM_PALABRAS),
            "lessons_file": str(LESSONS_FILE_LSM)
        }


def get_lessons_file():
    """Retorna la ruta del archivo de lecciones según el idioma activo."""
    if ACTIVE_SIGN_LANGUAGE == SignLanguage.ASL:
        return LESSONS_FILE_ASL
    return LESSONS_FILE_LSM


# =============================================================================
# Señas Soportadas
# =============================================================================

class SignCategory(Enum):
    """Categorías de señas."""
    ALFABETO = "alfabeto"
    NUMEROS = "numeros"
    PALABRAS = "palabras"
    FRASES = "frases"
    # Aliases para compatibilidad
    ALPHABET = "alfabeto"
    NUMBERS = "numeros"
    WORDS = "palabras"
    PHRASES = "frases"


# =============================================================================
# ASL - American Sign Language (Original)
# =============================================================================

# Alfabeto ASL (26 letras, sin Ñ ni RR)
ASL_ALFABETO_ORIGINAL = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
    15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y",
    25: "Z"
}

# Números ASL (0-9)
ASL_NUMEROS_ORIGINAL = {
    26: "0", 27: "1", 28: "2", 29: "3", 30: "4",
    31: "5", 32: "6", 33: "7", 34: "8", 35: "9"
}

# Palabras/frases comunes en ASL (en inglés)
ASL_PALABRAS_ORIGINAL = {
    36: "HELLO",
    37: "GOODBYE",
    38: "THANK_YOU",
    39: "PLEASE",
    40: "YES",
    41: "NO",
    42: "HELP",
    43: "SORRY",
    44: "LOVE",
    45: "FRIEND"
}


# =============================================================================
# LSM - Lengua de Señas Mexicana
# =============================================================================

# Mapeo de señas del alfabeto LSM (Lenguaje de Señas Mexicano)
# Incluye Ñ y RR que son específicas del español mexicano
LSM_ALFABETO = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "Ñ",
    15: "O", 16: "P", 17: "Q", 18: "R", 19: "RR",
    20: "S", 21: "T", 22: "U", 23: "V", 24: "W",
    25: "X", 26: "Y", 27: "Z"
}

# Alias para compatibilidad
ASL_ALPHABET = LSM_ALFABETO

# Mapeo de números LSM
LSM_NUMEROS = {
    28: "0", 29: "1", 30: "2", 31: "3", 32: "4",
    33: "5", 34: "6", 35: "7", 36: "8", 37: "9",
    38: "10", 39: "100", 40: "1000"
}

# Alias para compatibilidad
ASL_NUMBERS = LSM_NUMEROS

# Palabras/frases comunes en LSM
LSM_PALABRAS = {
    41: "HOLA",
    42: "ADIOS",
    43: "GRACIAS",
    44: "POR_FAVOR",
    45: "SI",
    46: "NO",
    47: "AYUDA",
    48: "PERDON",
    49: "AMOR",
    50: "AMIGO",
    51: "FAMILIA",
    52: "MAMA",
    53: "PAPA",
    54: "HERMANO",
    55: "AGUA",
    56: "COMIDA",
    57: "CASA",
    58: "TRABAJO",
    59: "ESCUELA",
    60: "BUENOS_DIAS",
    61: "BUENAS_NOCHES",
    62: "COMO_ESTAS",
    63: "MUCHO_GUSTO",
    64: "ME_LLAMO"
}

# Alias para compatibilidad
ASL_WORDS = LSM_PALABRAS

# Mapeo de señas en inglés a español para compatibilidad
SIGN_TRANSLATION = {
    "HELLO": "HOLA",
    "GOODBYE": "ADIOS",
    "THANK_YOU": "GRACIAS",
    "PLEASE": "POR_FAVOR",
    "YES": "SI",
    "NO": "NO",
    "HELP": "AYUDA",
    "SORRY": "PERDON",
    "LOVE": "AMOR",
    "FRIEND": "AMIGO",
    "MY_NAME": "ME_LLAMO"
}

# Combinación completa de señas LSM
ALL_SIGNS = {**LSM_ALFABETO, **LSM_NUMEROS, **LSM_PALABRAS}
NUM_CLASSES = len(ALL_SIGNS)


# =============================================================================
# Configuración de Modelo
# =============================================================================

@dataclass
class ModelConfig:
    """Configuración del modelo de clasificación."""
    input_features: int = 93  # 63 (landmarks) + 15 (ángulos) + 15 (distancias)
    num_classes: int = NUM_CLASSES
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50


MODEL = ModelConfig()


# =============================================================================
# Configuración de Entrenamiento
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuración para entrenamiento de modelos."""
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    augmentation: bool = True
    early_stopping_patience: int = 10


TRAINING = TrainingConfig()


# =============================================================================
# Mensajes del Sistema
# =============================================================================

MESSAGES = {
    "welcome": "¡Bienvenido al Tutor de Lenguaje de Señas Mexicano (LSM)! 🤟",
    "camera_error": "Error: No se pudo acceder a la cámara.",
    "model_error": "Error: No se pudo cargar el modelo.",
    "no_hand_detected": "No se detecta ninguna mano.",
    "low_confidence": "Baja confianza en la detección.",
    "correct": "¡Correcto! 🎉",
    "incorrect": "Inténtalo de nuevo.",
    "lesson_complete": "¡Lección completada!",
    "practice_tip": "Consejo: Mantén la mano estable y bien iluminada.",
    "lsm_info": "LSM - Lenguaje de Señas Mexicano",
}


# =============================================================================
# Funciones de Utilidad de Configuración
# =============================================================================

def ensure_directories():
    """Crea los directorios necesarios si no existen."""
    directories = [DATA_DIR, MODELS_DIR, ASSETS_DIR, LOGS_DIR, USER_PROGRESS_DIR]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Crear subdirectorios
    (DATA_DIR / "landmarks").mkdir(exist_ok=True)
    (ASSETS_DIR / "reference_signs").mkdir(exist_ok=True)


def get_sign_name(class_id: int) -> str:
    """
    Obtiene el nombre de la seña dado su ID de clase.
    
    Args:
        class_id: ID numérico de la clase
        
    Returns:
        Nombre de la seña
    """
    return ALL_SIGNS.get(class_id, "UNKNOWN")


def get_class_id(sign_name: str) -> int:
    """
    Obtiene el ID de clase dado el nombre de la seña.
    
    Args:
        sign_name: Nombre de la seña
        
    Returns:
        ID de clase o -1 si no existe
    """
    for class_id, name in ALL_SIGNS.items():
        if name == sign_name:
            return class_id
    return -1


def get_signs_by_category(category: SignCategory) -> Dict[int, str]:
    """
    Obtiene señas de una categoría específica.
    
    Args:
        category: Categoría de señas
        
    Returns:
        Diccionario de señas de la categoría
    """
    if category in (SignCategory.ALPHABET, SignCategory.ALFABETO):
        return LSM_ALFABETO
    elif category in (SignCategory.NUMBERS, SignCategory.NUMEROS):
        return LSM_NUMEROS
    elif category in (SignCategory.WORDS, SignCategory.PALABRAS):
        return LSM_PALABRAS
    else:
        return ALL_SIGNS


# =============================================================================
# Inicialización
# =============================================================================

# Crear directorios al importar el módulo
ensure_directories()


# =============================================================================
# Test de configuración
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🤟 Tutor de Lenguaje de Señas Mexicano (LSM) - Config Test")
    print("=" * 60)
    
    print(f"\n📁 Project Root: {PROJECT_ROOT}")
    print(f"📁 Data Directory: {DATA_DIR}")
    print(f"📁 Models Directory: {MODELS_DIR}")
    
    print(f"\n📷 Camera Configuration:")
    print(f"   Resolution: {CAMERA.resolution}")
    print(f"   FPS: {CAMERA.fps}")
    
    print(f"\n🤖 MediaPipe Configuration:")
    print(f"   Max Hands: {MEDIAPIPE.max_num_hands}")
    print(f"   Detection Confidence: {MEDIAPIPE.min_detection_confidence}")
    
    print(f"\n🎯 Detection Configuration:")
    print(f"   Confidence Threshold: {DETECTION.confidence_threshold}")
    print(f"   Smoothing Window: {DETECTION.smoothing_window}")
    
    print(f"\n📚 Señas LSM Cargadas:")
    print(f"   Alfabeto (con Ñ): {len(LSM_ALFABETO)} señas")
    print(f"   Números: {len(LSM_NUMEROS)} señas")
    print(f"   Palabras: {len(LSM_PALABRAS)} señas")
    print(f"   Total de Clases: {NUM_CLASSES}")
    
    print(f"\n🧠 Model Configuration:")
    print(f"   Input Features: {MODEL.input_features}")
    print(f"   Hidden Layers: {MODEL.hidden_layers}")
    
    print("\n" + "=" * 60)
    print("✅ Configuración LSM cargada correctamente")
    print("=" * 60)
