# 🤟 Tutor de Lenguaje de Señas Mexicano (LSM) 🇲🇽

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.31-green.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13.0-red.svg)](https://opencv.org/)

## 📋 Descripción

**Tutor de LSM** es un sistema inteligente de aprendizaje de Lenguaje de Señas Mexicano que utiliza visión por computadora y aprendizaje automático para:

1. **Reconocer señas LSM** desde la cámara en tiempo real
2. **Funcionar como tutor** con lecciones estructuradas, práctica interactiva y retroalimentación en tiempo real
3. **Permitir roleplay conversacional** mediante un agente de IA que interpreta las señas como entrada de texto

## 🎯 Características Principales

### 🖐️ Reconocimiento de Señas LSM
- Detección de manos en tiempo real usando MediaPipe
- Extracción de 21 landmarks por mano
- Clasificación del alfabeto LSM (A-Z, **Ñ**, **RR**)
- Reconocimiento de números (0-9, 10, 100, 1000)
- Palabras y frases básicas en español mexicano

### 📚 Sistema de Tutoría
- **Lecciones estructuradas** por niveles (Principiante → Intermedio → Avanzado)
- **Práctica guiada** con retroalimentación visual inmediata
- **Sistema de puntuación** y seguimiento de progreso
- **Ejercicios interactivos** con repetición espaciada
- **Contenido cultural** sobre la comunidad sorda en México

### 🤖 Agente Conversacional
- Roleplay de situaciones cotidianas mexicanas
- Respuestas contextuales basadas en señas detectadas
- Escenarios: saludos, presentaciones, familia, necesidades básicas

## 🏗️ Arquitectura del Proyecto

```
15_Tutor_Lenguaje_Señas/
│
├── 📓 sign_language_tutor.ipynb    # Notebook principal con documentación
├── 🐍 sign_language_tutor.py       # Script principal ejecutable
├── 🛠️ utils.py                     # Funciones auxiliares
├── ⚙️ config.py                    # Configuraciones del proyecto
├── 📖 lessons_lsm.json             # Estructura de lecciones LSM
├── 📋 requirements.txt             # Dependencias
├── 📄 README.md                    # Este archivo
├── 🚫 .gitignore                   # Archivos ignorados por Git
│
├── 📁 models/                      # Modelos entrenados
│   └── gesture_classifier.pkl     # Clasificador de gestos
│
├── 📁 data/                        # Datos de entrenamiento
│   └── landmarks/                  # Landmarks extraídos
│
└── 📁 assets/                      # Recursos visuales
    └── reference_signs/            # Imágenes de referencia LSM
```

## 🚀 Instalación

### Requisitos Previos
- Python 3.12+
- Cámara web funcional
- GPU (opcional, mejora rendimiento)

### Pasos de Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/cesar530/tutor-lsm.git
cd tutor-lsm

# 2. Crear entorno virtual
py -3.14 -m venv venv

# 3. Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt
```

## 💻 Uso

### Ejecutar el Notebook (Recomendado para aprendizaje)
```bash
jupyter notebook sign_language_tutor.ipynb
```

### Ejecutar el Script Principal
```bash
python sign_language_tutor.py
```

### Modos de Ejecución

```python
from sign_language_tutor import SignLanguageTutor

# Inicializar el tutor
tutor = SignLanguageTutor()

# Modo 1: Reconocimiento en tiempo real
tutor.start_recognition()

# Modo 2: Lección interactiva
tutor.start_lesson(lesson_id="L1_01_alfabeto_am")

# Modo 3: Práctica libre con retroalimentación
tutor.start_practice(category="numeros")

# Modo 4: Roleplay conversacional
tutor.start_conversation(scenario="greeting")
```

## 📊 Tecnologías Utilizadas

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| Python | 3.12+ | Lenguaje principal |
| MediaPipe | 0.10.31 | Detección de manos y landmarks |
| OpenCV | 4.13.0 | Captura y procesamiento de video |
| NumPy | 1.26+ | Operaciones numéricas |
| Scikit-learn | 1.4+ | Clasificación de gestos |
| Gradio | 4.14+ | Interfaz web interactiva (opcional) |

## 🎓 Estructura de Lecciones LSM

### Nivel 1: Fundamentos LSM
- Alfabeto LSM (A-Z, **Ñ**, **RR**)
- Números (0-9, 10, 100, 1000)
- Posiciones básicas de manos

### Nivel 2: Palabras Básicas
- Saludos (Hola, Adiós, Buenos días, Buenas noches)
- Cortesía (Gracias, Por favor, Perdón)
- Familia (Mamá, Papá, Hermano, Familia)
- Emociones (Amor, Amigo)

### Nivel 3: Vida Cotidiana
- Necesidades básicas (Agua, Comida, Casa)
- Lugares (Trabajo, Escuela)

### Nivel 4: Frases Simples
- Presentaciones personales
- Preguntas básicas

### Nivel 5: Conversación
- Diálogos cotidianos mexicanos
- Situaciones específicas
- Práctica con agente IA

## 📈 Métricas de Rendimiento

| Métrica | Valor |
|---------|-------|
| Precisión alfabeto LSM | ~95% |
| Precisión números | ~97% |
| Latencia detección | <50ms |
| FPS promedio | 30+ |

## 🛠️ Configuración Avanzada

Editar `config.py` para personalizar:

```python
# Configuración de cámara
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Configuración de detección
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.5

# Configuración de tutoría
FEEDBACK_DELAY = 1.5  # segundos
MIN_CONFIDENCE_DISPLAY = 0.8
```

## 🇲🇽 Notas Culturales sobre LSM

- **LSM** es el Lenguaje de Señas Mexicano, diferente del ASL estadounidense
- El alfabeto incluye la **Ñ** (única del español)
- Las **expresiones faciales** son parte gramatical de la lengua
- Pueden existir **variaciones regionales** en diferentes estados de México
- La **comunidad sorda mexicana** tiene una rica cultura propia

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/NuevaCaracteristica`)
3. Commit tus cambios (`git commit -m 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/NuevaCaracteristica`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2026 César Adrián Delgado Díaz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 👨‍💻 Autor

- 👤 Autor : **César Adrián Delgado Díaz**
- 💼 LinkedIn: [linkedin.com/in/cesar-delgado-diaz](linkedin.com/in/cesar-delgado-diaz)
- 🐙 GitHub: [github.com/cesar530](https://github.com/cesar530)

## 🙏 Agradecimientos

- [MediaPipe](https://mediapipe.dev/) por su excelente framework de ML
- [Comunidad Sorda Mexicana](https://www.conadis.gob.mx/) por recursos sobre LSM
- [CONADIS](https://www.gob.mx/conadis) por información sobre accesibilidad
- Intérpretes de LSM y la comunidad sorda por su valiosa contribución cultural

## 📚 Recursos Adicionales sobre LSM

- [Diccionario de LSM - CONADIS](https://www.conadis.gob.mx/)
- [Señas Mexicanas](https://www.youtube.com/@seniasmexicanas) - Canal de YouTube
- Federación Mexicana de Sordos

---

🇲🇽 ¡Aprende LSM y conecta con la comunidad sorda mexicana!

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!
