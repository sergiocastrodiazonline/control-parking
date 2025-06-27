import cv2
import numpy as np
import json
from ultralytics import YOLO
import time
from datetime import datetime

# Cargar el modelo YOLOv8n pre-entrenado
model = YOLO("yolov8n.pt")

# Variables globales
historial_plazas = {}  # Diccionario de coches {ID: (x1, y1, x2, y2)}
id_counter = 0  # Contador para asignar IDs únicos

# Archivo JSON para eventos
DISPONIBILIDADES_TXT = "disponibilidades.txt"

# Tiempo de comprobación cada 5 minutos (300 segundos)
TIEMPO_COMPROBACION = 300

# ZONA DE EXCLUSIÓN DEFINIDA COMO TRIÁNGULO
exclusion_zone = np.array([[0, 0], [600, 0], [0, 325]])

def aplicar_zona_exclusion(frame):
    """Aplica una máscara negra a la zona de exclusión definida por un polígono."""
    mask = np.ones_like(frame, dtype=np.uint8) * 255  # Máscara blanca
    cv2.fillPoly(mask, [exclusion_zone], (0, 0, 0))   # Zona a excluir en negro
    return cv2.bitwise_and(frame, mask)

def detectar_lineas_parking(frame, plazas_ocupadas):
    """Detecta líneas de parking de colores específicos y calcula el número total de plazas disponibles."""

    # Definir la región de interés (ajusta las coordenadas según tu vídeo)
    roi = frame[150:frame.shape[0]-50, 50:frame.shape[1]-50].copy() # Usar una copia

    # Convertir la ROI a HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Definir el rango de color amarillo en HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Crear una máscara para el color amarillo
    mascara_amarilla = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)

    # Filtrar la imagen usando la máscara
    imagen_filtrada_roi = cv2.bitwise_and(roi, roi, mask=mascara_amarilla)

    # Convertir a escala de grises y aplicar Canny
    gray_roi = cv2.cvtColor(imagen_filtrada_roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edges_roi = cv2.Canny(blurred_roi, 60, 150)

    # Detectar líneas utilizando Hough
    lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, 50, minLineLength=150, maxLineGap=80) # Ajusta minLineLength y maxLineGap

    plazas_detectadas = 23
    if lines is not None:
        lineas_filtradas = []
        # Separar líneas horizontales y verticales (con tolerancia)
        lineas_horizontales = []
        lineas_verticales = []

        for line in lines:
            x1_roi, y1_roi, x2_roi, y2_roi = line[0]
            angulo = np.degrees(np.arctan2(y2_roi - y1_roi, x2_roi - x1_roi))

            if abs(angulo) < 15 or abs(angulo) > 165: # Líneas más horizontales
                lineas_horizontales.append(line)
                cv2.line(frame, (x1_roi + 50, y1_roi + 150), (x2_roi + 50, y2_roi + 150), (0, 255, 0), 2)
            elif abs(angulo) > 75 and abs(angulo) < 105: # Líneas más verticales
                lineas_verticales.append(line)
                cv2.line(frame, (x1_roi + 50, y1_roi + 150), (x2_roi + 50, y2_roi + 150), (0, 0, 255), 2)

        # Contar las plazas basándonos en las líneas horizontales (podría necesitar más ajuste)
        # Una idea es contar cuántas líneas horizontales hay que tengan una longitud razonable
        # y que estén más o menos espaciadas.
        # Por ahora, vamos a hacer una aproximación contando el número de líneas horizontales.
        # Esto podría sobreestimar el número de plazas si se detectan varias líneas por plaza.
        plazas_detectadas = len(lineas_horizontales) // 2 # Intento inicial, puede necesitar refinamiento

    # Calcular el número de plazas libres
    plazas_libres = plazas_detectadas - plazas_ocupadas
    cv2.putText(frame, f'Plazas: {plazas_detectadas} - Ocupadas: {plazas_ocupadas} - Libres: {plazas_libres}',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return frame, plazas_libres, plazas_detectadas

def asignar_id(x1, y1, x2, y2):
    """Asigna un ID único a un coche basado en su posición."""
    global historial_plazas, id_counter

    for coche_id, (cx1, cy1, cx2, cy2) in historial_plazas.items():
        if abs(x1 - cx1) < 50 and abs(y1 - cy1) < 50:
            return coche_id

    id_counter += 1
    return id_counter

def detectar_coches(frame, plazas_detectadas):
    """Detecta coches con YOLOv8 y actualiza plazas ocupadas."""
    global historial_plazas

    results = model(frame)
    coches_actuales = {}
    nuevos_coches_detectados = [] # Lista para almacenar los IDs de los coches recién detectados

    MIN_CAR_SIZE = 2000  # Ajusta el valor según el tamaño esperado del coche

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            area = (x2 - x1) * (y2 - y1)  # Área del cuadro delimitador

            if cls == 2 and conf > 0.12:
                # Calcular el centro del coche
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Verificar si el centro del coche está dentro de la zona de exclusión
                if cv2.pointPolygonTest(exclusion_zone, (cx, cy), False) >= 0:
                    continue  # Ignorar este coche

                coche_id = asignar_id(x1, y1, x2, y2)
                coches_actuales[coche_id] = (x1, y1, x2, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Car {coche_id}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if coche_id not in historial_plazas:
                    nuevos_coches_detectados.append(coche_id)
                # Opcionalmente, también podrías registrar si la posición ha cambiado significativamente
                elif abs(x1 - historial_plazas.get(coche_id, (0,0,0,0))[0]) > 50 or abs(y1 - historial_plazas.get(coche_id, (0,0,0,0))[1]) > 50:
                    # Podrías decidir si quieres registrar un evento por movimiento significativo aquí
                    pass

    historial_plazas = coches_actuales.copy()
    return frame, len(coches_actuales)

def verificar_disponibilidad(plazas_libres):
    """Verifica la disponibilidad de las plazas y escribe en el archivo de texto."""
    if plazas_libres <= 5:
        disponibilidad = "Poca Disponibilidad de Plazas"
    elif 6 <= plazas_libres <= 10:
        disponibilidad = "Disponibilidad Media"
    elif 11 <= plazas_libres <= 22:
        disponibilidad = "Disponibilidad Alta"
    elif plazas_libres >= 23:
        disponibilidad = "Todas vacías"
    else:
        disponibilidad = "Error al determinar disponibilidad"

    # Escribir la disponibilidad en el archivo
    with open(DISPONIBILIDADES_TXT, "a") as file:
        file.write(f"{datetime.now()} - {disponibilidad}\n")
    print(f"{datetime.now()} - {disponibilidad}")

cap = cv2.VideoCapture('R192_168_12_253_80_CH01_08_03_58.webm')

last_check_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dibujar la zona de exclusión en rojo (visual)
    cv2.polylines(frame, [exclusion_zone], isClosed=True, color=(0, 0, 255), thickness=2)

    # Aplicar la zona de exclusión antes de detección
    frame_con_exclusion = aplicar_zona_exclusion(frame)

    # Detectar coches y actualizar el número de plazas ocupadas
    frame, plazas_ocupadas = detectar_coches(frame, 0)  # Pasar 0 como valor inicial para plazas_detectadas

    # Detectar las líneas de parking y calcular el número de plazas libres
    frame, plazas_libres, plazas_detectadas = detectar_lineas_parking(frame, plazas_ocupadas)

    # Verificar la disponibilidad cada 5 minutos
    current_time = time.time()
    if current_time - last_check_time >= TIEMPO_COMPROBACION:
        verificar_disponibilidad(plazas_libres)
        last_check_time = current_time

    cv2.imshow('Detección de Coches y Parking', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()