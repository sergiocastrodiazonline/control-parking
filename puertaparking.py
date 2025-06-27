# Importación de librerías necesarias
import cv2
import json
from datetime import datetime, timedelta
from ultralytics import YOLO  # Para detección de matrículas con modelo YOLO
import pytesseract  # OCR para lectura de texto en las matrículas

# --- Cargar modelo personalizado de detección de matrículas ---
modelo_matriculas = YOLO("license_plate_detector.pt")

# Ruta del ejecutable de Tesseract OCR (ajústalo si usas Linux o Mac)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Configuración del sistema ---
VIDEO_PATH = 'R192_168_12_254_80_CH01_20_36_14.webm'  # Ruta del video a procesar

# Coordenadas de líneas para detectar entrada y salida
LINEA_SALIDA_P1 = (100, 400)
LINEA_SALIDA_P2 = (300, 250)
LINEA_ENTRADA_P1 = (700, 350)
LINEA_ENTRADA_P2 = (500, 200)

# Parámetros de detección y filtrado
UMBRAL_AREA_MINIMA = 500  # Área mínima para considerar un objeto como vehículo
HISTORY = 500
VAR_THRESHOLD = 16
DETECT_SHADOWS = False

# Elemento estructurante para operaciones morfológicas
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Archivo donde se guardarán los eventos
EVENTOS_FILE = 'eventos.json'

# Tiempo mínimo entre detecciones del mismo tipo (entrada/salida)
INTERVALO_MINIMO = timedelta(seconds=30)

# --- Variables de estado ---
background_subtractor = cv2.createBackgroundSubtractorMOG2(HISTORY, VAR_THRESHOLD, DETECT_SHADOWS)
detecciones_previas = {}
contador_id = 0
eventos = []
vehiculos_detectados = {}  # {id: "entrada"/"salida"}
ultimo_evento = None
contador_entradas = 0
contador_salidas = 0


def detectar_matricula(frame, bbox_vehiculo):
    """
    Extrae la región de interés del vehículo, detecta la matrícula y aplica OCR.
    """
    x, y, w, h = bbox_vehiculo
    vehiculo_roi = frame[y:y+h, x:x+w]

    resultados = modelo_matriculas.predict(vehiculo_roi, conf=0.6, verbose=True)

    if not resultados or not resultados[0].boxes:
        return ""

    boxes = resultados[0].boxes.xyxy.cpu().numpy().astype(int)

    for box in boxes:
        x1, y1, x2, y2 = box
        placa_roi = vehiculo_roi[y1:y2, x1:x2]

        # Preprocesar imagen para mejorar OCR
        gris = cv2.cvtColor(placa_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        texto = pytesseract.image_to_string(thresh, config='--psm 8')
        texto_limpio = ''.join(filter(str.isalnum, texto))
        return texto_limpio.upper()

    return ""


def guardar_evento(tipo, bbox, vehiculo_id, frame):
    """
    Guarda el evento (entrada/salida) en un archivo JSON.
    """
    global eventos, ultimo_evento, vehiculos_detectados, contador_entradas, contador_salidas
    timestamp = datetime.now().isoformat()

    # Obtener matrícula usando OCR
    matricula = detectar_matricula(frame, bbox)

    evento = {
        "tipo": tipo,
        "timestamp": timestamp,
        "bbox": bbox,
        "vehiculo_id": vehiculo_id,
        "matricula": matricula
    }

    eventos.append(evento)

    # Guardar en archivo
    try:
        with open(EVENTOS_FILE, 'w') as f:
            json.dump(eventos, f, indent=4)
    except Exception as e:
        print(f"Error al guardar el evento en {EVENTOS_FILE}: {e}")

    print(f"Evento detectado: {tipo} - {timestamp} - BBox: {bbox} - Vehiculo ID: {vehiculo_id} - Matrícula: {matricula}")
    
    # Actualizar estado
    ultimo_evento = datetime.now()
    vehiculos_detectados[vehiculo_id] = tipo

    if tipo == "entrada":
        contador_entradas += 1
    elif tipo == "salida":
        contador_salidas += 1


def calcular_centroide(bbox):
    """
    Calcula el centro del bounding box.
    """
    x, y, w, h = bbox
    return int(x + w / 2), int(y + h / 2)


def detectar_cruces_linea(detecciones, frame):
    """
    Compara posiciones anteriores y actuales para detectar si se cruzó una línea.
    """
    global detecciones_previas, contador_id, vehiculos_detectados, ultimo_evento

    detecciones_actuales = detecciones.copy()
    nuevas_detecciones = []

    for det in detecciones_actuales:
        bbox_actual, centroide_actual = det
        objeto_encontrado = False

        for obj_id, (centroide_previo, bbox_previo) in detecciones_previas.items():
            if abs(centroide_actual[0] - centroide_previo[0]) < 50 and abs(centroide_actual[1] - centroide_previo[1]) < 50:
                objeto_encontrado = True
                cruzado_salida = False
                cruzado_entrada = False

                # Verificar cruce de la línea de salida
                orientacion_salida = (LINEA_SALIDA_P2[0] - LINEA_SALIDA_P1[0]) * (centroide_previo[1] - LINEA_SALIDA_P1[1]) - \
                                     (LINEA_SALIDA_P2[1] - LINEA_SALIDA_P1[1]) * (centroide_previo[0] - LINEA_SALIDA_P1[0])
                orientacion_actual_salida = (LINEA_SALIDA_P2[0] - LINEA_SALIDA_P1[0]) * (centroide_actual[1] - LINEA_SALIDA_P1[1]) - \
                                            (LINEA_SALIDA_P2[1] - LINEA_SALIDA_P1[1]) * (centroide_actual[0] - LINEA_SALIDA_P1[0])

                if orientacion_salida > 0 and orientacion_actual_salida <= 0:
                    if obj_id not in vehiculos_detectados:
                        if ultimo_evento is None or (datetime.now() - ultimo_evento) >= INTERVALO_MINIMO:
                            guardar_evento("salida", bbox_previo, obj_id, frame)
                            cruzado_salida = True
                            vehiculos_detectados[obj_id] = "salida"
                        else:
                            print("Evento de salida ignorado por intervalo mínimo.")

                # Verificar cruce de la línea de entrada
                orientacion_entrada = (LINEA_ENTRADA_P2[0] - LINEA_ENTRADA_P1[0]) * (centroide_previo[1] - LINEA_ENTRADA_P1[1]) - \
                                      (LINEA_ENTRADA_P2[1] - LINEA_ENTRADA_P1[1]) * (centroide_previo[0] - LINEA_ENTRADA_P1[0])
                orientacion_actual_entrada = (LINEA_ENTRADA_P2[0] - LINEA_ENTRADA_P1[0]) * (centroide_actual[1] - LINEA_ENTRADA_P1[1]) - \
                                             (LINEA_ENTRADA_P2[1] - LINEA_ENTRADA_P1[1]) * (centroide_actual[0] - LINEA_ENTRADA_P1[0])

                if orientacion_entrada < 0 and orientacion_actual_entrada >= 0:
                    if obj_id not in vehiculos_detectados:
                        if ultimo_evento is None or (datetime.now() - ultimo_evento) >= INTERVALO_MINIMO:
                            guardar_evento("entrada", bbox_previo, obj_id, frame)
                            cruzado_entrada = True
                            vehiculos_detectados[obj_id] = "entrada"
                        else:
                            print("Evento de entrada ignorado por intervalo mínimo.")
                break

        if not objeto_encontrado:
            nuevas_detecciones.append(det)

    # Asignar ID a las nuevas detecciones
    for det in nuevas_detecciones:
        bbox, centroide = det
        contador_id += 1
        detecciones_previas[contador_id] = (centroide, bbox)

    return contador_id


def main():
    """
    Función principal que procesa el video y detecta eventos de entrada y salida.
    """
    global detecciones_previas, vehiculos_detectados, ultimo_evento, contador_id

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error al abrir el video: {VIDEO_PATH}")
        return

    # Cargar eventos anteriores si existen
    try:
        with open(EVENTOS_FILE, 'r') as f:
            eventos = json.load(f)
            if eventos:
                ultimo_evento = datetime.fromisoformat(eventos[-1]['timestamp'])
                for evento in eventos:
                    if 'vehiculo_id' in evento:
                        vehiculos_detectados[evento['vehiculo_id']] = evento['tipo']
                    else:
                        contador_id += 1
                        evento['vehiculo_id'] = contador_id
                        vehiculos_detectados[contador_id] = evento['tipo']
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        print(f"Error al decodificar {EVENTOS_FILE}. Se inicializa vacío.")
        eventos = []
        vehiculos_detectados = {}
        ultimo_evento = None
        contador_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sustracción de fondo y limpieza de ruido
        mask = background_subtractor.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
        mask = cv2.dilate(mask, KERNEL, iterations=2)

        # Detectar contornos de objetos en movimiento
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detecciones_frame = []
        for contour in contours:
            if cv2.contourArea(contour) >= UMBRAL_AREA_MINIMA:
                x, y, w, h = cv2.boundingRect(contour)
                centroide = calcular_centroide((x, y, w, h))
                detecciones_frame.append(((x, y, w, h), centroide))

        # Detectar si cruzan línea de entrada o salida
        if detecciones_frame:
            contador_id = detectar_cruces_linea(detecciones_frame, frame)

        # Dibujar bounding boxes de vehículos detectados
        for obj_id, (centroide, (x, y, w, h)) in detecciones_previas.items():
            if obj_id in vehiculos_detectados:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Dibujar líneas de entrada y salida
        cv2.line(frame, LINEA_SALIDA_P1, LINEA_SALIDA_P2, (255, 0, 255), 2)
        cv2.putText(frame, "SALIDA", (LINEA_SALIDA_P1[0], LINEA_SALIDA_P1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        cv2.line(frame, LINEA_ENTRADA_P1, LINEA_ENTRADA_P2, (0, 255, 255), 2)
        cv2.putText(frame, "ENTRADA", (LINEA_ENTRADA_P2[0] - 50, LINEA_ENTRADA_P2[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Mostrar contadores en pantalla
        cv2.putText(frame, f"Entradas: {contador_entradas}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Salidas: {contador_salidas}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Mostrar imagen procesada
        cv2.imshow("Detección de Vehículos", frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()


# Punto de entrada del programa
if __name__ == "__main__":
    main()
