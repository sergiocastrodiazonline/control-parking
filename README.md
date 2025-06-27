# Control-Parking
Aplicación para Controlar el Parking usando Visión Artificial (OpenCV) etc.

# Código

El código se ha dividido en dos programas distintos para facilitar el trabajo siendo los dos:

El 1º se enfoca en la puerta del parking, tenemos dos contadores uno de Salida y uno de Entrada, el código comprueba que cada 30 segundos se añada un evento (esto es modificable sin embargo para poner un rango especifico entre la llegada e ida de vehículos se ha añadido). Cuando detecte que pase un coche por uno de los box de las líneas lo añade como evento a un fichero llamado eventos.json donde guardamos la fecha del evento, el ID del vehiculo, las box o localización y si ha sido de salida o de entrada. Actualmente el sistema de matriculas falla mucho usando PyTesseract, se nota un poco que la calidad que recibe OpenCV no es la más perfecta aunque el video haya sido extraído de la cámara, porque a veces detecta números y letras y otras veces no usando los parámetros más típicos de la propia librería. 

Se ha investigado además sobre sistemas de Deep Learning como EasyOCR que son mas potentes para bajas calidades pero han sido muy complicados de instalar y no he logrado que funcione.

El 2º se enfoca en la zona del parking, aquí cada 5 minutos comprueba cuantas plazas hay disponibles (falla un poco porque a veces confunde vehículos y reconoce mas o menos vehiculos), en este fijamos el numero como 23 porque sabemos que en este caso son siempre este numero y va contando gracias a un modelo YOLO los vehículos de toda la plaza. 

Hemos colocado una linea para no contar los del exterior, cada 5 minutos el programa guardará en un fichero la disponibilidad actualizada y la fecha/hora siendo de 3 rangos distintos:

# ¿Que se ha aprendido con el desarrollo?

