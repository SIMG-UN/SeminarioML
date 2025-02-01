from ultralytics import YOLO
import cv2
import time
import argparse

def process_video(source_path='0', output_path=None, conf_threshold=0.5):
    """
    Procesa video usando YOLOv8 para detección de objetos
    
    Args:
        source_path: Puede ser '0' para webcam o ruta a un archivo de video
        output_path: Ruta para guardar el video procesado (opcional)
        conf_threshold: Umbral de confianza para las detecciones
    """
    # Cargar modelo YOLO
    model = YOLO('yolov8n.pt')
    
    # Abrir fuente de video
    if source_path == '0':
        cap = cv2.VideoCapture(0)  # Webcam
        print("Usando webcam - Presiona 'q' para salir")
    else:
        cap = cv2.VideoCapture(source_path)
        print(f"Procesando video: {source_path}")
    
    # Obtener propiedades del video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Configurar writer si se especifica output_path
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, 
                               (frame_width, frame_height))
    
    # Variables para FPS
    fps_start_time = time.time()
    fps_counter = 0
    fps_display = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Actualizar contador FPS
        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps_display = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Realizar detección
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        
        # Procesar detecciones
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Obtener coordenadas
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Obtener confianza y clase
                confidence = float(box.conf)
                class_id = int(box.cls)
                class_name = result.names[class_id]
                
                # Dibujar rectángulo y etiqueta
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{class_name}: {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Mostrar FPS
        cv2.putText(frame, f'FPS: {fps_display}', (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mostrar frame
        cv2.imshow('YOLO Detection', frame)
        
        # Guardar frame si se especificó output
        if writer:
            writer.write(frame)
        
        # Presionar 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Limpieza
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='YOLO Video Processing')
    parser.add_argument('--source', default='data/WhatsApp Video 2024-11-23 at 2.10.19 PM.mp4',
                        help='Fuente del video (0 para webcam o ruta al archivo)')
    parser.add_argument('--output', default=None,
                        help='Ruta para guardar el video procesado')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Umbral de confianza (0-1)')
    
    args = parser.parse_args()
    process_video(args.source, args.output, args.conf)

if __name__ == "__main__":
    main()