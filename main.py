"""
Reconhecimento de Gestos de Mão com Deep Learning
Script principal que captura vídeo da webcam e identifica números (0-5)
"""

import cv2
import numpy as np
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier


def draw_info_panel(frame: np.ndarray, finger_count: int, fingers: list, gesture_name: str):
    """
    Desenha painel de informações no frame.
    
    Args:
        frame: Frame do vídeo
        finger_count: Número de dedos levantados
        fingers: Lista de estados dos dedos
        gesture_name: Nome do gesto
    """
    h, w, _ = frame.shape
    
    # Fundo semi-transparente para o painel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Título
    cv2.putText(
        frame, "DETECTOR DE GESTOS", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )
    
    # Número grande
    cv2.putText(
        frame, str(finger_count), (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4
    )
    
    # Nome do gesto
    cv2.putText(
        frame, gesture_name, (100, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    
    # Estado dos dedos
    finger_names = ["P", "I", "M", "A", "Mi"]
    for i, (name, state) in enumerate(zip(finger_names, fingers)):
        color = (0, 255, 0) if state else (0, 0, 255)
        cv2.putText(
            frame, name, (100 + i * 35, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    
    # Instruções
    cv2.putText(
        frame, "Pressione 'Q' para sair", (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
    )


def main():
    """Função principal do programa."""
    
    print("=" * 50)
    print("  RECONHECIMENTO DE GESTOS DE MAO")
    print("  Mostre numeros de 0 a 5 com sua mao")
    print("=" * 50)
    print("\nIniciando webcam...")
    
    # Inicializar detector e classificador
    detector = HandDetector(
        max_hands=1,
        detection_confidence=0.7,
        tracking_confidence=0.5
    )
    classifier = GestureClassifier()
    
    # Inicializar webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERRO: Nao foi possivel abrir a webcam!")
        print("Verifique se a webcam esta conectada e nao esta sendo usada por outro programa.")
        return
    
    # Configurar resolução
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Webcam iniciada com sucesso!")
    print("Mostre sua mao para a camera...")
    print("Pressione 'Q' para sair\n")
    
    # Variáveis para FPS
    prev_time = 0
    
    while True:
        success, frame = cap.read()
        
        if not success:
            print("Erro ao capturar frame")
            break
        
        # Espelhar frame para experiência mais natural
        frame = cv2.flip(frame, 1)
        
        # Detectar mãos
        frame = detector.find_hands(frame, draw=True)
        
        
        finger_count = 0
        fingers = [0, 0, 0, 0, 0]
        gesture_name = "Nenhuma mao detectada"
        
        
        if detector.get_num_hands() > 0:
            
            landmarks = detector.get_landmarks_normalized(hand_index=0)
            handedness = detector.get_handedness(hand_index=0)
            
            if landmarks is not None and handedness is not None:
                
                finger_count, fingers = classifier.classify(landmarks, handedness)
                gesture_name = classifier.get_gesture_name(finger_count)
                
                # Desenhar bounding box
                bbox = detector.get_bounding_box(frame, hand_index=0)
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"{handedness} Hand", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
        
        
        curr_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        
        draw_info_panel(frame, finger_count, fingers, gesture_name)
        
        
        cv2.putText(
            frame, f"FPS: {int(fps)}", (frame.shape[1] - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        
        
        cv2.imshow("Reconhecimento de Gestos - Pressione Q para sair", frame)
        
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\nEncerrando programa...")
            break
    
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    
    print("Programa encerrado com sucesso!")


if __name__ == "__main__":
    main()
