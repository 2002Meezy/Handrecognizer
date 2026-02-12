"""
Módulo de detecção de mãos usando MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional


class HandDetector:
    """
    Classe para detecção de mãos e extração de landmarks usando MediaPipe.
    """
    
    def __init__(
        self,
        static_mode: bool = False,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5
    ):
        """
        Inicializa o detector de mãos.
        
        Args:
            static_mode: Se True, trata cada frame independentemente
            max_hands: Número máximo de mãos a detectar
            detection_confidence: Confiança mínima para detecção
            tracking_confidence: Confiança mínima para tracking
        """
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Inicializar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
        # IDs dos landmarks das pontas dos dedos
        # 4: polegar, 8: indicador, 12: médio, 16: anelar, 20: mindinho
        self.tip_ids = [4, 8, 12, 16, 20]
        
        # Resultados da última detecção
        self.results = None
        
    def find_hands(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """
        Detecta mãos no frame e opcionalmente desenha os landmarks.
        
        Args:
            frame: Frame BGR do OpenCV
            draw: Se True, desenha os landmarks no frame
            
        Returns:
            Frame com landmarks desenhados (se draw=True)
        """
        # Converter BGR para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar frame
        self.results = self.hands.process(rgb_frame)
        
        # Desenhar landmarks se detectado
        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style()
                )
        
        return frame
    
    def get_landmarks(self, frame: np.ndarray, hand_index: int = 0) -> Optional[List[Tuple[int, int, int]]]:
        """
        Retorna a lista de landmarks para uma mão específica.
        
        Args:
            frame: Frame para obter dimensões
            hand_index: Índice da mão (0 para primeira, 1 para segunda)
            
        Returns:
            Lista de tuplas (id, x, y) para cada landmark, ou None se não detectado
        """
        landmarks = []
        
        if self.results and self.results.multi_hand_landmarks:
            if hand_index < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_index]
                h, w, _ = frame.shape
                
                for id, lm in enumerate(hand.landmark):
                    # Converter coordenadas normalizadas para pixels
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((id, cx, cy))
                    
                return landmarks
        
        return None
    
    def get_landmarks_normalized(self, hand_index: int = 0) -> Optional[np.ndarray]:
        """
        Retorna landmarks normalizados (x, y, z) para uso em modelos de ML.
        
        Args:
            hand_index: Índice da mão
            
        Returns:
            Array numpy de shape (21, 3) com coordenadas normalizadas
        """
        if self.results and self.results.multi_hand_landmarks:
            if hand_index < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_index]
                landmarks = []
                
                for lm in hand.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                    
                return np.array(landmarks)
        
        return None
    
    def get_handedness(self, hand_index: int = 0) -> Optional[str]:
        """
        Retorna se a mão é esquerda ou direita.
        
        Args:
            hand_index: Índice da mão
            
        Returns:
            'Left' ou 'Right', ou None se não detectado
        """
        if self.results and self.results.multi_handedness:
            if hand_index < len(self.results.multi_handedness):
                return self.results.multi_handedness[hand_index].classification[0].label
        
        return None
    
    def get_num_hands(self) -> int:
        """
        Retorna o número de mãos detectadas.
        
        Returns:
            Número de mãos detectadas
        """
        if self.results and self.results.multi_hand_landmarks:
            return len(self.results.multi_hand_landmarks)
        return 0
    
    def get_bounding_box(self, frame: np.ndarray, hand_index: int = 0, padding: int = 20) -> Optional[Tuple[int, int, int, int]]:
        """
        Retorna a bounding box da mão.
        
        Args:
            frame: Frame para obter dimensões
            hand_index: Índice da mão
            padding: Pixels de margem ao redor da mão
            
        Returns:
            Tupla (x, y, w, h) ou None se não detectado
        """
        landmarks = self.get_landmarks(frame, hand_index)
        
        if landmarks:
            x_coords = [lm[1] for lm in landmarks]
            y_coords = [lm[2] for lm in landmarks]
            
            x_min = max(0, min(x_coords) - padding)
            y_min = max(0, min(y_coords) - padding)
            x_max = min(frame.shape[1], max(x_coords) + padding)
            y_max = min(frame.shape[0], max(y_coords) + padding)
            
            return (x_min, y_min, x_max - x_min, y_max - y_min)
        
        return None
    
    def release(self):
        """Libera recursos do MediaPipe."""
        self.hands.close()
