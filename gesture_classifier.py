"""
Módulo de classificação de gestos de mão
"""

import numpy as np
from typing import List, Optional, Tuple
from hand_detector import HandDetector


class GestureClassifier:
    """
    Classificador de gestos de mão baseado na posição dos dedos.
    Identifica números de 0 a 5 baseado em quantos dedos estão levantados.
    """
    
    # IDs dos landmarks das pontas dos dedos
    TIP_IDS = [4, 8, 12, 16, 20]  # polegar, indicador, médio, anelar, mindinho
    
    # IDs das articulações PIP (proximal interphalangeal) - para comparação
    PIP_IDS = [2, 6, 10, 14, 18]
    
    def __init__(self):
        """Inicializa o classificador de gestos."""
        self.finger_states = [0, 0, 0, 0, 0]  # Estado de cada dedo (0=fechado, 1=aberto)
        
    def classify(
        self, 
        landmarks: np.ndarray, 
        handedness: str = "Right"
    ) -> Tuple[int, List[int]]:
        """
        Classifica o gesto baseado nos landmarks da mão.
        
        Args:
            landmarks: Array numpy de shape (21, 3) com coordenadas normalizadas
            handedness: 'Left' ou 'Right' para ajustar lógica do polegar
            
        Returns:
            Tupla (número_dedos, lista_estados_dedos)
        """
        if landmarks is None or len(landmarks) != 21:
            return 0, [0, 0, 0, 0, 0]
        
        fingers = []
        
        # Polegar - movimento lateral (eixo X)
        # A lógica é invertida dependendo da mão
        thumb_tip = landmarks[self.TIP_IDS[0]]
        thumb_ip = landmarks[self.TIP_IDS[0] - 1]  # Articulação IP do polegar
        
        if handedness == "Right":
            # Mão direita: polegar aberto quando tip.x < ip.x
            if thumb_tip[0] < thumb_ip[0]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # Mão esquerda: polegar aberto quando tip.x > ip.x
            if thumb_tip[0] > thumb_ip[0]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # Outros 4 dedos - movimento vertical (eixo Y)
        # Dedo aberto quando a ponta está acima (y menor) da articulação PIP
        for i in range(1, 5):
            tip = landmarks[self.TIP_IDS[i]]
            pip = landmarks[self.TIP_IDS[i] - 2]  # Articulação PIP
            
            if tip[1] < pip[1]:  # Y menor = mais acima na imagem
                fingers.append(1)
            else:
                fingers.append(0)
        
        self.finger_states = fingers
        return sum(fingers), fingers
    
    def get_finger_names(self) -> List[str]:
        """
        Retorna os nomes dos dedos que estão levantados.
        
        Returns:
            Lista com nomes dos dedos levantados
        """
        names = ["Polegar", "Indicador", "Médio", "Anelar", "Mindinho"]
        return [names[i] for i, state in enumerate(self.finger_states) if state == 1]
    
    def get_gesture_name(self, count: int) -> str:
        """
        Retorna o nome do gesto baseado no número de dedos.
        
        Args:
            count: Número de dedos levantados
            
        Returns:
            Nome do gesto
        """
        gestures = {
            0: "Zero / Punho fechado",
            1: "Um",
            2: "Dois / Paz",
            3: "Três",
            4: "Quatro",
            5: "Cinco / Mão aberta"
        }
        return gestures.get(count, "Desconhecido")
    
    def is_thumbs_up(self, landmarks: np.ndarray, handedness: str = "Right") -> bool:
        """
        Verifica se o gesto é "joinha" (polegar para cima).
        
        Args:
            landmarks: Array de landmarks
            handedness: Lateralidade da mão
            
        Returns:
            True se for joinha
        """
        count, fingers = self.classify(landmarks, handedness)
        
        # Joinha: apenas polegar levantado
        return fingers == [1, 0, 0, 0, 0]
    
    def is_peace_sign(self, landmarks: np.ndarray, handedness: str = "Right") -> bool:
        """
        Verifica se o gesto é "paz" (indicador e médio levantados).
        
        Args:
            landmarks: Array de landmarks
            handedness: Lateralidade da mão
            
        Returns:
            True se for sinal de paz
        """
        count, fingers = self.classify(landmarks, handedness)
        
        # Paz: indicador e médio levantados, outros fechados
        return fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0
    
    def is_rock_sign(self, landmarks: np.ndarray, handedness: str = "Right") -> bool:
        """
        Verifica se o gesto é "rock" (indicador e mindinho levantados).
        
        Args:
            landmarks: Array de landmarks
            handedness: Lateralidade da mão
            
        Returns:
            True se for sinal de rock
        """
        count, fingers = self.classify(landmarks, handedness)
        
        # Rock: indicador e mindinho levantados
        return fingers[1] == 1 and fingers[4] == 1 and fingers[2] == 0 and fingers[3] == 0


class AdvancedGestureClassifier(GestureClassifier):
    """
    Classificador avançado que pode ser estendido com modelos de ML.
    Herda do GestureClassifier básico e adiciona funcionalidades.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa o classificador avançado.
        
        Args:
            model_path: Caminho para modelo treinado (opcional)
        """
        super().__init__()
        self.model = None
        self.model_path = model_path
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, path: str):
        """
        Carrega um modelo de ML treinado.
        
        Args:
            path: Caminho para o arquivo do modelo
        """
        try:
            # Placeholder para carregar modelo TensorFlow/PyTorch
            # from tensorflow import keras
            # self.model = keras.models.load_model(path)
            print(f"Modelo carregado de: {path}")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            self.model = None
    
    def classify_with_model(self, landmarks: np.ndarray) -> Tuple[int, float]:
        """
        Classifica usando modelo de ML (se disponível).
        
        Args:
            landmarks: Array de landmarks
            
        Returns:
            Tupla (classe_predita, confiança)
        """
        if self.model is None:
            # Fallback para classificação baseada em regras
            count, _ = self.classify(landmarks)
            return count, 1.0
        
        # Preparar input para o modelo
        input_data = landmarks.flatten().reshape(1, -1)
        
        # Fazer predição
        # prediction = self.model.predict(input_data)
        # predicted_class = np.argmax(prediction)
        # confidence = np.max(prediction)
        
        # Placeholder
        predicted_class = 0
        confidence = 0.0
        
        return predicted_class, confidence
    
    def extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extrai features dos landmarks para treinamento de modelos.
        
        Args:
            landmarks: Array de landmarks (21, 3)
            
        Returns:
            Array de features processadas
        """
        if landmarks is None:
            return np.zeros(63)
        
        # Normalizar em relação ao pulso (landmark 0)
        wrist = landmarks[0]
        normalized = landmarks - wrist
        
        # Flatten para vetor 1D
        features = normalized.flatten()
        
        return features
