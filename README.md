# Hand Gesture Recognition

Projeto de visão computacional que identifica números (0-5) através de gestos de mão capturados pela webcam usando MediaPipe e OpenCV.

## Demonstração

O sistema detecta a mão em tempo real e conta quantos dedos estão levantados, exibindo o número correspondente na tela.

## Tecnologias

- **Python 3.8+**
- **OpenCV** - Captura e processamento de vídeo
- **MediaPipe** - Detecção de mãos e landmarks
- **NumPy** - Operações numéricas

## Instalação

```bash
# Clonar repositório
git clone https://github.com/2002Meezy/Handrecognizer.git
cd handrecognizer

# Criar ambiente virtual (recomendado)
python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac

# Instalar dependências
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

- Mostre sua mão para a webcam
- O sistema identificará quantos dedos estão levantados (0-5)
- Pressione **Q** para sair

## Estrutura do Projeto

```
handrecognizer/
├── main.py               # Script principal
├── hand_detector.py      # Classe de detecção de mãos (MediaPipe)
├── gesture_classifier.py # Classificador de gestos
├── requirements.txt      # Dependências
└── README.md             # Este arquivo
```

## Como Funciona

1. **Captura**: OpenCV captura frames da webcam em tempo real
2. **Detecção**: MediaPipe detecta a mão e extrai 21 pontos (landmarks)
3. **Classificação**: Algoritmo analisa posição dos dedos para determinar quais estão levantados
4. **Exibição**: Número identificado é mostrado na tela com informações visuais

## Landmarks da Mão (MediaPipe)

```
        8   12  16  20
        |   |   |   |
    4   7   11  15  19
    |   |   |   |   |
    3   6   10  14  18
    |   |   |   |   |
    2   5   9   13  17
     \  |   |   |   |
      1 +---+---+---+
       \    |
        0---+ (pulso)
```

## Licença

MIT License
