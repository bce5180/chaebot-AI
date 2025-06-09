# chaebot-AI
 chaebot-AI는 음악 파일(YouTube 링크 또는 MP3)에서 드럼 소리를 분리하고, <br>
 타격 시점 및 악기를 분석하여 **자동으로 채보를 생성**하는 Python 기반의 프로세스입니다. 최종 결과는 **PDF 악보**로 출력됩니다.
<br>
<br>
## 📁 파일 구성

inference/<br>
└── all_process.py<br>
(드럼 분리 → BPM 추출 → onset 분할 → ViT 예측 → 악보 생성 전체 과정이 하나로 통합되어 있습니다)
<br>
<br>
train/<br>
└── train_ViT.ipynb<br>
└── train_ResNet101.ipynb<br>
└── train_Unet.ipynb<br>
(데이터셋 변환 - 로드 - 학습 설정 - 학습 - 성능 그래프 까지의 모든 과정이 담겨있습니다)
<br>
<br>
## 🎯 checkpoint_ViT.pth
- 예측을 위해 학습된 ViT 모델 (직접 다운로드 필요)
- https://drive.google.com/file/d/11D6qrbZKqx7cvYrpE4yQIakvJP4Qq5uy/view?usp=sharing
<br>
<br>
## 📦 설치 방법
### 1. 시스템 패키지 설치 (PDF 악보 생성을 위한 LilyPond)

<br>
'''python
sudo apt-get update && sudo apt-get install -y lilypond
'''
<br>

### 2. Python 패키지 설치

<br>
'''python
pip install yt_dlp essentia spleeter demucs pydub music21 pywavelets torch torchvision torchaudio timm librosa soundfile pandas
'''
<br>
