# chaebot-AI
---
이 프로젝트는 음악 파일(YouTube 링크 또는 MP3)에서 드럼 소리를 분리하고, 타격 시점 및 악기를 분석하여 **자동으로 채보를 생성**하는 Python 기반의 도구입니다. 최종 결과는 **PDF 악보**로 출력됩니다.

---

## 📁 파일 구성
inference/
└── all_process.py # 드럼 분리 → BPM 추출 → onset 분할 → ViT 예측 → 악보 생성 전체 과정이 하나로 통합되어 있습니다
train/
└── train_ViT.ipynb
└── train_ResNet101.ipynb
└── train_Unet.ipynb

---

## 🎯 checkpoint_ViT.pth
- 예측을 위해 학습된 ViT 모델 (직접 다운로드 필요)
- https://drive.google.com/file/d/11D6qrbZKqx7cvYrpE4yQIakvJP4Qq5uy/view?usp=sharing

---

## 📦 설치 방법

### 1. 시스템 패키지 설치 (PDF 악보 생성을 위한 LilyPond)
'''python
sudo apt-get update && sudo apt-get install -y lilypond
'''
### 2. Python 패키지 설치

'''python
pip install yt_dlp essentia spleeter demucs pydub music21 pywavelets torch torchvision torchaudio timm librosa soundfile pandas
'''
