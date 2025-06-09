# 시스템 관련
import os
import shutil
import subprocess
import re
# 오디오 처리
import librosa
import torchaudio
import essentia
from essentia.standard import MonoLoader, RhythmExtractor2013
import soundfile as sf
from scipy.signal import butter, filtfilt
import pywt
from scipy.io import wavfile
# 소스 분리 및 다운로드
import yt_dlp
from demucs.pretrained import get_model
from demucs.apply import apply_model
from spleeter.separator import Separator
# 모델 및 학습
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import timm
# 악보 처리
from music21 import stream, note, chord, tempo, metadata, instrument, environment
# 기타
import numpy as np
import pandas as pd
from pydub import AudioSegment

class ViTForCWTMultiLabel(nn.Module):
    def __init__(self, num_classes):
        super(ViTForCWTMultiLabel, self).__init__()

        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            img_size=(128, 1024),
            in_chans=3,
            num_classes=num_classes
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # (B, 1, H, W) → (B, 3, H, W)
        return self.vit(x)  # logits 반환 (sigmoid는 inference 시 따로 적용)
    
class chaebot:
    def __init__(self, mp3_file_path = None, youtube_link = None, model_path = None): # mp3_file: 파일 경로 / youtube_link: 링크 문자열
        self.mp3_file_path = mp3_file_path
        self.youtube_link = youtube_link
        self.bpm = None
        self.drum_wav_file_path = None
        self.chunked_file_path = None
        self.chunk_length_ms = 0
        self.predictions = []
        self.title = "my music"
        self.model_path = model_path

    def save_mp3_file_path(self):  # mp3 파일을 지정된 경로에 저장
        # mp3 파일을 지정된 경로에 저장
        save_path = "music.mp3"
        with open(save_path, 'wb') as f:
            f.write(self.mp3_file_path)

        # mp3 파일 경로 업데이트
        self.mp3_file_path = save_path

        # 그리고 악보의 title을 미리 지정 (확장자를 제거한 파일 이름만 추출)
        self.title = os.path.splitext(self.mp3_file_path)[0][:20]

    def save_youtube_link_to_mp3(self): # 유튜브 영상을 mp3 파일로 변환
        # yt-dlp를 사용하여 오디오 다운로드
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'output.%(ext)s',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(self.youtube_link, download=True)
            file_name = ydl.prepare_filename(info_dict)

            # 유튜브 동영상 제목 가져오기
            video_title = info_dict.get('title', 'unknown_title')
            self.title = video_title[:20]

        # 다운로드된 파일을 mp3로 변환
        save_path = 'music.mp3'
        subprocess.run(['ffmpeg', '-i', file_name, '-q:a', '0', '-map', 'a', save_path])

        # 원본 오디오 파일 삭제
        os.remove(file_name)

        # mp3 파일 경로 업데이트
        self.mp3_file_path = save_path

    def extract_bpm(self): # 음악의 bpm을 추출
        # 오디오 파일 로드
        loader = MonoLoader(filename=self.mp3_file_path)
        audio = loader()

        # bpm 추출 및 저장
        rhythm_extractor = RhythmExtractor2013()
        bpm, ticks, confidence, estimates, bpmIntervals = rhythm_extractor(audio)
        self.bpm = bpm

    def transform_wav_to_drum(self):  # 음악에서 드럼 소리만 추출

        def selective_freq_boost(waveform, sr=44100, low_cut=250, high_cut=5000, low_gain=1.2, high_gain=1.5):
          b_low, a_low = butter(2, low_cut / (sr / 2), btype='low', analog=False)
          low_boosted = filtfilt(b_low, a_low, waveform) * low_gain

          b_high, a_high = butter(2, high_cut / (sr / 2), btype='high', analog=False)
          high_boosted = filtfilt(b_high, a_high, waveform) * high_gain

          enhanced_waveform = low_boosted + high_boosted
          return enhanced_waveform

        # Hybrid Demucs 모델 로드
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = get_model('htdemucs').to(device)

        # 오디오 파일 로드
        waveform, sample_rate = torchaudio.load(self.mp3_file_path)
        waveform = waveform.to(device)

        # 모델 적용 (오디오 분리)
        sources = apply_model(model, waveform[None, :, :], split=True, device=device)

        # 드럼 요소 추출
        drum_audio = sources[0, 0, :, :].cpu()

        # 필터 적용 (하이햇 강조 + 킥/  유지)
        drum_audio_boosted = torch.tensor(selective_freq_boost(drum_audio.squeeze(0).numpy(), sr=sample_rate))

        # 2D 텐서로 변환
        if drum_audio_boosted.dim() == 1:
            drum_audio_boosted = drum_audio_boosted.unsqueeze(0)

        # 드럼 wav 저장 경로 설정
        self.drum_wav_file_path = os.path.join("music", "drums.wav")
        os.makedirs("music", exist_ok=True)
        torchaudio.save(self.drum_wav_file_path, drum_audio_boosted, sample_rate)

    def cut_drum_intro(self): # 음악에서 드럼이 없는 앞부분 자르기
        # 드럼 트랙 로드
        y, sr = librosa.load(self.drum_wav_file_path)

        # RMS 값 계산
        rms = librosa.feature.rms(y=y)[0]

        # RMS 평균의 1.5배를 임계치로 설정
        threshold = np.mean(rms) * 1.5

        # 임계치를 처음 초과하는 샘플 인덱스 찾기
        start_sample = np.argmax(rms > threshold) * 512  # RMS는 512 프레임 간격으로 계산됨

        # 음성을 임계치 이전 신호를 잘라내기
        y_trimmed = y[start_sample:]

        sf.write(self.drum_wav_file_path, y_trimmed, sr)

    def adjust_onsets_to_pattern_fixed(self, bpm=60, beats_per_measure=4, target_counts=[4, 8, 12]): # 음악의 onset들을 추출하고, 마디별로 부족한 부분이 있다면 onset을 삽입하여 패턴을 맞춤

        # 오디오 파일 로드
        audio, sr = librosa.load(self.drum_wav_file_path, sr=None)

        # onset 감지 → 프레임 단위 → 시간(초) 단위로 변환
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        onset_times = np.array([int(time * 1000) for time in onset_times])  # ms 단위로 변환

        # 한 마디에 몇 초인지 계산
        seconds_per_measure = (60 / bpm) * beats_per_measure

        # 전체 오디오에서 가장 마지막 onset 시간
        total_time = max(onset_times)

        # 총 마디수 계산
        num_measures = int(np.ceil(total_time / seconds_per_measure))

        # 각 마디의 시작 시간 리스트 생성 (0, 한 마디 길이, 두 마디 길이, ...)
        measure_times = np.arange(0, num_measures * seconds_per_measure, seconds_per_measure)

        original_onsets = np.copy(onset_times)
        new_onsets = []

        # 각 마디에 대해 처리
        for i in range(num_measures):
            start_time = measure_times[i]
            end_time = start_time + seconds_per_measure

            # 현재 마디에 포함된 onset들만 추출해서 시간순으로 정렬
            mask = (onset_times >= start_time) & (onset_times < end_time)
            measure_onsets = np.sort(onset_times[mask])

            # 해당 마디에 onset이 하나도 없으면 건너뜀
            if len(measure_onsets) == 0:
                continue

            # onset 개수가 불규칙적일 경우, 규칙적이도록 맞춰줌
            target_count = 8
            current_count = len(measure_onsets)

            # target_count에 맞게 onset 간격을 추가하는 로직
            if current_count < target_count:
                temp_onsets = list(measure_onsets)

                while len(temp_onsets) < target_count:
                    if len(temp_onsets) < 2:
                        break
                    max_gap_idx = np.argmax(np.diff(temp_onsets))
                    new_onset = (temp_onsets[max_gap_idx] + temp_onsets[max_gap_idx + 1]) / 2
                    temp_onsets.insert(max_gap_idx + 1, new_onset)
                    
                new_onsets.extend(temp_onsets)
            else:
                new_onsets.extend(measure_onsets)

        final_onsets = np.sort(np.array(new_onsets))
        return final_onsets, measure_times, seconds_per_measure

    def onset_segment_and_chunking(self, adjusted_onsets, max_duration=0.2, search_window=0.05): # onset 근처 segment만 16분음표 길이로 자르고, 피크 중심으로 보정해 저장
        # onset 시간을 초에서 ms로 변환
        onset_ms = adjusted_onsets

        # 16분음표 단위 chunk 길이 계산
        self.chunk_length_ms = int((60000 / self.bpm) / 4)

        # 저장 폴더 지정 및 없을 경우 생성
        self.chunked_file_path = "chunked_music/"
        if not os.path.exists(self.chunked_file_path):
            os.makedirs(self.chunked_file_path)

        # 오디오 불러오기
        y, sr = librosa.load(self.drum_wav_file_path, sr=None)
        total_duration_ms = int(len(y) / sr * 1000)
        base_name = os.path.splitext(os.path.basename(self.drum_wav_file_path))[0]

        segment_samples = int(self.chunk_length_ms / 1000.0 * sr)

        # chunk 단위로 전체 오디오를 순회
        for i, start_ms in enumerate(range(0, total_duration_ms, self.chunk_length_ms)):
            # 현재 chunk의 중심 위치를 기준으로 segment 추출
            center_sample = int((start_ms + self.chunk_length_ms // 2) / 1000 * sr)
            start_sample = max(0, center_sample - segment_samples // 2)
            end_sample = min(len(y), center_sample + segment_samples // 2)
            segment = y[start_sample:end_sample]

            # padding (길이 부족할 경우)
            if len(segment) < segment_samples:
                pad_width = segment_samples - len(segment)
                segment = np.pad(segment, (pad_width // 2, pad_width - pad_width // 2), mode='constant')

            # onset 근접성 판단
            is_onset = any(abs(start_ms - o) < (self.chunk_length_ms // 2) for o in onset_ms)

            if is_onset:
                # 이 chunk에 해당하는 onset 중 가장 가까운 것을 기준으로 피크 찾기
                closest_onset = min(onset_ms, key=lambda o: abs(o - start_ms))
                onset_sample = int(closest_onset / 1000 * sr)

                search_start = max(0, onset_sample - int(search_window * sr))
                search_end = min(len(y), onset_sample + int(search_window * sr))
                segment_search = y[search_start:search_end]

                peak_relative = np.argmax(np.abs(segment_search))
                peak_sample = search_start + peak_relative

                # 피크 중심으로 다시 segment 잘라줌
                start_sample = max(0, peak_sample - segment_samples // 2)
                end_sample = min(len(y), peak_sample + segment_samples // 2)
                segment = y[start_sample:end_sample]

                if len(segment) < segment_samples:
                    pad_width = segment_samples - len(segment)
                    segment = np.pad(segment, (pad_width // 2, pad_width - pad_width // 2), mode='constant')

            # 파일 이름 설정
            seg_name = f"{base_name}_chunk{i:04d}{'_on' if is_onset else ''}.wav"
            seg_path = os.path.join(self.chunked_file_path, seg_name)

            # float32 → int16로 변환 후 저장
            segment_int16 = (segment * 32767).astype(np.int16)
            wavfile.write(seg_path, sr, segment_int16)

    def onset_detection_and_save(self):
        adjusted_onsets, _, _ = self.adjust_onsets_to_pattern_fixed()
        self.onset_segment_and_chunking(adjusted_onsets)

    def predict(self): # 모델 예측
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ViTForCWTMultiLabel(num_classes=6)
        checkpoint = torch.load(self.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()  # 평가 모드로 전환

        files = os.listdir(self.chunked_file_path)
        a = 0
        model_output_list = []
        for file in files:
            if file.endswith('on.wav'):
                sr = 22050
                target_freq_bins = 128
                target_time_steps = 1024

                path = os.path.join(self.chunked_file_path, file)
                
                audio, sr = librosa.load(path, sr=sr)

                # 윈도우 함수 적용 (예: Hamming 윈도우)
                window = np.hamming(len(audio))  # 오디오 길이만큼 Hamming 윈도우 생성
                audio = audio * window  # 윈도우 적용

                # CWT 스펙트로그램 생성
                scales = np.arange(1, target_freq_bins + 1) # 주파수축 설정
                cwt_matrix, _ = pywt.cwt(audio, scales, 'morl')
                cwt_magnitude = np.abs(cwt_matrix)

                time_len = cwt_magnitude.shape[1]
                if time_len < target_time_steps:
                    pad_width = target_time_steps - time_len
                    cwt_magnitude = np.pad(cwt_magnitude, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    cwt_magnitude = cwt_magnitude[:, :target_time_steps]

                cwt_tensor = torch.tensor(cwt_magnitude).unsqueeze(0).unsqueeze(0).float().to(device)

                # 모델에 입력하여 예측 수행
                with torch.no_grad():
                    class_output = model(cwt_tensor)
                    # Classification output을 이진화해서 저장
                    softmax_output = F.softmax(class_output, dim=1)
                    softmax_output_values = softmax_output.detach().cpu().numpy().flatten().tolist()
                    softmax_output_rounded = [round(v, 6) for v in softmax_output_values]
                    model_output_list.append(softmax_output_rounded)
                    print(softmax_output_rounded)

                    thresholds = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]).to(softmax_output.device)

                    binary_prediction = (softmax_output > thresholds).int().detach().cpu().numpy()
                self.predictions.append(binary_prediction.tolist()) # 예측 결과를 리스트에 저장
                a += 1
            else:
                # 예측 결과를 무음으로 만들어서 저장시킴
                class_output = torch.zeros((1, 8), device=device)
                binary_prediction = (class_output > 0.5).int().detach().cpu().numpy()
                self.predictions.append(binary_prediction.tolist())
                a += 1
        df = pd.DataFrame(model_output_list)
        df.to_csv('predictions.csv', index=False)

    def remove_non_ascii(self):
        self.title =  ''.join(filter(lambda x: ord(x) < 128, self.title))
        self.title = self.title.strip()

    def make_pdf(self):

        drum_lilypond_code = r'''
    \version "2.22.1"
    \header {
      title = "''' + self.title + r'''"
    }

    \layout {
      \context {
        \DrumVoice
        \omit Rest
      }
    }

    '''

        up = []
        down = []

        index_drum_dict = {
            0: "crashcymbal16", # 크래시
            1: "cymr16", # 하이햇
            2: "cb16", # 라이드
            3: "snare16", # 스네어
            4: "tommh16", # 탐
            5: "bd16", # 베이스드럼(킥)
        }

        # self.predictions의 예측 결과를 기반으로 악보에 음표 추가
        for prediction in self.predictions:
            prediction  = prediction[0]  # 3중 리스트에서 2중 리스트로 변환
            if prediction[5] == 1:
                down.append(index_drum_dict[5])
            else:
                down.append("r16")

            up_sum = sum(prediction[0:5])

            if up_sum == 0:
                up.append("r16")
            elif up_sum == 1:
                for i in range(0,5):
                    if prediction[i] == 1:
                        up.append(index_drum_dict[i])
                        break
            else:
                overlap_text = "<"
                for i in range(0,5):
                    if prediction[i] == 1:
                        overlap_text += index_drum_dict[i][:-2]
                        overlap_text += " "

                overlap_text = overlap_text[:-1]
                overlap_text += ">16"
                up.append(overlap_text)

        drum_lilypond_code += f'''
    up = \drummode {{
      {" ".join(up)}
    }}

    down = \drummode {{
      {" ".join(down)}
    }}

    '''

        # LilyPond 코드의 끝 부분 설정
        drum_lilypond_code += r'''

    \score {
      \new DrumStaff <<
        \new DrumVoice = "up" {
          \voiceOne
          \tempo 4 = ''' + str(round(self.bpm)) + r'''
          \up
        }
        \new DrumVoice = "down" { \voiceTwo \down }
      >>
    }
    '''

        print(drum_lilypond_code)

        # LilyPond 파일로 저장
        lilypond_filename = "drum_pattern.ly"
        with open(lilypond_filename, "w") as file:
            file.write(drum_lilypond_code)

        # PDF로 변환
        os.system(f"lilypond {lilypond_filename}")

        # 변환된 PDF 파일 이름을 출력
        pdf_filename = lilypond_filename.replace(".ly", ".pdf")

    def delete_temp_files(self):
        # 삭제할 폴더 목록
        folders_to_delete = ['chunked_music', 'music', 'pretrained_models']

        # 삭제할 파일 목록
        files_to_delete = ['drum_pattern.ly', 'music.mp3']

        # 현재 작업 경로
        current_path = os.getcwd()

        # 폴더와 그 안의 모든 파일 및 폴더 삭제
        for folder in folders_to_delete:
            folder_path = os.path.join(current_path, folder)
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
                print(f"Deleted folder and its contents: {folder_path}")
            else:
                print(f"Folder not found: {folder_path}")

        # 개별 파일 삭제
        for file in files_to_delete:
            file_path = os.path.join(current_path, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"File not found: {file_path}")

    def main(self): # 모든 실행을 관리
        # try:
        # 1. 받은 파일들을 처리하여 self.mp3_file에 저장한다
        if self.youtube_link is not None:
            self.save_youtube_link_to_mp3()
        elif self.mp3_file is not None:
            self.save_mp3_file()
        else:
            print("파일이 둘다 None으로 입력 됨") # 오류 처리
            return None

        # 2. bpm을 추출한다
        self.extract_bpm()

        # 3. mp3 파일에서 드럼 소리만 추출하여 drum.wav로 변환한다
        self.transform_wav_to_drum()

        # 4. 임계값 처리를 통해 드럼이 없는 전주 부분은 자른다
        self.cut_drum_intro()

        # 5. wav 파일에 onset detection 알고리즘을 사용하여 타격 시점을 알아내고, 16분 음표 길이로 잘라 저장한다.
        self.onset_detection_and_save()

        # 6. 잘린 파일들을 모두 STFT 변환 시킨 후, 256*256으로 resize 시키고 model에 넣어 예측값을 뽑아낸다
        self.predict()

        # 7. 예측 결과를 토대로 악보를 그린 후 pdf를 저장한다
        self.remove_non_ascii()
        self.make_pdf()

        # 8. 채보 과정 중에 생성된 temp 파일들은 삭제한다
        self.delete_temp_files()

        return None

if __name__ == "__main__":
    # youtube 링크일 경우
    youtube_link = "https://www.youtube.com/watch?v=Emj1_vmbFD8"
    c = chaebot(youtube_link= youtube_link, model_path = "./checkpoint_ViT.pth")
    c.main()

    # # mp3 파일일 경우
    # c = chaebot(mp3_file_path="my_music.mp3")
    # c.main()