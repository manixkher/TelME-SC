import argparse
import torch
import cv2
import numpy as np
import sounddevice as sd
import queue
import threading
import time
import whisper
import sys
import os
import warnings
import librosa
import soundfile as sf
from collections import deque
from dataclasses import dataclass

import tempfile
import os
import wave
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
warnings.filterwarnings("ignore", message=".*__audioread_load.*")

from transformers import RobertaTokenizer, RobertaModel, AutoProcessor, AutoImageProcessor, Data2VecAudioModel, TimesformerModel

affwild2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, affwild2_path)
from models.model import Teacher_model, Student_Audio, Student_Video
from utils.utils import *

meld_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MELD'))
sys.path.insert(0, meld_path)
from models_new.TelMESC import ASF_SCMMTelMESC

import sys
import importlib.util
models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models.py'))
spec = importlib.util.spec_from_file_location("vad_models", models_path)
vad_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vad_models)
crnn = vad_models.crnn

VIDEO_FPS = 24
VAD_SAMPLE_RATE = 22050
ASR_SAMPLE_RATE = 16000
AUDIO_BLOCK_SECONDS = 0.2
VIDEO_BUFFER_SIZE = 16
SPEECH_SEGMENT_MIN_FRAMES = 8
AUDIO_MAX_LENGTH = 400000

VAD_START_THRESHOLD = 0.4
VAD_END_THRESHOLD = 0.05

def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, 
                             text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=2, padding=5):
    """Draw text with background.

    Args:
        frame (numpy.ndarray): Frame
        text (str): Text
        position (tuple): Position
        font (int, optional): Font. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float, optional): Font scale. Defaults to 0.7.
        text_color (tuple, optional): Text color. Defaults to (255, 255, 255).
        bg_color (tuple, optional): Background color. Defaults to (0, 0, 0).
        thickness (int, optional): Thickness. Defaults to 2.
        padding (int, optional): Padding. Defaults to 5.

    Returns:
        int: Text height + baseline + padding * 2
    """
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    bg_x1 = x - padding
    bg_y1 = y - text_height - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + baseline + padding
    
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    cv2.putText(frame, text, position, font, font_scale, text_color, thickness)
    
    return text_height + baseline + padding * 2

teacher_model = None
student_audio_model = None
student_video_model = None
telmec_model = None
audio_processor = None
video_processor = None

def parse_args():
    """Parse arguments.

    Returns:
        argparse.Namespace: Arguments
    """
    parser = argparse.ArgumentParser(description="Live streaming emotion inference with TelMESC")
    parser.add_argument('--model_path', type=str, default='save_model/exp_fixed_telmesc_affwild2_epoch_4_concurrent.bin', help='Path to TelMESC model weights')
    parser.add_argument('--teacher_path', type=str, default='save_model/exp_new_teacher_affwild2_whisper_epoch_4.bin', help='Path to AffWild2 teacher model')
    parser.add_argument('--student_audio_path', type=str, default='save_model/student_audio/exp_student_audio_best_affwild2.bin', help='Path to AffWild2 student audio model')
    parser.add_argument('--student_video_path', type=str, default='save_model/student_video/exp_student_visual_best_affwild2.bin', help='Path to AffWild2 student video model')
    parser.add_argument('--asr_path', type=str, default='base.en', help='Whisper model name or path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device for inference')
    parser.add_argument('--gate_mode', default='learned', choices=['learned', 'uniform'], help='gate mode: learned or uniform routing')
    return parser.parse_args()

def load_vad_model(vad_path, device):
    """Load VAD model.

    Args:
        vad_path (str): VAD model path
        device (str): Device for inference

    Returns:
        crnn: VAD model
        int: Speech label index
    """
    model_path = os.path.join(os.path.dirname(__file__), '..', 'pretrained_models', 'sre', 'model.pth')
    model = crnn(inputdim=64, outputdim=2, pretrained_from=model_path)
    model.eval()
    model.to(device)
    
    encoder_path = os.path.join(os.path.dirname(__file__), '..', 'pretrained_models', 'labelencoders', 'students.pth')
    encoder = torch.load(encoder_path, map_location=device, weights_only=False)
    
    speech_label_idx = np.where('Speech' == encoder.classes_)[0].squeeze()
    print(f"VAD: Speech label index = {speech_label_idx}")
    
    return model, speech_label_idx

def vad_infer(audio, vad_model, speech_label_idx, device):
    """VAD inference.

    Args:
        audio (numpy.ndarray): Audio
        vad_model (crnn): VAD model
        speech_label_idx (int): Speech label index
        device (str): Device for inference

    Returns:
        float: Speech probability
    """
    t_start = time.time()
    
    EPS = np.spacing(1)
    lms = np.log(
        librosa.feature.melspectrogram(
            y=audio.astype(np.float32), 
            sr=VAD_SAMPLE_RATE, 
            n_fft=2048, 
            n_mels=64,
            hop_length=int(VAD_SAMPLE_RATE * 0.02),
            win_length=int(VAD_SAMPLE_RATE * 0.04)
        ) + EPS
    ).T
    
    lms = torch.tensor(lms, dtype=torch.float32).unsqueeze(0).to(device)
    t_feature = time.time()
    
    with torch.no_grad():
        prediction_tag, prediction_time = vad_model(lms)
        speech_prob = prediction_time[..., speech_label_idx].cpu().numpy()
        speech_prob = speech_prob.mean()
    t_model = time.time()
    
    if hasattr(vad_infer, 'frame_count'):
        vad_infer.frame_count += 1
    else:
        vad_infer.frame_count = 1
        
    if vad_infer.frame_count % 50 == 0:
        print(f"  VAD: feature={((t_feature-t_start)*1000):.1f}ms, model={((t_model-t_feature)*1000):.1f}ms")
    
    return speech_prob

def load_models(args, device):
    """Load models.

    Args:
        args (argparse.Namespace): Arguments
        device (str): Device for inference
    """
    global teacher_model, student_audio_model, student_video_model, telmec_model, audio_processor, video_processor
        
    @dataclass
    class Config():
        mask_time_length: int = 3
    
    init_config = Config()
    clsNum = 7
    
    text_model = "roberta-large"
    print("Loading AffWild2 teacher model (RoBERTa)...")
    teacher_model = Teacher_model(text_model, clsNum)
    teacher_model.load_state_dict(torch.load(args.teacher_path, map_location=device))
    for para in teacher_model.parameters():
        para.requires_grad = False
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    audio_model = "facebook/data2vec-audio-base-960h"
    print("Loading AffWild2 student audio model...")
    student_audio_model = Student_Audio(audio_model, clsNum, init_config)
    student_audio_model.load_state_dict(torch.load(args.student_audio_path, map_location=device))
    for para in student_audio_model.parameters():
        para.requires_grad = False
    student_audio_model = student_audio_model.to(device)
    student_audio_model.eval()

    video_model = "facebook/timesformer-base-finetuned-k400"
    print("Loading AffWild2 student video model...")
    student_video_model = Student_Video(video_model, clsNum)
    student_video_model.load_state_dict(torch.load(args.student_video_path, map_location=device))
    for para in student_video_model.parameters():
        para.requires_grad = False
    student_video_model = student_video_model.to(device)
    student_video_model.eval()

    print("Initializing audio and video processors...")
    audio_processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
    video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    print("Loading TelMESC fusion model...")
    hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
    telmec_model = ASF_SCMMTelMESC(
        clsNum, hidden_size, beta_shift, dropout_prob, num_head,
        scmm_hidden_size=768,
        scmm_max_context=12,
        scmm_dropout=0.1,
        scmm_path_dropout=0.0
    )
    
    asf_checkpoint_path = './AffWild2/save_model/total_fusion_affwild2_telme.bin'
    if os.path.exists(asf_checkpoint_path):
        print(f"Loading pre-trained ASF weights from {asf_checkpoint_path}")
        asf_state_dict = torch.load(asf_checkpoint_path, map_location=device)
        telmec_model.load_asf_weights(asf_state_dict)
        print("ASF weights loaded successfully")
    
    if os.path.exists(args.model_path):
        load_TelMESC(telmec_model, args.model_path, target_batch_size=1)
    else:
        print(f"Warning: TelMESC model not found at {args.model_path}")
        print("Using randomly initialized TelMESC model")
    
    telmec_model = telmec_model.to(device)
    telmec_model.eval()

def load_TelMESC(model, model_path, target_batch_size=1):
    """Load TelMESC model.

    Args:
        model (nn.Module): Model
        model_path (str): Model path
        target_batch_size (int, optional): Target batch size. Defaults to 1.
    """
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        return False
    
    state_dict = torch.load(model_path, map_location='cpu')
    
    buffer_keys = ['scmm.context_buffer', 'scmm.buffer_mask', 'scmm.buffer_indices']
    buffer_size_mismatch = False
    
    for key in buffer_keys:
        if key in state_dict:
            saved_shape = state_dict[key].shape
            current_shape = model.state_dict()[key].shape
            
            if saved_shape[0] != current_shape[0]:
                buffer_size_mismatch = True
                print(f"Buffer size mismatch detected for {key}:")
                print(f"  Saved: {saved_shape}, Current: {current_shape}")
                print(f"  Resizing buffers to batch size {target_batch_size}")
                break
    
    if buffer_size_mismatch:
        for key in buffer_keys:
            if key in state_dict:
                saved_tensor = state_dict[key]
                if key == 'scmm.context_buffer':
                    new_tensor = torch.zeros(target_batch_size, saved_tensor.shape[1], saved_tensor.shape[2])
                elif key == 'scmm.buffer_mask':
                    new_tensor = torch.zeros(target_batch_size, saved_tensor.shape[1], dtype=torch.bool)
                elif key == 'scmm.buffer_indices':
                    new_tensor = torch.zeros(target_batch_size, dtype=torch.long)
                
                if saved_tensor.shape[0] > 0:
                    copy_size = min(target_batch_size, saved_tensor.shape[0])
                    new_tensor[:copy_size] = saved_tensor[:copy_size]
                
                state_dict[key] = new_tensor
    
    try:
        model.load_state_dict(state_dict)
        print(f"Successfully loaded TelMESC model from {model_path}")
        if buffer_size_mismatch:
            print("Note: Buffer sizes were adjusted to match target batch size")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def encode_single_utterance(text, tokenizer, max_length=511):
    """Encode single utterance.

    Args:
        text (str): Text
        tokenizer (RobertaTokenizer): Tokenizer
        max_length (int, optional): Max length. Defaults to 511.
    """
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    return ids + [tokenizer.mask_token_id]

def extract_text_features(text, tokenizer, teacher_model, device):
    """Extract text features.

    Args:
        text (str): Text
        tokenizer (RobertaTokenizer): Tokenizer
        teacher_model (nn.Module): Teacher model
        device (str): Device for inference
    """
    if not text.strip() or teacher_model is None:
        return torch.zeros((1, 768), device=device)
    
    try:
        t_start = time.time()
        token_ids = encode_single_utterance(text, tokenizer)
        token_tensor = torch.tensor([token_ids], device=device)
        attention_mask = torch.ones_like(token_tensor)
        t_tokenize = time.time()
        
        with torch.no_grad():
            text_hidden, _ = teacher_model(token_tensor, attention_mask)
        t_model = time.time()
        
        if hasattr(extract_text_features, 'frame_count'):
            extract_text_features.frame_count += 1
        else:
            extract_text_features.frame_count = 1
            
        if extract_text_features.frame_count % 30 == 0:
            print(f"  Text: tokenize={((t_tokenize-t_start)*1000):.1f}ms, model={((t_model-t_tokenize)*1000):.1f}ms")
        
        return text_hidden
    except Exception as e:
        print(f"Text feature extraction error: {e}")
        return torch.zeros((1, 768), device=device)

def audio_embedding_from_all_audio(audio_16k, model, processor, device, chunk_seconds=5.0):
    """Extract audio features.

    Args:
        audio_16k (numpy.ndarray): Audio
        model (nn.Module): Model
        processor (AutoProcessor): Processor
        device (str): Device for inference
        chunk_seconds (float, optional): Chunk seconds. Defaults to 5.0.
    """
    if len(audio_16k) == 0 or model is None or processor is None:
        return torch.zeros((1, 768), device=device)
    
    try:
        chunk_len = int(ASR_SAMPLE_RATE * chunk_seconds)
        chunks = []
        for start in range(0, len(audio_16k), chunk_len):
            end = min(start + chunk_len, len(audio_16k))
            chunks.append(audio_16k[start:end])
        
        embs = []
        with torch.no_grad():
            for c in chunks:
                inputs = processor(c, sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt")
                audio_inputs = inputs["input_values"].to(device)
                emb, _ = model(audio_inputs)
                embs.append(emb)
        
        if embs:
            return torch.mean(torch.stack(embs, dim=0), dim=0)
        else:
            return torch.zeros((1, 768), device=device)
            
    except Exception as e:
        print(f"Audio embedding error: {e}")
        return torch.zeros((1, 768), device=device)

def extract_audio_features(audio, student_audio_model, device):
    """Extract audio features.

    Args:
        audio (numpy.ndarray): Audio
        student_audio_model (nn.Module): Student audio model
        device (str): Device for inference
    """
    global audio_processor
    if audio is None or len(audio) == 0 or student_audio_model is None or audio_processor is None:
        return torch.zeros((1, 768), device=device)
    
    try:
        t_start = time.time()
        inputs = audio_processor(audio, sampling_rate=16000, return_tensors="pt")
        audio_inputs = inputs["input_values"].to(device)
        t_preprocess = time.time()
        
        with torch.no_grad():
            audio_hidden, _ = student_audio_model(audio_inputs)
        t_model = time.time()
        
        if hasattr(extract_audio_features, 'frame_count'):
            extract_audio_features.frame_count += 1
        else:
            extract_audio_features.frame_count = 1
            
        if extract_audio_features.frame_count % 30 == 0:
            print(f"  Audio: preprocess={((t_preprocess-t_start)*1000):.1f}ms, model={((t_model-t_preprocess)*1000):.1f}ms")
        
        return audio_hidden
    except Exception as e:
        print(f"Audio feature extraction error: {e}")
        return torch.zeros((1, 768), device=device)

def video_embedding_from_all_frames(frames_raw, model, processor, device, window=8, stride=8):
    """Extract video features.

    Args:
        frames_raw (list): Frames raw
        model (nn.Module): Model
        processor (AutoImageProcessor): Processor
        device (str): Device for inference
        window (int, optional): Window. Defaults to 8.
        stride (int, optional): Stride. Defaults to 8.
    """
    if not frames_raw or model is None or processor is None:
        return torch.zeros((1, 768), device=device)
    
    try:
        if len(frames_raw) < window:
            frames_raw = frames_raw + [frames_raw[-1]] * (window - len(frames_raw))
        
        idxs = list(range(0, max(1, len(frames_raw) - window + 1), stride))
        if idxs and idxs[-1] != len(frames_raw) - window:
            idxs.append(len(frames_raw) - window)
        
        embs = []
        with torch.no_grad():
            for i in idxs:
                clip = frames_raw[i:i+window]
                inputs = processor(clip, return_tensors="pt")
                video_inputs = inputs["pixel_values"].to(device)
                emb, _ = model(video_inputs)
                embs.append(emb)
        
        if embs:
            return torch.mean(torch.stack(embs, dim=0), dim=0)
        else:
            return torch.zeros((1, 768), device=device)
            
    except Exception as e:
        print(f"Video embedding error: {e}")
        return torch.zeros((1, 768), device=device)

def extract_video_features(frames, student_video_model, device):
    """Extract video features.

    Args:
        frames (list): Frames
        student_video_model (nn.Module): Student video model
        device (str): Device for inference

    Returns:
        torch.Tensor: Video features
    """
    global video_processor
    if not frames or student_video_model is None or video_processor is None:
        return torch.zeros((1, 768), device=device)
    
    try:
        if len(frames) >= 8:
            step = len(frames) // 8
            sampled_frames = [frames[i * step] for i in range(8)]
        else:
            sampled_frames = frames + [frames[-1]] * (8 - len(frames))
        
        t_start = time.time()
        inputs = video_processor(sampled_frames, return_tensors="pt")
        video_inputs = inputs["pixel_values"].to(device)
        t_preprocess = time.time()
        
        with torch.no_grad():
            video_hidden, _ = student_video_model(video_inputs)
        t_model = time.time()
        
        if hasattr(extract_video_features, 'frame_count'):
            extract_video_features.frame_count += 1
        else:
            extract_video_features.frame_count = 1
            
        if extract_video_features.frame_count % 10 == 0:
            print(f"  Video: preprocess={((t_preprocess-t_start)*1000):.1f}ms, model={((t_model-t_preprocess)*1000):.1f}ms")
        
        return video_hidden
    except Exception as e:
        print(f"Video feature extraction error: {e}")
        return torch.zeros((1, 768), device=device)

def asr_transcribe(audio, asr_model):
    """ASR transcribe.

    Args:
        audio (numpy.ndarray): Audio
        asr_model (whisper): ASR model

    Returns:
        str: Transcription
    """
    
    t_start = time.time()
    
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"temp_audio_{os.getpid()}.wav")
    
    try:
        with wave.open(temp_file, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(ASR_SAMPLE_RATE)
            wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())
        
        t_write = time.time()
        result = asr_model.transcribe(temp_file, fp16=False)
        t_transcribe = time.time()
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    if hasattr(asr_transcribe, 'frame_count'):
        asr_transcribe.frame_count += 1
    else:
        asr_transcribe.frame_count = 1
        
    if asr_transcribe.frame_count % 10 == 0:
        print(f"  ASR: write={((t_write-t_start)*1000):.1f}ms, transcribe={((t_transcribe-t_write)*1000):.1f}ms")
    
    return result['text']

class AudioStream:
    """Audio stream.

    Args:
        samplerate (int): Samplerate
        blocksize (int): Blocksize
    """
    def __init__(self, samplerate, blocksize):
        self.q = queue.Queue()
        self.stream = sd.InputStream(samplerate=samplerate, channels=1, blocksize=blocksize, callback=self.callback)
        self.running = False
        self.buffer = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()

    def callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        with self.lock:
            self.buffer = np.concatenate([self.buffer, indata[:, 0]])

    def start(self):
        self.running = True
        self.stream.start()

    def stop(self):
        self.running = False
        self.stream.stop()

    def get_audio(self, n_samples):
        with self.lock:
            if len(self.buffer) >= n_samples:
                out = self.buffer[:n_samples]
                self.buffer = self.buffer[n_samples:]
                return out
            else:
                return None

class VideoFrameBuffer:
    """Video frame buffer.

    Args:
        buffer_size (int, optional): Buffer size. Defaults to 8.
    """
    def __init__(self, buffer_size=8):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
    
    def add_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        self.buffer.append(frame_resized)
    
    def get_frames(self):
        frames = list(self.buffer)
        
        while len(frames) < self.buffer_size:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        return frames[:self.buffer_size]

class UtteranceBuffer:
    """Utterance buffer.

    Args:
        buffer_size (int, optional): Buffer size. Defaults to 8.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.audio_blocks = []
        self.frames = []
        self.blocks_below_end = 0
    
    def add_audio(self, block):
        self.audio_blocks.append(block)
    
    def add_frame(self, frame_raw):
        self.frames.append(frame_raw)
    
    def get_audio_concat(self):
        if not self.audio_blocks:
            return None
        return np.concatenate(self.audio_blocks, axis=0)
    
    def get_frames(self):
        return self.frames

class VADStateMachine:
    """VAD state machine.

    Args:
        start_threshold (float, optional): Start threshold. Defaults to 0.5.
        end_threshold (float, optional): End threshold. Defaults to 0.05.
        min_blocks (int, optional): Min blocks. Defaults to 3.
        end_hangover_blocks (int, optional): End hangover blocks. Defaults to 2.
    """
    def __init__(self, start_threshold=0.5, end_threshold=0.05,
                 min_blocks=3, end_hangover_blocks=2):
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.min_blocks = min_blocks
        self.end_hangover_blocks = end_hangover_blocks
        self.in_utterance = False
        self.blocks_in_utterance = 0
        self.buffer = UtteranceBuffer()
        self.last_speech_prob = 0.0

    def update_with_audio_block(self, speech_prob, audio_block):
        self.last_speech_prob = speech_prob
        utterance_started = False
        utterance_ended = False

        if not self.in_utterance:
            if speech_prob > self.start_threshold:
                self.in_utterance = True
                self.blocks_in_utterance = 0
                self.buffer.reset()
                utterance_started = True
                print(f"UTTERANCE STARTED (prob={speech_prob:.3f})")
        
        if self.in_utterance:
            self.buffer.add_audio(audio_block)
            self.blocks_in_utterance += 1
            
            if speech_prob < self.end_threshold and self.blocks_in_utterance >= self.min_blocks:
                self.buffer.blocks_below_end += 1
                if self.buffer.blocks_below_end >= self.end_hangover_blocks:
                    self.in_utterance = False
                    utterance_ended = True
                    print(f"UTTERANCE ENDED (prob={speech_prob:.3f}, duration={self.blocks_in_utterance} blocks)")
            else:
                self.buffer.blocks_below_end = 0

        return utterance_started, utterance_ended

def process_full_utterance(audio_22k, frames_raw, device, args, asr_model, roberta_tokenizer):
    """Process full utterance.

    Args:
        audio_22k (numpy.ndarray): Audio
        frames_raw (list): Frames raw
        device (str): Device for inference
        args (argparse.Namespace): Arguments
        asr_model (whisper): ASR model
        roberta_tokenizer (RobertaTokenizer): Tokenizer

    Returns:
        str: Top emotion
        str: Second emotion
        float: Top probability
        float: Second probability
        str: Text
        dict: Timing information
    """
    global teacher_model, student_audio_model, audio_processor, student_video_model, video_processor, telmec_model, VAD_SAMPLE_RATE, ASR_SAMPLE_RATE
    
    try:
        t_start = time.time()
        
        t_resample_start = time.time()
        audio_16k = librosa.resample(audio_22k, orig_sr=VAD_SAMPLE_RATE, target_sr=ASR_SAMPLE_RATE)
        t_resample_end = time.time()
        
        t_asr_start = time.time()
        text = asr_transcribe(audio_16k, asr_model)
        t_asr_end = time.time()
        print(f"ASR text: '{text}'")
        
        t_text_start = time.time()
        formatted_text = f"<s1> {text} </s> Now <s1> feels"
        text_hidden = extract_text_features(formatted_text, roberta_tokenizer, teacher_model, device)
        t_text_end = time.time()
        
        t_audio_start = time.time()
        audio_hidden = audio_embedding_from_all_audio(audio_16k, student_audio_model, audio_processor, device)
        t_audio_end = time.time()
        
        t_video_start = time.time()
        video_hidden = video_embedding_from_all_frames(frames_raw, student_video_model, video_processor, device)
        t_video_end = time.time()
        
        t_fusion_start = time.time()
        with torch.no_grad():
            pred_logits, _, _ = telmec_model(text_hidden, video_hidden, audio_hidden,
                                             return_scmm_info=True, gate_mode=args.gate_mode)
        t_fusion_end = time.time()
        
        probs = torch.softmax(pred_logits, dim=-1).cpu().numpy()[0]
        emotion_names = ["anger","disgust","fear","joy","neutral","sadness","surprise"]
        
        top_indices = np.argsort(probs)[-2:][::-1]
        top_emotion = emotion_names[top_indices[0]]
        second_emotion = emotion_names[top_indices[1]]
        top_prob = probs[top_indices[0]]
        second_prob = probs[top_indices[1]]
        
        print(f"PREDICTION: {top_emotion} ({top_prob:.3f}) | {second_emotion} ({second_prob:.3f})")
        
        print("Emotion probabilities:")
        for i, emotion_name in enumerate(emotion_names):
            print(f"  {emotion_name:8}: {probs[i]:.3f}")
        
        timing_info = {
            'resample': (t_resample_end - t_resample_start) * 1000,
            'asr': (t_asr_end - t_asr_start) * 1000,
            'text': (t_text_end - t_text_start) * 1000,
            'audio': (t_audio_end - t_audio_start) * 1000,
            'video': (t_video_end - t_video_start) * 1000,
            'fusion': (t_fusion_end - t_fusion_start) * 1000,
            'total': (t_fusion_end - t_start) * 1000
        }
        
        return top_emotion, second_emotion, top_prob, second_prob, text, timing_info
        
    except Exception as e:
        print(f"Error processing utterance: {e}")
        return "neutral", "neutral", 0.0, 0.0, "", {}

def main():
    args = parse_args()
    device = torch.device(args.device)

    print("Initializing tokenizer...")
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    speaker_list = ['<s1>', '<s2>', '<s3>', '<s4>', '<s5>', '<s6>', '<s7>', '<s8>', '<s9>']
    speaker_tokens_dict = {'additional_special_tokens': speaker_list}
    roberta_tokenizer.add_special_tokens(speaker_tokens_dict)

    load_models(args, device)
    
    print("Loading Whisper ASR model...")
    asr_model = whisper.load_model(args.asr_path, device=args.device)
    print("Loading VAD model...")
    vad_model, speech_label_idx = load_vad_model(None, device)

    audio_stream = AudioStream(samplerate=VAD_SAMPLE_RATE, blocksize=int(VAD_SAMPLE_RATE * AUDIO_BLOCK_SECONDS))
    audio_stream.start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    min_blocks = max(1, int(round(0.5 / AUDIO_BLOCK_SECONDS)))
    vad_sm = VADStateMachine(
        start_threshold=VAD_START_THRESHOLD,
        end_threshold=VAD_END_THRESHOLD,
        min_blocks=min_blocks,
        end_hangover_blocks=2
    )

    last_emotion = "x"
    last_second_emotion = "x"
    last_top_prob = 0.0
    last_second_prob = 0.0
    last_asr_text = ""
    last_timing_info = {}
    frame_count = 0
    fps_start_time = time.time()
    
    utterance_history = []
    max_utterance_history = 10
    
    print("Starting live inference... Press 'q' to quit")
    print("Camera window should open shortly...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            if vad_sm.in_utterance:
                vad_sm.buffer.add_frame(frame)

            audio_block = audio_stream.get_audio(int(VAD_SAMPLE_RATE * AUDIO_BLOCK_SECONDS))
            
            if audio_block is not None:
                speech_prob = vad_infer(audio_block, vad_model, speech_label_idx, device)
                
                started, ended = vad_sm.update_with_audio_block(speech_prob, audio_block)
                
                if started:
                    print("UTTERANCE STARTED")
                    
                if ended:
                    print("UTTERANCE ENDED")
                    full_audio_22k = vad_sm.buffer.get_audio_concat()
                    full_frames = vad_sm.buffer.get_frames()
                    
                    if full_audio_22k is not None and full_frames:
                        print(f"Processing utterance: {len(full_audio_22k)} audio samples, {len(full_frames)} frames")
                        top_emotion, second_emotion, top_prob, second_prob, text, timing_info = process_full_utterance(full_audio_22k, full_frames, device, args, asr_model, roberta_tokenizer)
                        
                        last_emotion = top_emotion
                        last_second_emotion = second_emotion
                        last_top_prob = top_prob
                        last_second_prob = second_prob
                        last_asr_text = text
                        last_timing_info = timing_info
                        
                        utterance_history.append({
                            'frame': frame_count,
                            'text': text,
                            'emotion': top_emotion,
                            'second_emotion': second_emotion,
                            'top_prob': top_prob,
                            'second_prob': second_prob,
                            'frames': len(full_frames),
                            'timing': timing_info
                        })
                        
                        if len(utterance_history) > max_utterance_history:
                            utterance_history.pop(0)
                        
                        print(f"\nUTTERANCE HISTORY (last {len(utterance_history)}):")
                        for i, hist in enumerate(utterance_history):
                            timing = hist.get('timing', {})
                            total_time = timing.get('total', 0)
                            print(f"   {i+1}. Frame {hist['frame']}: '{hist['text']}' -> {hist['emotion']} ({hist['top_prob']:.2f}) | {hist['second_emotion']} ({hist['second_prob']:.2f}) ({hist['frames']} frames, {total_time:.0f}ms)")
                        print("=" * 60)

            y_pos = 30
            
            y_pos += draw_text_with_background(frame, f"Top: {last_emotion} ({last_top_prob:.2f})", (10, y_pos), 
                                             font_scale=0.8, text_color=(255, 255, 255), thickness=2)
            
            y_pos += draw_text_with_background(frame, f"2nd: {last_second_emotion} ({last_second_prob:.2f})", (10, y_pos), 
                                             font_scale=0.8, text_color=(255, 255, 255), thickness=2)
            
            asr_display = last_asr_text[:40] + "..." if len(last_asr_text) > 40 else last_asr_text
            y_pos += draw_text_with_background(frame, f"ASR: {asr_display}", (10, y_pos), 
                                             font_scale=0.6, text_color=(255, 255, 255), thickness=1)
            
            y_pos += draw_text_with_background(frame, f"Frame: {frame_count}", (10, y_pos), 
                                             font_scale=0.7, text_color=(255, 255, 255), thickness=2)
            y_pos += draw_text_with_background(frame, f"VAD: {vad_sm.last_speech_prob:.3f}", (10, y_pos), 
                                             font_scale=0.7, text_color=(255, 255, 255), thickness=2)
            
            if last_timing_info:
                y_pos += draw_text_with_background(frame, f"ASR: {last_timing_info.get('asr', 0):.0f}ms", (10, y_pos), 
                                                 font_scale=0.5, text_color=(255, 255, 255), thickness=1)
                y_pos += draw_text_with_background(frame, f"Video: {last_timing_info.get('video', 0):.0f}ms", (10, y_pos), 
                                                 font_scale=0.5, text_color=(255, 255, 255), thickness=1)
                y_pos += draw_text_with_background(frame, f"Audio: {last_timing_info.get('audio', 0):.0f}ms", (10, y_pos), 
                                                 font_scale=0.5, text_color=(255, 255, 255), thickness=1)
                y_pos += draw_text_with_background(frame, f"Fusion: {last_timing_info.get('fusion', 0):.0f}ms", (10, y_pos), 
                                                 font_scale=0.5, text_color=(255, 255, 255), thickness=1)
                y_pos += draw_text_with_background(frame, f"Total: {last_timing_info.get('total', 0):.0f}ms", (10, y_pos), 
                                                 font_scale=0.5, text_color=(255, 255, 255), thickness=1)
            
            cv2.imshow('Live Emotion Inference', frame)
            
            if frame_count % 200 == 0:
                fps = frame_count / (time.time() - fps_start_time)
                print(f"Frame {frame_count}: FPS={fps:.1f} (target: {VIDEO_FPS}), VAD={vad_sm.last_speech_prob:.3f}, Emotion={last_emotion}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        audio_stream.stop()
        print("Live inference stopped")

if __name__ == "__main__":
    main()
