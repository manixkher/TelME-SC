import os
import torch
from transformers import RobertaTokenizer, RobertaModel, AutoProcessor, AutoImageProcessor
import warnings
import librosa
import cv2
import numpy as np
import decord
from decord import VideoReader


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
warnings.filterwarnings("ignore", message=".*__audioread_load.*")

audio_processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
speaker_list = ['<s1>', '<s2>', '<s3>', '<s4>', '<s5>', '<s6>', '<s7>', '<s8>', '<s9>']
speaker_tokens_dict = {'additional_special_tokens': speaker_list}
roberta_tokenizer.add_special_tokens(speaker_tokens_dict)

def fix_video_path(video_path):
    # Replace old machine paths with new machine paths
    old_path_pattern = "/home/s2751435/Work/msc/TelME/dataset/AffWild2/"
    new_path_pattern = "/work/tc067/tc067/s2751435/Work/msc/TelME/dataset/AffWild2/"
    
    # Also handle old scratch paths that might still be in CSV files
    old_scratch_pattern = "/disk/scratch/s2751435/TelME_dataset/AffWild2_VAD/"
    new_scratch_pattern = "/work/tc067/tc067/s2751435/Work/msc/TelME/dataset/AffWild2/"
    
    if old_path_pattern in video_path:
        corrected_path = video_path.replace(old_path_pattern, new_path_pattern)
        return corrected_path
    
    if old_scratch_pattern in video_path:
        corrected_path = video_path.replace(old_scratch_pattern, new_scratch_pattern)
        return corrected_path
    
    return video_path

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    
    return ids + [tokenizer.mask_token_id]

def padding(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    attention_masks = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        attention_mask = [ 1 for _ in range(len(ids))]
        add_attention = [ 0 for _ in range(len(add_ids))]
        pad_ids.append(add_ids+ids)
        attention_masks.append(add_attention+attention_mask)
    return torch.tensor(pad_ids), torch.tensor(attention_masks)

def padding_video(batch):
    max_len = 0
    for ids in batch:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    for ids in batch:
        pad_len = max_len-len(ids)
        add_ids = [ 0 for _ in range(pad_len)]
        
        pad_ids.append(add_ids+ids.tolist())
    
    return torch.tensor(pad_ids)

def padding_audio_with_mask(batch):
    # Find maximum length in batch
    max_len = max(audio.size(0) for audio in batch)
    
    padded = []
    masks = []
    
    for wav in batch:
        pad_len = max_len - wav.size(0)
        # Right-pad with zeros
        padded.append(F.pad(wav, (0, pad_len)))
        # Create mask: 1 for real samples, 0 for padded
        mask = torch.cat([
            torch.ones(wav.size(0), dtype=torch.long),  # 1 = keep
            torch.zeros(pad_len, dtype=torch.long)      # 0 = ignore
        ])
        masks.append(mask)
    
    batch_audio = torch.stack(padded)      # (B, T_max)
    batch_mask = torch.stack(masks)        # (B, T_max)
    
    return batch_audio, batch_mask

def get_audio(processor, path):
    audio, rate = librosa.load(path, sr=16000)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    return inputs["input_values"][0]

def get_audio_segment(processor, path, start_time=None, end_time=None):
    try:
        audio, rate = librosa.load(path, sr=16000)
        
        # Extract time segment if timestamps provided
        if start_time is not None and end_time is not None:
            start_sample = int(start_time * rate)
            end_sample = int(end_time * rate)
            end_sample = min(end_sample, len(audio))
            start_sample = min(start_sample, end_sample - 1)
            audio = audio[start_sample:end_sample]
        
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        return inputs["input_values"][0]
    except Exception as e:
        print(f"Error loading audio segment from {path}: {str(e)}")
        return torch.zeros([1412])

def get_video_segment(feature_extractor, path, start_time=None, end_time=None):
    return get_video_segment_opencv(feature_extractor, path, start_time, end_time)

def get_video_segment_opencv(feature_extractor, path, start_time=None, end_time=None):
    try:
        video = cv2.VideoCapture(path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        # Calculate frame range if timestamps provided
        if start_time is not None and end_time is not None:
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))
            frame_range = list(range(start_frame, end_frame))
        else:
            frame_range = list(range(total_frames))
        
        # Extract frames
        if len(frame_range) >= 8:
            # Sample 8 frames evenly
            step = len(frame_range) // 8
            for i in range(8):
                frame_idx = frame_range[i * step]
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, image = video.read()
                if ret:
                    frames.append(image)
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            # Use all frames and pad if needed
            for frame_idx in frame_range:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, image = video.read()
                if ret:
                    frames.append(image)
            
            # Pad with last frame if needed
            lack = 8 - len(frames)
            if lack > 0 and frames:
                extend_frames = [frames[-1].copy() for _ in range(lack)]
                frames.extend(extend_frames)
            elif lack > 0:
                # No frames available, use zeros
                frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(8)]
        
        video.release()
        
        if not frames:
            return torch.zeros([8, 3, 224, 224])
            
        inputs = feature_extractor(frames[:8], return_tensors="pt")
        return inputs["pixel_values"][0]
    except Exception as e:
        print(f"OpenCV also failed for {path}: {str(e)}")
        return torch.zeros([8, 3, 224, 224])
    
def get_video(feature_extractor, path):
    video = cv2.VideoCapture(path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    step = length // 8 if length >= 8 else 1
    count = 0
    if length >= 8:
        while(video.isOpened()):
            ret, image = video.read()
            if(ret==False):
                break
            count += 1
            if count % step == 0:
                frames.append(image)
        video.release()
    else:
        while(video.isOpened()):
            ret, image = video.read()
            if(ret==False):
                break
            frames.append(image)
        video.release()
        lack = 8 - len(frames)
        extend_frames = [ frames[-1].copy() for _ in range(lack)]
        frames.extend(extend_frames)
    inputs = feature_extractor(frames[:8], return_tensors="pt")
    return inputs["pixel_values"][0]

def make_batchs(sessions):
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_input, batch_audio, batch_video, batch_labels = [], [], [], []
    max_length = 400000

    for session in sessions:
        try:
            inputString = ""
            now_speaker = None
            for turn, line in enumerate(session):
                speaker, utt, video_path, emotion = line

                inputString += f'<s{speaker + 1}> {utt} '
                now_speaker = speaker


            video_path = fix_video_path(video_path)


            if not os.path.exists(video_path):
                print(f"Missing file: {video_path}")
                continue

            audio, rate = librosa.load(video_path, sr=16000)
            duration = librosa.get_duration(y=audio, sr=rate)

            if duration > 30:
                batch_video.append(torch.zeros([8, 3, 224, 224]))
                batch_audio.append(torch.zeros([1412]))
            else:
                audio_input = get_audio(audio_processor, video_path)
                audio_input = audio_input[-max_length:]
                batch_audio.append(audio_input)

                video_input = get_video(video_processor, video_path)
                batch_video.append(video_input)

            prompt = f"Now <s{now_speaker + 1}> feels"
            concat_string = inputString.strip() + " </s> " + prompt
            batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

            label_ind = label_list.index(emotion)
            batch_labels.append(label_ind)

        except Exception as e:
            print(f"Skipping file {video_path} due to error:\n{e}\n")
            continue

    if not batch_input:
        raise RuntimeError("All sessions in this batch failed. Nothing to batch.")

    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
    batch_audio = padding_video(batch_audio)
    batch_video = torch.stack(batch_video)
    batch_labels = torch.tensor(batch_labels)

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels


def make_batchs_affwild(sessions):
    """
    Make batchs for AffWild2 dataset.

    Args:
        sessions (list): Sessions

    Raises:
        RuntimeError: Runtime error

    Returns:
        tuple: Batch input tokens, batch attention masks, batch audio, batch video, batch labels
    """
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_input, batch_audio, batch_video, batch_labels = [], [], [], []
    max_length = 400000

    for session in sessions:
        try:
            inputString = ""
            now_speaker = None
            
            current_utterance = session[-1]  
            if len(current_utterance) >= 7:
                current_speaker, current_utt, current_video_path, current_emotion, start_time, end_time, duration = current_utterance
            else:
                current_speaker, current_utt, current_video_path, current_emotion = current_utterance
            
            for turn, line in enumerate(session):
                speaker, utt = line[0], line[1]
                inputString += f'<s{speaker + 1}> {utt} '
                now_speaker = speaker

            current_video_path = fix_video_path(current_video_path)
            
            if not os.path.exists(current_video_path):
                print(f"Missing file: {current_video_path}")
                continue

            # Try loading audio to check duration
            try:
                audio, rate = librosa.load(current_video_path, sr=16000)
                duration = librosa.get_duration(y=audio, sr=rate)
            except Exception as audio_error:
                print(f"Failed to load audio from {current_video_path}: {audio_error}")
                continue


            MIN_LEN = 4800  # 0.3 s * 16 kHz
            if len(audio) < MIN_LEN:
                print(f"Skipping very short audio: {current_video_path} (duration: {duration:.3f}s)")
                continue

            if duration > 30:
                batch_video.append(torch.zeros([8, 3, 224, 224]))
                batch_audio.append(torch.zeros([1412]))
            else:
                audio_input = get_audio(audio_processor, current_video_path)
                audio_input = audio_input[-max_length:]
                batch_audio.append(audio_input)

                video_input = get_video(video_processor, current_video_path)
                batch_video.append(video_input)

            prompt = f"Now <s{now_speaker + 1}> feels"
            concat_string = inputString.strip() + " </s> " + prompt
            batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

            label_ind = label_list.index(current_emotion)
            batch_labels.append(label_ind)

        except Exception as e:
            try:
                error_video_path = current_video_path if 'current_video_path' in locals() else "unknown"
            except:
                error_video_path = "unknown"
            print(f"Skipping file {error_video_path} due to error:\n{e}\n")
            continue

    if not batch_input:
        raise RuntimeError("All sessions in this batch failed. Nothing to batch.")

    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
    batch_audio = padding_video(batch_audio)
    batch_video = torch.stack(batch_video)
    batch_labels = torch.tensor(batch_labels)

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels
