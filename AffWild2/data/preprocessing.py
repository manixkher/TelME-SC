import csv
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def split(session, max_context=12):
    final_data = []
    
    for i in range(len(session)):
        start_idx = max(0, i - max_context + 1)
        context_window = session[start_idx:i + 1]
        final_data.append(context_window)
    
    return final_data

def preprocessing(data_path, max_context=12):
    """
    Preprocess the AffWild2 dataset.
    Args:
        data_path (str): Path to the AffWild2 dataset.
        max_context (int, optional): The maximum context window size. Defaults to 12.

    Returns:
        list: Preprocessed AffWild2 dataset.
    """
    f = open(data_path, 'r')
    rdr = csv.reader(f)
    session_dataset = []
    session = []
    speaker_set = []
    pre_sess = 'start'
    has_timestamps = False
    
    for i, line in enumerate(rdr):
        if i == 0:
            # First row grab fields
            header  = line
            utt_idx = header.index('Utterance')
            speaker_idx = header.index('Speaker')
            emo_idx = header.index('Emotion')
            sess_idx = header.index('Dialogue_ID')
            video_idx = header.index('Video_Path')
            has_timestamps = 'Start_Time' in header and 'End_Time' in header and 'Duration' in header
            if has_timestamps:
                start_time_idx = header.index('Start_Time')
                end_time_idx = header.index('End_Time')
                duration_idx = header.index('Duration')
        else:
            # Other rows
            utt = line[utt_idx]
            speaker = line[speaker_idx]
            sess = line[sess_idx]
            video_path = line[video_idx]

            if pre_sess != 'start' and sess != pre_sess:
                session_dataset += split(session)
                session = []
                speaker_set = []
            
            if speaker in speaker_set:
                uniq_speaker = speaker_set.index(speaker)
            else:
                speaker_set.append(speaker)
                uniq_speaker = speaker_set.index(speaker)
            
            emotion = line[emo_idx].lower()
            # If timestamps are available, add them to the session
            if has_timestamps and len(line) > max(start_time_idx, end_time_idx, duration_idx):
                start_time = float(line[start_time_idx])
                end_time = float(line[end_time_idx])
                duration = float(line[duration_idx])
                session.append([uniq_speaker, utt, video_path, emotion, start_time, end_time, duration])
            else:
                session.append([uniq_speaker, utt, video_path, emotion])
            
            pre_sess = sess   
    
    session_dataset += split(session, max_context)           
    f.close()
    return session_dataset

def get_affwild2_emotion_labels():

    return ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
