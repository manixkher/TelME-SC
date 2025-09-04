

import os
import csv
import json
import torch
import whisper
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    
    def __init__(self, model_name: str = "base", device: str = "cuda" if torch.cuda.is_available() else "cpu", videos_dir: str = None):
        """
        Initialize Whisper Transcriber.

        Args:
            model_name (str, optional): Whisper model name. Defaults to "base".
            device (str, optional): Device to use for inference. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
            videos_dir (str, optional): Videos directory. Defaults to None.
        """
        self.device = device
        self.videos_dir = videos_dir
        logger.info(f"Loading Whisper model: {model_name} on {device}")
        
        # Load Whisper model
        self.model = whisper.load_model(model_name, device=device)
        logger.info(f"Whisper model loaded successfully")
        
    def get_video_path(self, video_name: str) -> Optional[str]:
        """
        Get video path.

        Args:
            video_name (str): Video name

        Returns:
            Optional[str]: Video path
        """
        if not self.videos_dir:

            video_base_path = "/home/s2751435/Work/msc/TelME/dataset/AffWild2/videos"
        else:
            video_base_path = self.videos_dir
            
        for batch in ["batch1", "batch2"]:
            video_path = os.path.join(video_base_path, batch, video_name)
            if os.path.exists(video_path):
                return video_path
                
        video_name_avi = video_name.replace(".mp4", ".avi")
        for batch in ["batch1", "batch2"]:
            video_path = os.path.join(video_base_path, batch, video_name_avi)
            if os.path.exists(video_path):
                return video_path
                
        logger.warning(f"Video not found: {video_name}")
        return None
    
    def transcribe_video(self, video_path: str) -> Dict:
        """
        Transcribe video.

        Args:
            video_path (str): Video path

        Returns:
            Dict: Transcription
        """
        try:
            logger.info(f"Transcribing: {video_path}")
            
            result = self.model.transcribe(
                video_path,
                word_timestamps=True,
                verbose=False
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing {video_path}: {e}")
            return None
    
    def extract_text_for_timestamp(self, transcription: Dict, start_time: float, end_time: float) -> str:
        """
        Extract text for timestamp.

        Args:
            transcription (Dict): Transcription
            start_time (float): Start time
            end_time (float): End time

        Returns:
            str: Text
        """
        if not transcription or 'segments' not in transcription:
            return ""
        
        words_in_range = []
        
        for segment in transcription['segments']:
            segment_start = segment['start']
            segment_end = segment['end']
            

            if segment_end >= start_time and segment_start <= end_time:
                # Extract words that fall within our range
                for word_info in segment.get('words', []):
                    word_start = word_info['start']
                    word_end = word_info['end']
                    
                    # Check if word overlaps with our time range
                    if word_end >= start_time and word_start <= end_time:
                        words_in_range.append(word_info['word'])
        

        text = " ".join(words_in_range).strip()
        return text
    
    def process_csv_with_whisper(self, input_csv_path: str, output_csv_path: str):
        """
        Process CSV with Whisper.

        Args:
            input_csv_path (str): Input CSV path
            output_csv_path (str): Output CSV path
        """
        
        logger.info(f"Processing CSV: {input_csv_path}")
        
        df = pd.read_csv(input_csv_path)
        logger.info(f"Loaded {len(df)} utterances from CSV")
        
        video_groups = df.groupby('Dialogue_ID')
        logger.info(f"Found {len(video_groups)} unique videos")
        
        video_transcriptions = {}
        
        for video_name, group in tqdm(video_groups, desc="Processing videos"):
            video_path = self.get_video_path(video_name)
            
            if not video_path:
                logger.warning(f"Skipping video {video_name} - not found")
                continue
            
            if video_name not in video_transcriptions:
                transcription = self.transcribe_video(video_path)
                if transcription:
                    video_transcriptions[video_name] = transcription
                else:
                    logger.error(f"Failed to transcribe {video_name}")
                    continue
        
        transcripts = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting transcripts"):
            video_name = row['Dialogue_ID']
            start_time = row['Start_Time']
            end_time = row['End_Time']
            
            if video_name in video_transcriptions:
                transcript = self.extract_text_for_timestamp(
                    video_transcriptions[video_name], 
                    start_time, 
                    end_time
                )
            else:
                transcript = ""
            
            new_row = {
                'Utterance': transcript,
                'Speaker': row.get('Speaker', 0),
                'Dialogue_ID': video_name,
                'Emotion': row.get('Emotion', 'Unknown'),
                'Video_Path': row.get('Video_Path', ''),
                'Start_Time': start_time,
                'End_Time': end_time,
                'Duration': end_time - start_time
            }
            
            transcripts.append(new_row)
        
        output_df = pd.DataFrame(transcripts)
        output_df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved {len(output_df)} utterances to {output_csv_path}")
        
        non_empty_transcripts = output_df[output_df['Utterance'].str.strip() != '']
        logger.info(f"Non-empty transcripts: {len(non_empty_transcripts)}/{len(output_df)}")
        
        logger.info("Sample transcripts:")
        for i, row in output_df.head(10).iterrows():
            logger.info(f"  {row['Start_Time']:.2f}-{row['End_Time']:.2f}s: '{row['Utterance']}'")

def parse_arguments():
    """
    Parse arguments.

    Returns:
        argparse.Namespace: Arguments
    """
    parser = argparse.ArgumentParser(description='Whisper transcription for AffWild2 dataset')
    parser.add_argument('--input-csv', type=str, 
                       default="/home/s2751435/Work/msc/TelME/AffWild2/vad_output/affwild2_utterances_original_backup.csv",
                       help='Path to input CSV with VAD timestamps')
    parser.add_argument('--output-csv', type=str,
                       default="/home/s2751435/Work/msc/TelME/AffWild2/data/affwild2_train_emotions_whisper.csv",
                       help='Path to output CSV with transcripts')
    parser.add_argument('--videos-dir', type=str,
                       help='Directory containing video files (batch1 and batch2 subdirectories)')
    parser.add_argument('--model-size', type=str, default='large',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference')
    
    return parser.parse_args()

def main():
    
    args = parse_arguments()
    
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    transcriber = WhisperTranscriber(
        model_name=args.model_size,
        device=device,
        videos_dir=args.videos_dir
    )
    
    if not os.path.exists(args.input_csv):
        logger.error(f"Input CSV not found: {args.input_csv}")
        return
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    transcriber.process_csv_with_whisper(args.input_csv, args.output_csv)
    
    logger.info("Transcription process completed!")

if __name__ == "__main__":
    main() 