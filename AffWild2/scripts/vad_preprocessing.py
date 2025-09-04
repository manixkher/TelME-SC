
import os
import sys
import time
import logging
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vad_preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

EXPR_CLASSES = [
    'Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise'
]

def process_video_worker(args):
    """
    Process video worker.

    Args:
        args (tuple): Arguments

    Returns:
        list: Utterances
    """
    video_path, videos_dir, expr_dir, vad_model, vad_threshold = args
    
    try:
        temp_preprocessor = VADPreprocessor(
            videos_dir=videos_dir,
            expr_dir=expr_dir,
            output_dir="./temp_output",
            vad_model=vad_model,
            vad_threshold=vad_threshold
        )
        
        utterances = temp_preprocessor.process_single_video(Path(video_path))
        return utterances
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

class VADPreprocessor:
    """
    VAD Preprocessor.
    """
    def __init__(self, 
                 videos_dir: str,
                 expr_dir: str,
                 output_dir: str,
                 vad_model: str = 'sre',
                 vad_threshold: Tuple[float, float] = (0.5, 0.1)):
        """
        Initialize VAD Preprocessor.

        Args:
            videos_dir (str): Videos directory
            expr_dir (str): Expression directory
            output_dir (str): Output directory
            vad_model (str, optional): VAD model. Defaults to 'sre'.
            vad_threshold (Tuple[float, float], optional): VAD threshold. Defaults to (0.5, 0.1).
        """
        self.videos_dir = Path(videos_dir)
        self.expr_dir = Path(expr_dir)
        self.output_dir = Path(output_dir)
        self.vad_model = vad_model
        self.vad_threshold = vad_threshold
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def find_video_files(self) -> List[Path]:
        """
        Find video files.

        Returns:
            List[Path]: Video files
        """
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(self.videos_dir.rglob(f'*{ext}'))
        
        logger.info(f"Found {len(video_files)} video files")
        return sorted(video_files)
    
    def extract_audio_for_vad(self, video_path: Path, sample_rate: int = 22050) -> Optional[Path]:
        """
        Extract audio for VAD.

        Args:
            video_path (Path): Video path
            sample_rate (int, optional): Sample rate. Defaults to 22050.
        """
        try:
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            temp_audio_path = Path(temp_audio.name)
            
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', str(sample_rate),
                '-ac', '1',
                '-f', 'wav',
                '-y',
                str(temp_audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and temp_audio_path.exists():
                return temp_audio_path
            else:
                logger.error(f"Failed to extract audio from {video_path}: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting audio from {video_path}: {e}")
            return None
    
    def run_vad_analysis(self, audio_path: Path, model: str = 'sre', 
                        threshold: Tuple[float, float] = (0.5, 0.1)) -> Optional[pd.DataFrame]:
        """
        Run VAD analysis.

        Args:
            audio_path (Path): Audio path
            model (str, optional): VAD model. Defaults to 'sre'.
            threshold (Tuple[float, float], optional): VAD threshold. Defaults to (0.5, 0.1).

        Args:
            audio_path (Path): Audio path
            model (str, optional): VAD model. Defaults to 'sre'.
            threshold (Tuple[float, float], optional): VAD threshold. Defaults to (0.5, 0.1).

        Returns:
            Optional[pd.DataFrame]: VAD results
        """
        try:
            vad_dir = Path("/home/s2751435/Work/msc/Datadriven-VAD")
            if not vad_dir.exists():
                logger.error(f"Datadriven-VAD directory not found: {vad_dir}")
                return None
            
            forward_script = vad_dir / "forward.py"
            
            cmd = [
                sys.executable, str(forward_script),
                "-w", str(audio_path.resolve()),
                "-model", model,
                "-th", str(threshold[0]), str(threshold[1])
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=vad_dir)
            
            if result.returncode == 0:
                vad_results = self.parse_vad_output(result.stdout)
                return vad_results
            else:
                logger.error(f"VAD analysis failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error running VAD: {e}")
            return None

    def parse_vad_output(self, stdout: str) -> pd.DataFrame:
        """
        Parse VAD output.

        Args:
            stdout (str): Standard output

        Returns:
            pd.DataFrame: VAD results
        """
        try:
            lines = stdout.strip().split('\n')
            table_start = None
            table_end = None
            
            for i, line in enumerate(lines):
                if '|' in line and 'event_label' in line:
                    table_start = i
                elif table_start and line.strip() == '':
                    table_end = i
                    break
            
            if table_start is None:
                return pd.DataFrame()
            
            table_lines = lines[table_start:table_end] if table_end else lines[table_start:]
            
            data = []
            for line in table_lines[2:]:
                if '|' in line:
                    parts = [part.strip() for part in line.split('|')[1:-1]]
                    if len(parts) >= 4:
                        data.append({
                            'index': int(parts[0]),
                            'event_label': parts[1],
                            'onset': float(parts[2]),
                            'offset': float(parts[3]),
                            'filename': parts[4] if len(parts) > 4 else ''
                        })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error parsing VAD output: {e}")
            return pd.DataFrame()
    
    def load_expr_labels(self, expr_file: Path) -> np.ndarray:
        """
        Load EXPR labels.

        Args:
            expr_file (Path): EXPR file

        Returns:
            np.ndarray: EXPR labels
        """
        try:
            df = pd.read_csv(expr_file, sep=' ', header=None, engine='python')
            df = df.apply(pd.to_numeric, errors='coerce')
            
            if df.shape[1] == 1:
                arr = np.zeros((len(df), len(EXPR_CLASSES)), dtype=np.float32)
                for i, v in enumerate(df.iloc[:, 0].values):
                    if pd.notna(v) and v >= 0 and v < len(EXPR_CLASSES):
                        arr[i, int(v)] = 1.0
                return arr
            else:
                arr = df.values.astype(np.float32)
                arr[np.isnan(arr)] = 0
                arr[arr < 0] = 0
                return arr
                
        except Exception as e:
            logger.error(f"Error loading EXPR labels from {expr_file}: {e}")
            return np.zeros((0, len(EXPR_CLASSES)), dtype=np.float32)
    
    def find_expr_file(self, video_base: str) -> Optional[Path]:
        """
        Find EXPR file.

        Args:
            video_base (str): Video base

        Returns:
            Optional[Path]: EXPR file
        """
        base_dir = self.expr_dir.parent
        
        train_file = base_dir / "Train_Set" / f"{video_base}.txt"
        if train_file.exists():
            return train_file
        
        validation_file = base_dir / "Validation_Set" / f"{video_base}.txt"
        if validation_file.exists():
            return validation_file
        
        train_dir = base_dir / "Train_Set"
        if train_dir.exists():
            for pattern in [f"{video_base}-*.txt", f"{video_base}_*.txt"]:
                matches = list(train_dir.glob(pattern))
                if matches:
                    return matches[0]
        
        validation_dir = base_dir / "Validation_Set"
        if validation_dir.exists():
            for pattern in [f"{video_base}-*.txt", f"{video_base}_*.txt"]:
                matches = list(validation_dir.glob(pattern))
                if matches:
                    return matches[0]
        
        return None

    def get_emotion_for_utterance(self, expr_file: Path, onset: float, offset: float) -> str:
        """
        Get emotion for utterance.

        Args:
            expr_file (Path): EXPR file
            onset (float): Onset
            offset (float): Offset

        Returns:
            str: Emotion
        """
        if not expr_file.exists():
            return 'Unknown'
        
        try:
            expr_arr = self.load_expr_labels(expr_file)
            if expr_arr.size == 0:
                return 'Unknown'
            
            start_frame = int(np.floor(onset * 30))
            end_frame = int(np.ceil(offset * 30))
            
            if start_frame >= len(expr_arr):
                return 'Unknown'
            
            end_frame = min(end_frame, len(expr_arr))
            window = expr_arr[start_frame:end_frame]
            
            if window.size > 0:
                mean_emotions = window.mean(axis=0)
                emotion_idx = int(np.argmax(mean_emotions))
                return EXPR_CLASSES[emotion_idx]
            else:
                return 'Unknown'
                
        except Exception as e:
            logger.error(f"Error getting emotion for utterance: {e}")
            return 'Unknown'
    
    def process_single_video(self, video_path: Path) -> List[Dict]:
        """
        Process single video.

        Args:
            video_path (Path): Video path

        Returns:
            List[Dict]: Utterances
        """
        logger.info(f"Processing video: {video_path.name}")
        start_time = time.time()
        
        utterances = []
        
        try:
            audio_path = self.extract_audio_for_vad(video_path)
            if not audio_path:
                logger.error(f"Failed to extract audio from {video_path}")
                return utterances
            
            vad_results = self.run_vad_analysis(audio_path, self.vad_model, self.vad_threshold)
            
            if vad_results is None or vad_results.empty:
                logger.warning(f"No speech segments found in {video_path}")
                return utterances
            
            speech_segments = vad_results[vad_results['event_label'] == 'Speech']
            logger.info(f"Found {len(speech_segments)} speech segments")
            
            video_base = video_path.stem
            expr_file = self.find_expr_file(video_base)
            
            for idx, row in speech_segments.iterrows():
                onset = row['onset']
                offset = row['offset']
                
                emotion = 'Unknown'
                if expr_file:
                    emotion = self.get_emotion_for_utterance(expr_file, onset, offset)
                
                utterance = {
                    'video_path': str(video_path),
                    'video_name': video_path.name,
                    'utterance_id': row['index'],
                    'start_time': onset,
                    'end_time': offset,
                    'duration': offset - onset,
                    'emotion': emotion,
                    'speech_segment': True
                }
                
                utterances.append(utterance)
            
            try:
                os.remove(audio_path)
            except:
                pass
            
            total_time = time.time() - start_time
            logger.info(f"✓ Processed {video_path.name} in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
        
        return utterances
    
    def process_dataset(self, num_workers: int = 4):
        """
        Process dataset.

        Args:
            num_workers (int, optional): Number of workers. Defaults to 4.
        """
        start_time = time.time()
        logger.info(f"Starting AffWild2 VAD preprocessing with {num_workers} workers...")
        
        video_files = self.find_video_files()
        if not video_files:
            logger.error("No video files found!")
            return
        
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        logger.info(f"Processing {len(video_files)} videos with {num_workers} parallel workers...")
        
        output_csv = Path("/home/s2751435/Work/msc/TelME/AffWild2/vad_output/affwild2_utterances.csv")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_headers = [
            'utterance_id', 'video_name', 'start_time', 'end_time', 'duration',
            'emotion', 'video_path'
        ]
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            f.write(','.join(csv_headers) + '\n')
        
        total_utterances = 0
        processed_videos = 0
        emotion_counts = {}
        durations = []
        
        worker_args = []
        for video_path in video_files:
            args = (
                str(video_path),
                str(self.videos_dir),
                str(self.expr_dir),
                self.vad_model,
                self.vad_threshold
            )
            worker_args.append(args)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_video = {
                executor.submit(process_video_worker, args): args[0] 
                for args in worker_args
            }
            
            with tqdm(total=len(video_files), desc="Processing videos") as pbar:
                for future in as_completed(future_to_video):
                    video_path = future_to_video[future]
                    
                    try:
                        utterances = future.result()
                        
                        if utterances:
                            with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                                for utterance in utterances:
                                    row = [
                                        str(utterance['utterance_id']),
                                        f'"{utterance["video_name"]}"',
                                        str(utterance['start_time']),
                                        str(utterance['end_time']),
                                        str(utterance['duration']),
                                        f'"{utterance["emotion"]}"',
                                        f'"{utterance["video_path"]}"'
                                    ]
                                    f.write(','.join(row) + '\n')
                            
                            total_utterances += len(utterances)
                            
                            for utterance in utterances:
                                emotion = utterance['emotion']
                                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                                durations.append(utterance['duration'])
                        
                        processed_videos += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing {video_path}: {e}")
                        processed_videos += 1
                    
                    pbar.update(1)
                    
                    if processed_videos % 50 == 0:
                        logger.info(f"Progress: {processed_videos}/{len(video_files)} videos, "
                                  f"{total_utterances} utterances processed")
        
        if total_utterances > 0:
            logger.info(f"✓ Processing completed!")
            logger.info(f"  Total videos processed: {processed_videos}/{len(video_files)}")
            logger.info(f"  Total utterances: {total_utterances}")
            logger.info(f"  Output saved to: {output_csv}")
            
            logger.info("Emotion distribution:")
            for emotion, count in sorted(emotion_counts.items()):
                logger.info(f"  {emotion}: {count}")
            
            if durations:
                durations = np.array(durations)
                logger.info(f"Duration statistics:")
                logger.info(f"  Mean: {durations.mean():.2f}s")
                logger.info(f"  Median: {np.median(durations):.2f}s")
                logger.info(f"  Min: {durations.min():.2f}s")
                logger.info(f"  Max: {durations.max():.2f}s")
            
            total_time = time.time() - start_time
            videos_per_second = processed_videos / total_time
            logger.info(f"Performance statistics:")
            logger.info(f"  Total processing time: {total_time:.2f}s")
            logger.info(f"  Videos per second: {videos_per_second:.2f}")
            logger.info(f"  Average time per video: {total_time/processed_videos:.2f}s")
            
        else:
            logger.error("No utterances found!")

def main():
    parser = argparse.ArgumentParser(description="AffWild2 VAD Preprocessing")
    parser.add_argument(
        "--videos-dir",
        type=str,
        required=True,
        help="Directory containing AffWild2 video files"
    )
    parser.add_argument(
        "--expr-dir", 
        type=str,
        required=True,
        help="Directory containing AffWild2 EXPR annotation files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./vad_output",
        help="Output directory for results (default: ./vad_output)"
    )
    parser.add_argument(
        "--scratch-dir",
        type=str,
        help="Scratch directory for temporary files (optional)"
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        nargs=2,
        default=[0.5, 0.1],
        help="VAD thresholds: high_threshold low_threshold (default: 0.5 0.1)"
    )
    parser.add_argument(
        "--vad-model",
        type=str,
        default="sre",
        choices=["sre", "t1", "t2", "v2", "a2", "a2_v2", "c1"],
        help="VAD model to use (default: sre)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for processing (default: 4)"
    )
    
    args = parser.parse_args()
    
    if args.scratch_dir:
        scratch_videos_dir = Path(args.scratch_dir) / "videos"
        scratch_expr_dir = Path(args.scratch_dir) / "expr"
        scratch_output_dir = Path(args.scratch_dir) / "output"
        
        scratch_videos_dir.mkdir(parents=True, exist_ok=True)
        scratch_expr_dir.mkdir(parents=True, exist_ok=True)
        scratch_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Using scratch directory: {args.scratch_dir}")
        
        if not list(scratch_videos_dir.glob("*.mp4")):
            print("Copying videos to scratch...")
            subprocess.run(["cp", "-r", f"{args.videos_dir}/*", str(scratch_videos_dir)])
        
        if not list(scratch_expr_dir.glob("*.txt")):
            print("Copying EXPR annotations to scratch...")
            subprocess.run(["cp", "-r", f"{args.expr_dir}/*", str(scratch_expr_dir)])
        
        videos_dir = str(scratch_videos_dir)
        expr_dir = str(scratch_expr_dir)
        output_dir = str(scratch_output_dir)
    else:
        videos_dir = args.videos_dir
        expr_dir = args.expr_dir
        output_dir = args.output_dir
    
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg is not installed or not available in PATH")
        return
    
    preprocessor = VADPreprocessor(
        videos_dir=videos_dir,
        expr_dir=expr_dir,
        output_dir=output_dir,
        vad_model=args.vad_model,
        vad_threshold=tuple(args.vad_threshold)
    )
    
    preprocessor.process_dataset(num_workers=args.num_workers)

if __name__ == "__main__":
    main() 