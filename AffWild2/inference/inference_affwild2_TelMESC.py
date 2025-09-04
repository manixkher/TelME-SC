import os
import numpy as np
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_recall_fscore_support, classification_report

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import gc

from transformers import RobertaTokenizer

from AffWild2.data.preprocessing import preprocessing
from AffWild2.data.dataset import affwild2_dataset
from AffWild2.utils.utils import make_batchs_affwild, seed_everything
from AffWild2.models.teacher import Teacher_model
from AffWild2.models.student import Student_Audio
from AffWild2.models.student_video import Student_Video
from MELD.models_new.TelMESC import ASF_SCMMTelMESC


def parse_args():
    """
    Parse arguments.

    Returns:
        argparse.Namespace: Arguments
    """
    parser = argparse.ArgumentParser(description='AffWild2 TelMESC Inference with Analysis')
    parser.add_argument('--batch_size', default=4, type=int, help='batch for inference.')
    parser.add_argument('--seed', default=42, type=int, help='random seed fix')
    parser.add_argument('--model_path', default='./AffWild2/save_model/telmesc_affwild2_epoch_4_concurrent.bin', type=str, help='path to TelMESC model')
    parser.add_argument('--teacher_path', default='./AffWild2/save_model/exp_new_teacher_affwild2_whisper_epoch_4.bin', type=str, help='path to teacher model')
    parser.add_argument('--audio_student_path', default='./AffWild2/save_model/student_audio/exp_student_audio_best_affwild2.bin', type=str, help='path to audio student model')
    parser.add_argument('--video_student_path', default='./AffWild2/save_model/student_video/exp_student_visual_best_affwild2.bin', type=str, help='path to video student model')
    parser.add_argument('--output_dir', default='./AffWild2/output/TelMESC_analysis', type=str, help='output directory for analysis')
    parser.add_argument('--gate_mode', default='learned', choices=['learned', 'uniform'], help='gate mode: learned or uniform routing')
    parser.add_argument('--scmm_max_context', default=12, type=int, help='Maximum context size for TelMESC')
    parser.add_argument('--scmm_dropout', default=0.1, type=float, help='TelMESC dropout rate')
    parser.add_argument('--scmm_path_dropout', default=0.0, type=float, help='Path dropout probability for TelMESC')
    parser.add_argument('--test_split', default='test', type=str, choices=['dev', 'test'], help='Which split to evaluate on')
    parser.add_argument('--save_predictions_csv', action='store_true', help='Save predictions to CSV with video paths')
    parser.add_argument('--handle_neutral_bias', action='store_true', help='Handle neutral bias by predicting 2nd highest when neutral confidence >0.62')
    parser.add_argument('--num_runs', default=1, type=int, help='Number of runs with different random seeds to average results')
    args = parser.parse_args()
    return args

def seed_everything(seed):
    """
    Seed everything.

    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def inference_with_analysis(model_t, audio_s, video_s, fusion, dataloader, output_dir, gate_mode="learned", save_csv=False, test_split="test", handle_neutral_bias=False):
    """
    Inference with analysis.

    Args:
        model_t (torch.nn.Module): Teacher model
        audio_s (torch.nn.Module): Audio student model
        video_s (torch.nn.Module): Video student model
        fusion (torch.nn.Module): Fusion model
        dataloader (torch.utils.data.DataLoader): Data loader
        output_dir (str): Output directory
        gate_mode (str): Gate mode
        save_csv (bool): Save predictions to CSV
        test_split (str): Test split
        handle_neutral_bias (bool): Handle neutral bias
    """
    fusion.eval()
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    pred_list = []
    label_list_gt = []
    video_paths_list = []
    sample_indices = []
    confidence_list = []
    confidence_list_biased = []
    
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader, desc="Running TelMESC inference")):
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), batch_labels.cuda()
            text_hidden, _ = model_t(batch_input_tokens, attention_masks)
            audio_hidden, _ = audio_s(audio_inputs)
            video_hidden, _ = video_s(video_inputs)
            pred_logits, _ = fusion(
                text_hidden, audio_hidden, video_hidden,
                return_scmm_info=False,
                gate_mode=gate_mode
            )
            
            probs = F.softmax(pred_logits, dim=1)
            
            pred_label = pred_logits.argmax(1).detach().cpu().numpy()
            true_label = batch_labels.detach().cpu().numpy()
            confidence = probs.max(1)[0].detach().cpu().numpy()
            
            pred_list.extend(pred_label)
            label_list_gt.extend(true_label)
            confidence_list.extend(confidence)
                
            batch_size = len(pred_label)
            for i in range(batch_size):
                sample_indices.append(i_batch * batch_size + i)
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'predictions_{gate_mode}.npy'), np.array(pred_list))
    np.save(os.path.join(output_dir, f'labels_{gate_mode}.npy'), np.array(label_list_gt))
    
    if save_csv:
        data_path = './AffWild2/data/'
        if test_split == 'dev':
            test_path = data_path + 'affwild2_dev_emotions_whisper_filtered_complete_fixed.csv'
        else:  # test
            test_path = data_path + 'affwild2_test_emotions_whisper_filtered_complete_fixed.csv'
        
        df_original = pd.read_csv(test_path)
        
        results_df = pd.DataFrame({
            'sample_index': sample_indices,
            'video_path': df_original['Video_Path'].iloc[:len(pred_list)].values,
            'dialogue_id': df_original['Dialogue_ID'].iloc[:len(pred_list)].values,
            'utterance': df_original['Utterance'].iloc[:len(pred_list)].values,
            'start_time': df_original['Start_Time'].iloc[:len(pred_list)].values if 'Start_Time' in df_original.columns else [None] * len(pred_list),
            'end_time': df_original['End_Time'].iloc[:len(pred_list)].values if 'End_Time' in df_original.columns else [None] * len(pred_list),
            'ground_truth_emotion': [label_list[gt] for gt in label_list_gt],
            'predicted_emotion': [label_list[pred] for pred in pred_list],
            'ground_truth_label': label_list_gt,
            'predicted_label': pred_list,
            'confidence': confidence_list,
            'correct': [gt == pred for gt, pred in zip(label_list_gt, pred_list)]
        })
        
        csv_path = os.path.join(output_dir, f'predictions_video_mapping_{gate_mode}.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"Original predictions CSV saved to: {csv_path}")
        
      
        print(f"Total samples: {len(results_df)}")
        print(f"Correct predictions: {results_df['correct'].sum()}")
        print(f"Accuracy: {results_df['correct'].mean():.4f}")
        
        
        print("\nOriginal emotion-wise accuracy:")
        for emotion in label_list:
            emotion_mask = results_df['ground_truth_emotion'] == emotion
            if emotion_mask.sum() > 0:
                emotion_acc = results_df[emotion_mask]['correct'].mean()
                emotion_count = emotion_mask.sum()
                print(f"  {emotion}: {emotion_acc:.4f} ({emotion_count} samples)")
        
    return pred_list, label_list_gt

def main(args):
    @dataclass
    class Config():
        mask_time_length: int = 3
    
    text_model = "roberta-large"
    audio_model = "facebook/data2vec-audio-base-960h"
    video_model = "facebook/timesformer-base-finetuned-k400"
    
    data_path = './AffWild2/data/'
    print(f"Using local data directory: {data_path}")
    
    if args.test_split == 'dev':
        test_path = data_path + 'affwild2_dev_emotions_whisper_filtered_complete_fixed.csv'
        split_name = 'DEV'
    else:  # test
        test_path = data_path + 'affwild2_test_emotions_whisper_filtered_complete_fixed.csv'
        split_name = 'TEST'
    
    print(f"Loading {split_name} dataset from: {test_path}")
    

    
    test_dataset = affwild2_dataset(preprocessing(test_path))
    clsNum = len(test_dataset.emoList)
    print(f"Number of classes: {clsNum}")
    print(f"Class labels: {test_dataset.emoList}")
    print(f"Number of test samples: {len(test_dataset)}")
    
    init_config = Config()
    
    if args.num_runs > 1:
        print(f"Running {args.num_runs} inference runs with different random seeds...")
        all_results = []
        
        for run in range(args.num_runs):
            run_seed = args.seed + run
            print(f"\n=== Run {run + 1}/{args.num_runs} with seed {run_seed} ===")
            
            seed_everything(run_seed)
            
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, 
                                   num_workers=0, collate_fn=make_batchs_affwild)
            
            print(f"Loading teacher model from: {args.teacher_path}")
            model_t = Teacher_model(text_model, clsNum)
            model_t.load_state_dict(torch.load(args.teacher_path))
            for para in model_t.parameters():
                para.requires_grad = False
            model_t = model_t.cuda()
            model_t.eval()
            
            print(f"Loading student audio model from: {args.audio_student_path}")
            audio_s = Student_Audio(audio_model, clsNum, init_config)
            audio_s.load_state_dict(torch.load(args.audio_student_path))
            for para in audio_s.parameters():
                para.requires_grad = False
            audio_s = audio_s.cuda()
            audio_s.eval()
            
            print(f"Loading student video model from: {args.video_student_path}")
            video_s = Student_Video(video_model, clsNum)
            video_s.load_state_dict(torch.load(args.video_student_path))
            for para in video_s.parameters():
                para.requires_grad = False
            video_s = video_s.cuda()
            video_s.eval()
            
            print(f"Loading TelMESC fusion model from: {args.model_path}")
            if not os.path.exists(args.model_path):
                print(f"ERROR: Fusion model file not found: {args.model_path}")
                return
            
            fusion = ASF_SCMMTelMESC(
                clsNum, 768, 1e-1, 0.2, 3,
                scmm_hidden_size=768,
                scmm_max_context=args.scmm_max_context,
                scmm_dropout=args.scmm_dropout,
                scmm_path_dropout=args.scmm_path_dropout
            )
            fusion.load_state_dict(torch.load(args.model_path, map_location='cpu'))
            fusion = fusion.cuda()
            fusion.eval()
            
            pred_list, label_list = inference_with_analysis(
                model_t, audio_s, video_s, fusion, test_loader, args.output_dir, gate_mode=args.gate_mode, save_csv=args.save_predictions_csv, test_split=args.test_split
            )
            
            test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(label_list, pred_list, average='weighted')
            run_result = {
                'run': run + 1,
                'seed': run_seed,
                'precision': test_pre,
                'recall': test_rec,
                'f1_score': test_fbeta,
                'accuracy': (np.array(pred_list) == np.array(label_list)).mean()
            }
            all_results.append(run_result)
            
            print(f"Run {run + 1} Results:")
            print(f"Precision: {test_pre:.4f}")
            print(f"Recall: {test_rec:.4f}")
            print(f"F1-Score: {test_fbeta:.4f}")
            print(f"Accuracy: {run_result['accuracy']:.4f}")
            
    
            del model_t, audio_s, video_s, fusion
            torch.cuda.empty_cache()
        

        avg_precision = np.mean([r['precision'] for r in all_results])
        avg_recall = np.mean([r['recall'] for r in all_results])
        avg_f1 = np.mean([r['f1_score'] for r in all_results])
        avg_accuracy = np.mean([r['accuracy'] for r in all_results])
        
        std_precision = np.std([r['precision'] for r in all_results])
        std_recall = np.std([r['recall'] for r in all_results])
        std_f1 = np.std([r['f1_score'] for r in all_results])
        std_accuracy = np.std([r['accuracy'] for r in all_results])
        
        print(f"\n=== AVERAGE RESULTS OVER {args.num_runs} RUNS ===")
        print(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"Recall: {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        
        results_df = pd.DataFrame(all_results)
        results_csv_path = os.path.join(args.output_dir, f'multi_run_results_{args.gate_mode}.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"Detailed results saved to: {results_csv_path}")
        
        # Save summary results
        summary_results = {
            'model': 'TelMESC',
            'gate_mode': args.gate_mode,
            'num_runs': args.num_runs,
            'avg_precision': avg_precision,
            'std_precision': std_precision,
            'avg_recall': avg_recall,
            'std_recall': std_recall,
            'avg_f1_score': avg_f1,
            'std_f1_score': std_f1,
            'avg_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy
        }
        summary_csv_path = os.path.join(args.output_dir, f'summary_results_{args.gate_mode}.csv')
        pd.DataFrame([summary_results]).to_csv(summary_csv_path, index=False)
        print(f"Summary results saved to: {summary_csv_path}")
        
    else:

        seed_everything(args.seed)
        
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                               num_workers=0, collate_fn=make_batchs_affwild)
        
        print(f"Loading teacher model from: {args.teacher_path}")
        model_t = Teacher_model(text_model, clsNum)
        model_t.load_state_dict(torch.load(args.teacher_path))
        for para in model_t.parameters():
            para.requires_grad = False
        model_t = model_t.cuda()
        model_t.eval()
        
        print(f"Loading student audio model from: {args.audio_student_path}")
        audio_s = Student_Audio(audio_model, clsNum, init_config)
        audio_s.load_state_dict(torch.load(args.audio_student_path))
        for para in audio_s.parameters():
            para.requires_grad = False
        audio_s = audio_s.cuda()
        audio_s.eval()
        
        print(f"Loading student video model from: {args.video_student_path}")
        video_s = Student_Video(video_model, clsNum)
        video_s.load_state_dict(torch.load(args.video_student_path))
        for para in video_s.parameters():
            para.requires_grad = False
        video_s = video_s.cuda()
        video_s.eval()
        
        print(f"Loading TelMESC fusion model from: {args.model_path}")
        if not os.path.exists(args.model_path):
            print(f"ERROR: Fusion model file not found: {args.model_path}")
            return
        
        fusion = ASF_SCMMTelMESC(
            clsNum, 768, 1e-1, 0.2, 3,
            scmm_hidden_size=768,
            scmm_max_context=args.scmm_max_context,
            scmm_dropout=args.scmm_dropout,
            scmm_path_dropout=args.scmm_path_dropout
        )
        fusion.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        fusion = fusion.cuda()
        fusion.eval()
        
        print(f"Running inference with {args.gate_mode} gate mode...")
        pred_list, label_list = inference_with_analysis(
            model_t, audio_s, video_s, fusion, test_loader, args.output_dir, gate_mode=args.gate_mode, save_csv=args.save_predictions_csv, test_split=args.test_split
        )
        
        test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(label_list, pred_list, average='weighted')
        print(f"\nOriginal Test Results ({args.gate_mode} gate mode):")
        print(f"Precision: {test_pre:.4f}")
        print(f"Recall: {test_rec:.4f}")
        print(f"F1-Score: {test_fbeta:.4f}")
        
        print(f"Sample test predictions: {pred_list[:20]}")
        print(f"Sample test labels     : {label_list[:20]}")
        pred_unique, pred_counts = np.unique(pred_list, return_counts=True)
        label_unique, label_counts = np.unique(label_list, return_counts=True)
        print(f"Test predictions distribution: {dict(zip(pred_unique, pred_counts))}")
        print(f"Test labels distribution: {dict(zip(label_unique, label_counts))}")
        
    
    print(f"\nResults saved to {args.output_dir}")
    print("---------------TelMESC Inference Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args) 