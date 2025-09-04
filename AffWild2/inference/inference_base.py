import os
import numpy as np
import argparse
import random
import warnings
from dataclasses import dataclass
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import torch
from torch.utils.data import DataLoader
import gc

from AffWild2.data.preprocessing import preprocessing
from AffWild2.data.dataset import affwild2_dataset
from AffWild2.utils.utils import make_batchs_affwild, seed_everything
from AffWild2.models.teacher import Teacher_model
from AffWild2.models.student import Student_Audio
from AffWild2.models.student_video import Student_Video
from AffWild2.models.model import ASF

warnings.filterwarnings('ignore')

import glob
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
import gc

from AffWild2.utils import *
from AffWild2.utils.utils import make_batchs_affwild

def parse_args():
    """
    Parse arguments.

    Returns:
        argparse.Namespace: Arguments
    """
    parser = argparse.ArgumentParser(description='AffWild2 Base Fusion Inference')
    parser.add_argument('--batch_size', default=4, type=int, help='batch for inference.')
    parser.add_argument('--seed', default=42, type=int, help='random seed fix')
    parser.add_argument('--test_split', default='test', type=str, choices=['dev', 'test'], help='Which split to evaluate on')
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

def evaluation(model_t, audio_s, video_s, fusion, dataloader):
    """
    Evaluate the fusion model.

    Args:
        model_t (torch.nn.Module): Teacher model
        audio_s (torch.nn.Module): Audio student model
        video_s (torch.nn.Module): Video student model
        fusion (torch.nn.Module): Fusion model
        dataloader (torch.utils.data.DataLoader): Data loader
    """
    label_list = []
    pred_list = []
    
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader, desc="Running base fusion inference")):            
            """Prediction"""
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), batch_labels.cuda()
                
            text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
            audio_hidden, audio_logits = audio_s(audio_inputs)
            video_hidden, video_logits = video_s(video_inputs)
                
            pred_logits = fusion(text_hidden, video_hidden, audio_hidden)
            
            """Calculation"""    
            pred_label = pred_logits.argmax(1).detach().cpu().numpy() 
            true_label = batch_labels.detach().cpu().numpy()
            
            pred_list.extend(pred_label)
            label_list.extend(true_label)

    return pred_list, label_list

def main(args):
    @dataclass
    class Config():
        mask_time_length: int = 3
    """Dataset Loading"""
    
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
    
    if not os.path.exists(test_path):
        print(f"ERROR: Test file not found: {test_path}")
        print("Available files in data directory:")
        for f in os.listdir(data_path):
            if f.endswith('.csv'):
                print(f"  - {f}")
        return

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
            
            test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=True, num_workers=0, collate_fn=make_batchs_affwild)

            model_t = Teacher_model(text_model, clsNum)
            model_t.load_state_dict(torch.load('./AffWild2/save_model/exp_new_teacher_affwild2_whisper_epoch_4.bin'))
            for para in model_t.parameters():
                para.requires_grad = False
            model_t = model_t.cuda()
            model_t.eval()

            audio_s = Student_Audio(audio_model, clsNum, init_config)
            audio_s.load_state_dict(torch.load('./AffWild2/save_model/student_audio/exp_student_audio_best_affwild2.bin')) 
            for para in audio_s.parameters():
                para.requires_grad = False
            audio_s = audio_s.cuda()
            audio_s.eval()

            video_s = Student_Video(video_model, clsNum)
            video_s.load_state_dict(torch.load('./AffWild2/save_model/student_video/exp_student_visual_best_affwild2.bin')) 
            for para in video_s.parameters():
                para.requires_grad = False
            video_s = video_s.cuda()
            video_s.eval()

            hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
            fusion = ASF(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
            fusion.load_state_dict(torch.load('./AffWild2/save_model/exp_total_fusion_affwild2_telme.bin')) 
            for para in fusion.parameters():
                para.requires_grad = False
            fusion = fusion.cuda()
            fusion.eval()
            
            test_pred_list, test_label_list = evaluation(model_t, audio_s, video_s, fusion, test_loader)
            
            test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
            run_result = {
                'run': run + 1,
                'seed': run_seed,
                'precision': test_pre,
                'recall': test_rec,
                'f1_score': test_fbeta,
                'accuracy': (np.array(test_pred_list) == np.array(test_label_list)).mean()
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
        results_csv_path = os.path.join('./AffWild2/output', f'base_fusion_multi_run_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"Detailed results saved to: {results_csv_path}")
        
        summary_results = {
            'model': 'Base_Fusion',
            'gate_mode': 'N/A',
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
        summary_csv_path = os.path.join('./AffWild2/output', f'base_fusion_summary_results.csv')
        pd.DataFrame([summary_results]).to_csv(summary_csv_path, index=False)
        print(f"Summary results saved to: {summary_csv_path}")
        
    else:
        seed_everything(args.seed)
        
        test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0, collate_fn=make_batchs_affwild)

        '''teacher model load'''
        model_t = Teacher_model(text_model, clsNum)
        model_t.load_state_dict(torch.load('./AffWild2/save_model/exp_new_teacher_affwild2_whisper_epoch_4.bin'))
        for para in model_t.parameters():
            para.requires_grad = False
        model_t = model_t.cuda()
        model_t.eval()

        '''student model'''
        audio_s = Student_Audio(audio_model, clsNum, init_config)
        audio_s.load_state_dict(torch.load('./AffWild2/save_model/student_audio/exp_student_audio_best_affwild2.bin')) 
        for para in audio_s.parameters():
            para.requires_grad = False
        audio_s = audio_s.cuda()
        audio_s.eval()

        video_s = Student_Video(video_model, clsNum)
        video_s.load_state_dict(torch.load('./AffWild2/save_model/student_video/exp_student_visual_best_affwild2.bin')) 
        for para in video_s.parameters():
            para.requires_grad = False
        video_s = video_s.cuda()
        video_s.eval()

        '''fusion'''
        hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
        fusion = ASF(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
        fusion.load_state_dict(torch.load('./AffWild2/save_model/exp_total_fusion_affwild2_telme.bin')) 
        for para in fusion.parameters():
            para.requires_grad = False
        fusion = fusion.cuda()
        fusion.eval()
        
        test_pred_list, test_label_list = evaluation(model_t, audio_s, video_s, fusion, test_loader)
        print(classification_report(test_label_list, test_pred_list, target_names=test_dataset.emoList, digits=5))
        print(confusion_matrix(test_label_list, test_pred_list, normalize='true'))
    
    print("---------------Base Fusion Inference Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args) 