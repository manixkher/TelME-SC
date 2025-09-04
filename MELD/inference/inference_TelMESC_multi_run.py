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

from transformers import RobertaTokenizer
import gc
from sklearn.metrics import confusion_matrix

from MELD.data.preprocessing import *
from MELD.utils.utils import *
from MELD.data.dataset import *
from MELD.models.model_sac import Teacher_model, Student_Audio, Student_Video
from MELD.models_new.TelMESC import ASF_SCMMTelMESC


def parse_args():
    """Parse arguments.

    Returns:
        argparse.Namespace: Arguments
    """
    parser = argparse.ArgumentParser(description='TelMESC Multi-Run Inference with Analysis')
    parser.add_argument('--batch_size', default=4, type=int, help='batch for inference.')
    parser.add_argument('--seed', default=42, type=int, help='random seed fix')
    parser.add_argument('--model_path', default='./MELD/save_model/TelMESC_dropout0.2_1e-5_epoch_3.bin', type=str, help='path to TelMESC model')
    parser.add_argument('--output_dir', default='./MELD/output/TelMESC_analysis_multi_runs', type=str, help='output directory for analysis')
    parser.add_argument('--gate_mode', default='learned', choices=['learned', 'uniform'], help='gate mode: learned or uniform routing')
    parser.add_argument('--scmm_max_context', default=12, type=int, help='Maximum context size for TelMESC')
    parser.add_argument('--scmm_dropout', default=0.1, type=float, help='TelMESC dropout rate')
    parser.add_argument('--scmm_path_dropout', default=0.0, type=float, help='Path dropout probability for TelMESC')
    parser.add_argument('--num_runs', default=1, type=int, help='Number of runs with different random seeds to average results')
    args = parser.parse_args()
    return args

def seed_everything(seed):
    """Seed everything.

    Args:
        seed (int): Seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def inference_with_analysis(model_t, audio_s, video_s, fusion, dataloader, output_dir, gate_mode="learned"):
    """Inference with analysis.

    Args:
        model_t (nn.Module): Teacher model
        audio_s (nn.Module): Audio model
        video_s (nn.Module): Video model
        fusion (nn.Module): Fusion model
        dataloader (DataLoader): Dataloader
        output_dir (str): Output directory
        gate_mode (str, optional): Gate mode. Defaults to "learned".
    """
    fusion.eval()
    label_list = []
    pred_list = []
    
    gate_decisions_list = []
    gate_logits_list = []
    path_outputs_list = []
    path_norms_list = []
    context_usage_stats = {'direct': 0, 'local': 0, 'global': 0}
    
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader, desc="Running TelMESC inference")):            
            """Prediction"""
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), batch_labels.cuda()
                
            text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
            audio_hidden, audio_logits = audio_s(audio_inputs)
            video_hidden, video_logits = video_s(video_inputs)
                
            pred_logits, _, scmm_info = fusion(
                text_hidden, audio_hidden, video_hidden, 
                return_scmm_info=True, 
                gate_mode=gate_mode
            )
            
            """Calculation"""    
            pred_label = pred_logits.argmax(1).detach().cpu().numpy() 
            true_label = batch_labels.detach().cpu().numpy()
            
            pred_list.extend(pred_label)
            label_list.extend(true_label)
            
            gate_decisions = scmm_info['gate_decisions'].detach().cpu().numpy()
            gate_logits = scmm_info['gate_logits'].detach().cpu().numpy()
            path_outputs = scmm_info['path_outputs'].detach().cpu().numpy()
            
            path_norms = np.linalg.norm(path_outputs, axis=2)
            
            gate_decisions_list.append(gate_decisions)
            gate_logits_list.append(gate_logits)
            path_outputs_list.append(path_outputs)
            path_norms_list.append(path_norms)
            
            for i in range(gate_decisions.shape[0]):
                best_path = np.argmax(gate_decisions[i])
                if best_path == 0:
                    context_usage_stats['direct'] += 1
                elif best_path == 1:
                    context_usage_stats['local'] += 1
                else:
                    context_usage_stats['global'] += 1
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, f'predictions_{gate_mode}.npy'), np.array(pred_list))
    np.save(os.path.join(output_dir, f'labels_{gate_mode}.npy'), np.array(label_list))
    
    np.save(os.path.join(output_dir, f'gate_decisions_{gate_mode}.npy'), np.concatenate(gate_decisions_list, axis=0))
    np.save(os.path.join(output_dir, f'gate_logits_{gate_mode}.npy'), np.concatenate(gate_logits_list, axis=0))
    np.save(os.path.join(output_dir, f'path_outputs_{gate_mode}.npy'), np.concatenate(path_outputs_list, axis=0))
    np.save(os.path.join(output_dir, f'path_norms_{gate_mode}.npy'), np.concatenate(path_norms_list, axis=0))
    
    context_usage_df = pd.DataFrame([context_usage_stats])
    context_usage_df.to_csv(os.path.join(output_dir, f'context_usage_{gate_mode}.csv'), index=False)
    
    total_samples = len(pred_list)
    print(f"\nContext Usage Statistics ({gate_mode} gate mode):")
    print(f"Total samples: {total_samples}")
    for context_type, count in context_usage_stats.items():
        percentage = (count / total_samples) * 100
        print(f"  {context_type}: {count} ({percentage:.2f}%)")
    
    return pred_list, label_list

def load_TelMESC(model, model_path, target_batch_size=1):
    """Load TelMESC model.

    Args:
        model (nn.Module): Model
        model_path (str): Model path
        target_batch_size (int, optional): Target batch size. Defaults to 1.
    """
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        for key in state_dict.keys():
            if 'buffer' in key and state_dict[key].dim() > 0:
                current_size = state_dict[key].size(0)
                if current_size != target_batch_size:
                    print(f"Resizing buffer {key} from {current_size} to {target_batch_size}")
                    if target_batch_size > current_size:
                        new_buffer = torch.zeros(target_batch_size, *state_dict[key].shape[1:], 
                                               dtype=state_dict[key].dtype)
                        new_buffer[:current_size] = state_dict[key]
                        state_dict[key] = new_buffer
                    else:
                            state_dict[key] = state_dict[key][:target_batch_size]
        
        model.load_state_dict(state_dict)
        print(f"TelMESC model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading TelMESC model: {e}")
        return False

def main(args):
    @dataclass
    class Config():
        mask_time_length: int = 3
    
    text_model = "roberta-large"
    audio_model = "facebook/data2vec-audio-base-960h"
    video_model = "facebook/timesformer-base-finetuned-k400"

    data_path = './dataset/MELD.Raw/'
    test_path = data_path + 'test_meld_emo.csv'

    test_dataset = meld_dataset(preprocessing(test_path))
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
            
            test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=True, num_workers=16, collate_fn=make_batchs)

            '''teacher model load'''
            model_t = Teacher_model(text_model, clsNum)
            model_t.load_state_dict(torch.load('./MELD/save_model/teacher.bin'))
            for para in model_t.parameters():
                para.requires_grad = False
            model_t = model_t.cuda()
            model_t.eval()

            '''student model'''
            audio_s = Student_Audio(audio_model, clsNum, init_config)
            audio_s.load_state_dict(torch.load('./MELD/save_model/student_audio/total_student.bin')) 
            for para in audio_s.parameters():
                para.requires_grad = False
            audio_s = audio_s.cuda()
            audio_s.eval()

            video_s = Student_Video(video_model, clsNum)
            video_s.load_state_dict(torch.load('./MELD/save_model/student_video/total_student.bin')) 
            for para in video_s.parameters():
                para.requires_grad = False
            video_s = video_s.cuda()
            video_s.eval()

            '''TelMESC fusion'''
            hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
            fusion = ASF_SCMMTelMESC(
                clsNum, hidden_size, beta_shift, dropout_prob, num_head,
                scmm_hidden_size=768,
                scmm_max_context=args.scmm_max_context,
                scmm_dropout=args.scmm_dropout,
                scmm_path_dropout=args.scmm_path_dropout
            )
            
            asf_checkpoint_path = './MELD/save_model/total_fusion.bin'
            if os.path.exists(asf_checkpoint_path):
                print(f"Loading pre-trained ASF weights from {asf_checkpoint_path}")
                asf_state_dict = torch.load(asf_checkpoint_path, map_location='cpu')
                fusion.load_asf_weights(asf_state_dict)
                print("ASF weights loaded successfully")
            
            if os.path.exists(args.model_path):
                load_TelMESC(fusion, args.model_path)
            else:
                print(f"Warning: TelMESC model not found at {args.model_path}")
                print("Using randomly initialized TelMESC model for analysis")
            
            fusion = fusion.cuda()
            fusion.eval()
            
            pred_list, label_list = inference_with_analysis(
                model_t, audio_s, video_s, fusion, test_loader, 
                args.output_dir, gate_mode=args.gate_mode
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
        
        test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs)

        '''teacher model load'''
        model_t = Teacher_model(text_model, clsNum)
        model_t.load_state_dict(torch.load('./MELD/save_model/teacher.bin'))
        for para in model_t.parameters():
            para.requires_grad = False
        model_t = model_t.cuda()
        model_t.eval()

        '''student model'''
        audio_s = Student_Audio(audio_model, clsNum, init_config)
        audio_s.load_state_dict(torch.load('./MELD/save_model/student_audio/total_student.bin')) 
        for para in audio_s.parameters():
            para.requires_grad = False
        audio_s = audio_s.cuda()
        audio_s.eval()

        video_s = Student_Video(video_model, clsNum)
        video_s.load_state_dict(torch.load('./MELD/save_model/student_video/total_student.bin')) 
        for para in video_s.parameters():
            para.requires_grad = False
        video_s = video_s.cuda()
        video_s.eval()

        '''TelMESC fusion'''
        hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
        fusion = ASF_SCMMTelMESC(
            clsNum, hidden_size, beta_shift, dropout_prob, num_head,
            scmm_hidden_size=768,
            scmm_max_context=args.scmm_max_context,
            scmm_dropout=args.scmm_dropout,
            scmm_path_dropout=args.scmm_path_dropout
        )
        
        asf_checkpoint_path = './MELD/save_model/total_fusion.bin'
        if os.path.exists(asf_checkpoint_path):
            print(f"Loading pre-trained ASF weights from {asf_checkpoint_path}")
            asf_state_dict = torch.load(asf_checkpoint_path, map_location='cpu')
            fusion.load_asf_weights(asf_state_dict)
            print("ASF weights loaded successfully")
        
        if os.path.exists(args.model_path):
            load_TelMESC(fusion, args.model_path)
        else:
            print(f"Warning: TelMESC model not found at {args.model_path}")
            print("Using randomly initialized TelMESC model for analysis")
        
        fusion = fusion.cuda()
        fusion.eval()

        print(f"Running TelMESC inference with {args.gate_mode} gate mode...")
        pred_list, label_list = inference_with_analysis(
            model_t, audio_s, video_s, fusion, test_loader, 
            args.output_dir, gate_mode=args.gate_mode
        )
        
        test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(label_list, pred_list, average='weighted')
        print(f"\nTest Results ({args.gate_mode} gate mode):")
        print(f"Precision: {test_pre:.4f}")
        print(f"Recall: {test_rec:.4f}")
        print(f"F1-Score: {test_fbeta:.4f}")
        
        class_names = test_dataset.emoList
        report = classification_report(label_list, pred_list, target_names=class_names, output_dict=True)
        
        with open(os.path.join(args.output_dir, f'classification_report_{args.gate_mode}.txt'), 'w') as f:
            f.write(classification_report(label_list, pred_list, target_names=class_names))
        
        
        cm = confusion_matrix(label_list, pred_list)
        np.save(os.path.join(args.output_dir, f'confusion_matrix_{args.gate_mode}.npy'), cm)
    
    print(f"\nResults saved to {args.output_dir}")
    print("---------------TelMESC Inference Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args) 