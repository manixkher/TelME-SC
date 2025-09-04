import os
import numpy as np
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_recall_fscore_support, classification_report

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaTokenizer
import gc

from MELD.data.preprocessing import *
from MELD.utils.utils import *
from MELD.data.dataset import *
from MELD.models.model_sac import Teacher_model, Student_Audio, Student_Video
from MELD.models_new.TelMESC import ASF_SCMMTelMESC

from sklearn.metrics import confusion_matrix


def parse_args():
    """Parse arguments.

    Returns:
        argparse.Namespace: Arguments
    """
    parser = argparse.ArgumentParser(description='TelMESC Inference with Analysis')
    parser.add_argument('--batch_size', default=4, type=int, help='batch for inference.')
    parser.add_argument('--seed', default=42, type=int, help='random seed fix')
    parser.add_argument('--model_path', default='./MELD/save_model/TelMESC_dropout0.2_1e-5_epoch_3.bin', type=str, help='path to TelMESC model')
    parser.add_argument('--output_dir', default='./MELD/output/TelMESC_analysis', type=str, help='output directory for analysis')
    parser.add_argument('--gate_mode', default='learned', choices=['learned', 'uniform'], help='gate mode: learned or uniform routing')
    parser.add_argument('--scmm_max_context', default=12, type=int, help='Maximum context size for TelMESC')
    parser.add_argument('--scmm_dropout', default=0.1, type=float, help='TelMESC dropout rate')
    parser.add_argument('--scmm_path_dropout', default=0.0, type=float, help='Path dropout probability for TelMESC')
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
                route_idx = np.argmax(gate_decisions[i])
                
                if route_idx == 0:
                    context_usage_stats['direct'] += 1
                elif route_idx == 1:
                    context_usage_stats['local'] += 1
                else:
                    context_usage_stats['global'] += 1

    os.makedirs(output_dir, exist_ok=True)
    
    gate_decisions_array = np.concatenate(gate_decisions_list, axis=0)
    np.save(os.path.join(output_dir, 'gate_decisions.npy'), gate_decisions_array)
    
    gate_logits_array = np.concatenate(gate_logits_list, axis=0)
    np.save(os.path.join(output_dir, 'gate_logits.npy'), gate_logits_array)
    
    path_outputs_array = np.concatenate(path_outputs_list, axis=0)
    np.save(os.path.join(output_dir, 'path_outputs.npy'), path_outputs_array)
    
    path_norms_array = np.concatenate(path_norms_list, axis=0)
    np.save(os.path.join(output_dir, 'path_norms.npy'), path_norms_array)
    
    context_sizes = scmm_info['context_sizes']
    np.save(os.path.join(output_dir, 'context_sizes.npy'), np.array(context_sizes))
    
    total_utterances = sum(context_usage_stats.values())
    context_usage_percentages = {k: v/total_utterances*100 for k, v in context_usage_stats.items()}
    
    with open(os.path.join(output_dir, 'context_usage_stats.txt'), 'w') as f:
        f.write("TelMESC Context Usage Statistics:\n")
        f.write("=" * 40 + "\n")
        f.write(f"Gate mode: {gate_mode}\n\n")
        for context_type, count in context_usage_stats.items():
            percentage = context_usage_percentages[context_type]
            f.write(f"{context_type.capitalize()}: {count} ({percentage:.2f}%)\n")
        f.write(f"\nTotal utterances: {total_utterances}\n")
        
        f.write(f"\nTelMESC-Specific Statistics:\n")
        f.write(f"Average gate logits (Direct): {np.mean(gate_logits_array[:, 0]):.4f}\n")
        f.write(f"Average gate logits (Local): {np.mean(gate_logits_array[:, 1]):.4f}\n")
        f.write(f"Average gate logits (Global): {np.mean(gate_logits_array[:, 2]):.4f}\n")
        f.write(f"Context sizes: {context_sizes}\n")
        
        f.write(f"\nPath Output Statistics:\n")
        path_names = ['Direct', 'Local', 'Global']
        for i, name in enumerate(path_names):
            avg_norm = np.mean(path_norms_array[:, i])
            f.write(f"Average {name} path norm: {avg_norm:.4f}\n")
        
        f.write(f"\nGate Decision Statistics:\n")
        for i, name in enumerate(path_names):
            avg_decision = np.mean(gate_decisions_array[:, i])
            f.write(f"Average {name} gate decision: {avg_decision:.4f}\n")
    
    print(f"TelMESC Context Usage Statistics:")
    print("=" * 40)
    print(f"Gate mode: {gate_mode}")
    for context_type, count in context_usage_stats.items():
        percentage = context_usage_percentages[context_type]
        print(f"{context_type.capitalize()}: {count} ({percentage:.2f}%)")
    print(f"Total utterances: {total_utterances}")
    
    print(f"\nTelMESC-Specific Statistics:")
    print(f"Average gate logits (Direct): {np.mean(gate_logits_array[:, 0]):.4f}")
    print(f"Average gate logits (Local): {np.mean(gate_logits_array[:, 1]):.4f}")
    print(f"Average gate logits (Global): {np.mean(gate_logits_array[:, 2]):.4f}")
    print(f"Context sizes: {context_sizes}")
    
    print(f"\nPath Output Statistics:")
    path_names = ['Direct', 'Local', 'Global']
    for i, name in enumerate(path_names):
        avg_norm = np.mean(path_norms_array[:, i])
        print(f"Average {name} path norm: {avg_norm:.4f}")
    
    print(f"\nGate Decision Statistics:")
    for i, name in enumerate(path_names):
        avg_decision = np.mean(gate_decisions_array[:, i])
        print(f"Average {name} gate decision: {avg_decision:.4f}")
    
    return pred_list, label_list

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

def main(args):
    seed_everything(args.seed)
    @dataclass
    class Config():
        mask_time_length: int = 3
    
    text_model = "roberta-large"
    audio_model = "facebook/data2vec-audio-base-960h"
    video_model = "facebook/timesformer-base-finetuned-k400"

    data_path = './dataset/MELD.Raw/'
    test_path = data_path + 'test_meld_emo.csv'

    test_dataset = meld_dataset(preprocessing(test_path))
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs)
        
    clsNum = len(test_dataset.emoList)
    init_config = Config()

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
    
    np.save(os.path.join(args.output_dir, f'predictions_{args.gate_mode}.npy'), np.array(pred_list))
    np.save(os.path.join(args.output_dir, f'labels_{args.gate_mode}.npy'), np.array(label_list))
    
    print(f"\nResults saved to {args.output_dir}")
    print("---------------TelMESC Inference Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args) 