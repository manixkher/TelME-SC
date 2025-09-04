import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
from dataclasses import dataclass
import argparse

from tqdm import tqdm
from AffWild2.data.preprocessing import preprocessing
from AffWild2.data.dataset import affwild2_dataset
from AffWild2.utils.utils import make_batchs_affwild, seed_everything
from AffWild2.models.model import Teacher_model, Student_Audio, Student_Video
from MELD.models_new.TelMESC import ASF_SCMMTelMESC


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
warnings.filterwarnings("ignore", message=".*__audioread_load.*")

def parse_args():
    """
    Parse arguments.

    Returns:
        argparse.Namespace: Arguments
    """
    parser = argparse.ArgumentParser(description='SCMM TelMESC Training for AffWild2')
    parser.add_argument('--epochs', default=15, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--asf_lr', default=1e-6, type=float, help='ASF learning rate')
    parser.add_argument('--scmm_lr', default=1e-4, type=float, help='SCMM learning rate')
    parser.add_argument('--scmm_max_context', default=12, type=int, help='Maximum context size for SCMM')
    parser.add_argument('--scmm_dropout', default=0.1, type=float, help='SCMM dropout rate')
    parser.add_argument('--beta_shift', default=1e-1, type=float, help='Beta shift parameter for fusion model')
    parser.add_argument('--model_name', default='telmesc', type=str, help='Model name for saving (e.g., "telmesc_beta_2e1", "telmesc_beta_1e2")')

    parser.add_argument('--scmm_path_dropout', default=0.0, type=float, help='Path dropout probability for SCMM (0.0-0.3)')
    parser.add_argument('--scmm_total_epochs', default=15, type=int, help='Total epochs for path dropout annealing')
    parser.add_argument('--pretrained_asf_path', default='./AffWild2/save_model/total_fusion.bin', type=str, help='Path to pre-trained ASF weights')
    parser.add_argument('--train_concurrent', action='store_true', help='Train ASF and SCMM concurrently instead of loading pre-trained ASF')
    parser.add_argument('--model_suffix', default='', type=str, help='Suffix for saved model filename (e.g., "_concurrent", "_scmm_only")')
    args = parser.parse_args()
    return args


def seed_everything(seed):
    """
    Seed everything.

    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def CELoss(pred_outs, labels):
    """
    Compute the cross-entropy loss.

    Args:
        pred_outs (torch.Tensor): Predicted outputs
        labels (torch.Tensor): True labels

    Returns:
        torch.Tensor: Cross-entropy loss
    """
    return nn.CrossEntropyLoss()(pred_outs, labels)


def model_train(training_epochs, model_t, audio_s, video_s, fusion, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, scaler, save_path):
    """
    Train the fusion model.

    Args:
        training_epochs (int): Number of training epochs
        model_t (torch.nn.Module): Teacher model
        audio_s (torch.nn.Module): Audio student model
        video_s (torch.nn.Module): Video student model
        fusion (torch.nn.Module): Fusion model
        train_dataloader (torch.utils.data.DataLoader): Training data loader
        dev_dataloader (torch.utils.data.DataLoader): Development data loader
        test_dataloader (torch.utils.data.DataLoader): Test data loader
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler.LambdaLR): Learning rate scheduler
        max_grad_norm (float): Maximum gradient norm
        scaler (torch.cuda.amp.GradScaler): Gradient scaler
        save_path (str): Path to save the model
    """
    fusion.train()
    best_dev_fscore = 0
    best_test_fscore = 0
    best_epoch = 0
    
    # Track parameter changes for debugging
    initial_params = {}
    for name, param in fusion.named_parameters():
        initial_params[name] = param.clone().detach()
    
    for epoch in range(training_epochs):
        print(f"\n[INFO] Starting epoch {epoch+1}/{training_epochs}")
        

        fusion.train()
        
        fusion.update_epoch(epoch)
        
        epoch_loss = 0
        num_batches = 0
        

        for i_batch, data in enumerate(train_dataloader):
            # Get data from batch
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), batch_labels.cuda()
            
            # Pass data to text teacher, audio student, video student
            text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
            audio_hidden, audio_logits = audio_s(audio_inputs)
            video_hidden, video_logits = video_s(video_inputs)
            
            # Pass data to fusion model
            pred_logits, total_loss, scmm_info = fusion(
                text_hidden, audio_hidden, video_hidden, 
                labels=batch_labels, return_scmm_info=True, gate_mode="learned")
            
            scaler.scale(total_loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            scheduler.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            # Debug printing every 200 batches
            if (i_batch % 200 == 0) or (i_batch == 0):
                print(f"\n Epoch {epoch+1}, Batch {i_batch}")
                

                pred_labels = pred_logits.argmax(1).detach().cpu().numpy()
                true_labels = batch_labels.detach().cpu().numpy()
                print(f"Sample predictions: {pred_labels}")
                print(f"Sample labels: {true_labels}")
                

                gate_decisions = scmm_info['gate_decisions'].detach().cpu().numpy()
                path_outputs = scmm_info['path_outputs'].detach().cpu().numpy()
                path_names = ['Direct', 'Local', 'Global']
                

                avg_gate_decisions = gate_decisions.mean(axis=0)
                print(f"Path selection probabilities:")
                for i, (name, avg_prob) in enumerate(zip(path_names, avg_gate_decisions)):
                    print(f"  {name}: {avg_prob:.4f}")
                

                print(f"Individual sample analysis:")
                actual_batch_size = min(5, len(gate_decisions))  
                for i in range(actual_batch_size):
                    sample_decisions = gate_decisions[i]
                    dominant_path = path_names[sample_decisions.argmax()]
                    print(f"  Sample {i}: {sample_decisions} -> {dominant_path}")
                

                print(f"Path output statistics:")
                for i, name in enumerate(path_names):
                    if 'path_norms' in scmm_info:
                        path_norm = scmm_info['path_norms'][:, i].mean().item()
                    else:
                        path_norm = 0.0
                    if 'path_outputs_mean' in scmm_info:
                        path_mean = scmm_info['path_outputs_mean'][:, i].mean().item()
                    else:
                        path_mean = 0.0
                    if 'path_outputs_std' in scmm_info:
                        path_std = scmm_info['path_outputs_std'][:, i].mean().item()
                    else:
                        path_std = 0.0
                    print(f"  {name}: norm={path_norm:.4f}, mean={path_mean:.4f}, std={path_std:.4f}")
                

                print(f"Loss breakdown:")
                print(f"  Total loss: {total_loss.item():.6f}")
                if 'main_loss' in scmm_info:
                    print(f"  Main loss: {scmm_info['main_loss'].item():.6f}")
                if 'auxiliary_loss' in scmm_info:
                    print(f"  Auxiliary loss: {scmm_info['auxiliary_loss'].item():.6f}")
                if 'latency_loss' in scmm_info:
                    print(f"  Latency loss: {scmm_info['latency_loss'].item():.6f}")
                if 'entropy_loss' in scmm_info:
                    print(f"  Entropy loss: {scmm_info['entropy_loss'].item():.6f}")
                
                asf_lr = optimizer.param_groups[0]['lr']
                scmm_lr = optimizer.param_groups[1]['lr']
                print(f"Current learning rates - ASF: {asf_lr:.2e}, SCMM: {scmm_lr:.2e}")
                
                print("Sample pred_logits (first 2):", pred_logits[:2].detach().cpu().numpy())
        
        print(f"Average training loss for epoch {epoch+1}: {epoch_loss / num_batches:.4f}")
        
        asf_final_lr = optimizer.param_groups[0]['lr']
        scmm_final_lr = optimizer.param_groups[1]['lr']
        print(f"End of epoch {epoch+1}, Final ASF learning rate: {asf_final_lr:.2e}, Final SCMM learning rate: {scmm_final_lr:.2e}")
        
        last_fusion_param = list(fusion.parameters())[0].clone().detach().cpu().numpy()
        first_param_name = list(initial_params.keys())[0]
        first_param_value = initial_params[first_param_name].cpu().numpy()
        print("First fusion param (start of training):", first_param_value.flatten()[:5])
        print("First fusion param (end of epoch):", last_fusion_param.flatten()[:5])

        fusion.eval()   
        dev_pred_list, dev_label_list = evaluation(model_t, audio_s, video_s, fusion, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        print(f"Epoch {epoch+1}, dev_score: {dev_fbeta:.4f}")

        print("Sample dev predictions:", dev_pred_list[:20])
        print("Sample dev labels     :", dev_label_list[:20])

        pred_unique, pred_counts = np.unique(dev_pred_list, return_counts=True)
        label_unique, label_counts = np.unique(dev_label_list, return_counts=True)
        print("Dev predictions distribution:", dict(zip(pred_unique, pred_counts)))
        print("Dev labels distribution:", dict(zip(label_unique, label_counts)))

        test_pred_list, test_label_list = evaluation(model_t, audio_s, video_s, fusion, test_dataloader)
        test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
        print(f"Epoch {epoch+1}, test_score: {test_fbeta:.4f}")

        print("Sample test predictions:", test_pred_list[:20])
        print("Sample test labels     :", test_label_list[:20])

        pred_unique, pred_counts = np.unique(test_pred_list, return_counts=True)
        label_unique, label_counts = np.unique(test_label_list, return_counts=True)
        print("Test predictions distribution:", dict(zip(pred_unique, pred_counts)))
        print("Test labels distribution:", dict(zip(label_unique, label_counts)))
        
        _SaveModel(fusion, save_path, epoch, args.model_suffix, args.model_name)
        
        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta
            best_test_fscore = test_fbeta
            best_epoch = epoch
            print(f"New best model at epoch {epoch+1} (dev_fbeta: {dev_fbeta:.4f})")
        
        print(f"Best dev F1-score so far: {best_dev_fscore:.4f} (epoch {best_epoch + 1})")
        print(f"Best test F1-score so far: {best_test_fscore:.4f}")


def evaluation(model_t, audio_s, video_s, fusion, dataloader):
    """
    Evaluate the fusion model.

    Args:
        model_t (torch.nn.Module): Teacher model
        audio_s (torch.nn.Module): Audio student model
        video_s (torch.nn.Module): Video student model
        fusion (torch.nn.Module): Fusion model
        dataloader (torch.utils.data.DataLoader): Data loader

    Returns:
        list: Predicted labels
        list: True labels
    """
    fusion.eval()
    label_list = []
    pred_list = []
    
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), batch_labels.cuda()
                
            text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
            audio_hidden, audio_logits = audio_s(audio_inputs)
            video_hidden, video_logits = video_s(video_inputs)
                
            pred_logits, _ = fusion(text_hidden, audio_hidden, video_hidden, labels=None, return_scmm_info=False, gate_mode="learned")
            
            pred_label = pred_logits.argmax(1).detach().cpu().numpy() 
            true_label = batch_labels.detach().cpu().numpy()
            
            pred_list.extend(pred_label)
            label_list.extend(true_label)

    return pred_list, label_list


def _SaveModel(model, path, epoch=None, model_suffix='', model_name='telmesc'):
    """
    Save the model.

    Args:
        model (torch.nn.Module): Model
        path (str): Path to save the model
        epoch (int, optional): Epoch number. Defaults to None.
        model_suffix (str, optional): Model suffix. Defaults to ''.
        model_name (str, optional): Model name. Defaults to 'telmesc'.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
    if epoch is not None:
        # Save with epoch number
        filename = f'exp_fixed_{model_name}_affwild2_epoch_{epoch+1}{model_suffix}.bin'
    else:
        # Save as latest
        filename = f'exp_fixed_{model_name}_affwild2{model_suffix}.bin'
    
    torch.save(model.state_dict(), os.path.join(path, filename))
    print(f"Model saved as {filename}")


def main(args):
    seed_everything(args.seed)
    @dataclass
    class Config():
        mask_time_length: int = 3

    text_model = "roberta-large"
    audio_model = "facebook/data2vec-audio-base-960h"
    video_model = "facebook/timesformer-base-finetuned-k400"


    data_path = './AffWild2/data/'
    print(f"Using local data directory: {data_path}")


    train_path = data_path + 'affwild2_train_emotions_whisper_filtered_complete_no_missing.csv'
    dev_path = data_path + 'affwild2_dev_emotions_whisper_filtered_complete_fixed.csv'
    test_path = data_path + 'affwild2_test_emotions_whisper_filtered_complete_fixed.csv'

    train_dataset = affwild2_dataset(preprocessing(train_path))
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=0, collate_fn=make_batchs_affwild)

    dev_dataset = affwild2_dataset(preprocessing(dev_path))
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0, collate_fn=make_batchs_affwild)

    test_dataset = affwild2_dataset(preprocessing(test_path))
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0, collate_fn=make_batchs_affwild)

    save_path = os.path.join('./AffWild2/save_model')
    
    print("###Save Path### ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    clsNum = len(train_dataset.emoList)
    init_config = Config()

    '''teacher model load'''
    model_t = Teacher_model(text_model, clsNum)
    model_t.load_state_dict(torch.load('./AffWild2/save_model/exp_new_teacher_affwild2_whisper_epoch_4.bin'))
    for para in model_t.parameters():
        para.requires_grad = False
    model_t = model_t.cuda()
    model_t.eval()
    # affwild teacher
    '''student model'''
    audio_s = Student_Audio(audio_model, clsNum, init_config)
    audio_s.load_state_dict(torch.load('./AffWild2/save_model/student_audio/exp_student_audio_best_affwild2.bin')) 
    for para in audio_s.parameters():
        para.requires_grad = False
    audio_s = audio_s.cuda()
    audio_s.eval()
    # telme teacher for now
    video_s = Student_Video(video_model, clsNum)
    video_s.load_state_dict(torch.load('./AffWild2/save_model/student_video/exp_student_visual_best_affwild2.bin')) 
    for para in video_s.parameters():
        para.requires_grad = False
    video_s = video_s.cuda()
    video_s.eval()

    hidden_size, dropout_prob, num_head = 768, 0.2, 4
    fusion = ASF_SCMMTelMESC(
        clsNum, hidden_size, args.beta_shift, dropout_prob, num_head,
        scmm_hidden_size=768,
        scmm_max_context=args.scmm_max_context,
        scmm_dropout=args.scmm_dropout,
        scmm_path_dropout=args.scmm_path_dropout,
        scmm_total_epochs=args.scmm_total_epochs
    )
    
    if args.train_concurrent:
        print(f" Training ASF and SCMM concurrently from scratch")
        print("Using randomly initialized ASF weights")
    else:
        print(f"Loading pre-trained ASF weights from {args.pretrained_asf_path}")
        if os.path.exists(args.pretrained_asf_path):
            asf_state_dict = torch.load(args.pretrained_asf_path)
            fusion.load_asf_weights(asf_state_dict)
            print("ASF weights loaded successfully")
        else:
            print(f"Pre-trained ASF weights not found at {args.pretrained_asf_path}")
            print("Using randomly initialized ASF weights")
    
    if args.train_concurrent:
        print(f"Starting concurrent ASF+SCMM TelMESC training with {args.epochs} epochs")
        print(f"ASF learning rate: {args.asf_lr:.2e}")
        print(f"SCMM learning rate: {args.scmm_lr:.2e}")
    else:
        print(f"Starting SCMM TelMESC training with {args.epochs} epochs")
        print(f"ASF learning rate: {args.asf_lr:.2e}")
        print(f"SCMM learning rate: {args.scmm_lr:.2e}")
    
    print(f"Beta shift parameter: {args.beta_shift:.2e}")
    print(f"Model name for saving: {args.model_name}")
    
    fusion = fusion.cuda()
    fusion.train()

    training_epochs = args.epochs
    max_grad_norm = 10
    
    param_groups = fusion.get_parameter_groups(asf_lr=args.asf_lr, scmm_lr=args.scmm_lr)
    
    optimizer = torch.optim.AdamW(param_groups)
    
    
    # One epoch of warm-up
    num_training_steps = len(train_loader) * training_epochs
    num_warmup_steps = len(train_loader)  
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    scaler = torch.cuda.amp.GradScaler()
    
    model_train(training_epochs, model_t, audio_s, video_s, fusion, train_loader, dev_loader, test_loader, optimizer, scheduler, max_grad_norm, scaler, save_path)
    print("---------------SCMM TelMESC Training Done--------------")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args) 