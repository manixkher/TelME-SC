
import glob
import os
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_recall_fscore_support

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
import gc

from sklearn.metrics import confusion_matrix

from AffWild2.data.preprocessing import *
from AffWild2.utils.utils import *
from AffWild2.data.dataset import *
from AffWild2.models.model import *


def parse_args():
    """
    Parse arguments.

    Returns:
        argparse.Namespace: Arguments
    """
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--epochs', default=10, type=int, help='epoch for training.')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate for training.')
    parser.add_argument('--batch_size', default=4, type=int, help='batch for training.')
    parser.add_argument('--seed', default=42, type=int, help='random seed fix')
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

def CELoss(pred_outs, labels):
    """
    Compute the cross-entropy loss.

    Args:
        pred_outs (torch.Tensor): Predicted outputs
        labels (torch.Tensor): True labels

    Returns:
        torch.Tensor: Cross-entropy loss
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

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
    best_dev_fscore, best_test_fscore = 0, 0   
    best_epoch = 0

    first_fusion_param = list(fusion.parameters())[0].clone().detach().cpu().numpy()

    for epoch in tqdm(range(training_epochs)):
        fusion.train() 
        epoch_loss = 0
        num_batches = 0
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, Learning rate: {current_lr:.2e}")
        
        for i_batch, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
                batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), batch_labels.cuda()
                
                text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
                audio_hidden, audio_logits = audio_s(audio_inputs)
                video_hidden, video_logits = video_s(video_inputs)
                
                pred_logits = fusion(text_hidden, audio_hidden, video_hidden)

                loss_val = CELoss(pred_logits, batch_labels)

            scaler.scale(loss_val).backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            if i_batch % 200 == 0:
                print("Sample batch predictions:", pred_logits.argmax(1)[:5].detach().cpu().numpy())
                print("Sample batch labels     :", batch_labels[:5].detach().cpu().numpy())
                print(f"Batch {i_batch} loss_val: {loss_val.item():.6f}")
                print("Sample pred_logits (first 2):", pred_logits[:2].detach().cpu().numpy())
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Batch {i_batch} learning rate: {current_lr:.2e}")
                if torch.isnan(pred_logits).any():
                    print(f"NaN detected in pred_logits at batch {i_batch}!")
                    print(pred_logits)
                if torch.isnan(loss_val):
                    print(f"NaN detected in loss_val at batch {i_batch}!")

                if i_batch % 200 == 0:
                    print("ASF component gradients:")
                    asf_components = ['multihead_attn', 'W_hav', 'W_av', 'LayerNorm', 'AV_LayerNorm', 'W']
                    for component in asf_components:
                        for name, param in fusion.named_parameters():
                            if component in name:
                                if param.grad is not None:
                                    print(f"   ASF.{name}: {param.grad.norm().item():.6f}")
                                else:
                                    print(f"   ASF.{name}: No grad")
                                break  

            epoch_loss += loss_val.item()
            num_batches += 1

        print(f"Average training loss for epoch {epoch+1}: {epoch_loss / num_batches:.4f}")

        final_lr = scheduler.get_last_lr()[0]
        print(f"End of epoch {epoch+1}, Final learning rate: {final_lr:.2e}")

        last_fusion_param = list(fusion.parameters())[0].clone().detach().cpu().numpy()
        print("First fusion param (start of training):", first_fusion_param.flatten()[:5])
        print("First fusion param (end of epoch):", last_fusion_param.flatten()[:5])
        first_fusion_param = last_fusion_param.copy()

        fusion.eval()   
        dev_pred_list, dev_label_list = evaluation(model_t, audio_s, video_s, fusion, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        print(f"Epoch {epoch+1}, dev_score: {dev_fbeta:.4f}")

        print("Sample dev predictions:", dev_pred_list[:20])
        print("Sample dev labels     :", dev_label_list[:20])

        pred_unique, pred_counts = np.unique(dev_pred_list, return_counts=True)
        label_unique, label_counts = np.unique(dev_label_list, return_counts=True)
        print("Unique predictions and counts:", dict(zip(pred_unique, pred_counts)))
        print("Unique labels and counts     :", dict(zip(label_unique, label_counts)))
        
        print("Confusion matrix:\n", confusion_matrix(dev_label_list, dev_pred_list))

        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta
            best_epoch = epoch
            _SaveModel(fusion, save_path)

            fusion.eval()
            test_pred_list, test_label_list = evaluation(model_t, audio_s, video_s, fusion, test_dataloader)
            test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')                
            print(f"test_score: {test_fbeta:.4f}")

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
            """Prediction"""
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), batch_labels.cuda()
                
            text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
            audio_hidden, audio_logits = audio_s(audio_inputs)
            video_hidden, video_logits = video_s(video_inputs)
                
            pred_logits = fusion(text_hidden, audio_hidden, video_hidden)
            
            """Calculation"""    

            pred_label = pred_logits.argmax(1).detach().cpu().numpy() 
            true_label = batch_labels.detach().cpu().numpy()
            
            pred_list.extend(pred_label)
            label_list.extend(true_label)

    return pred_list, label_list


def _SaveModel(model, path):
    """
    Save the model.

    Args:
        model (torch.nn.Module): Model
        path (str): Path to save the model
    """
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'exp_total_fusion_affwild2_telme.bin'))

def main(args):
    seed_everything(args.seed)
    @dataclass
    class Config():
        mask_time_length: int = 3
    """Dataset Loading"""
    
    text_model = "roberta-large"
    audio_model = "facebook/data2vec-audio-base-960h"
    video_model = "facebook/timesformer-base-finetuned-k400"

    # Always use local data directory
    data_path = './AffWild2/data/'
    print(f"Using local data directory: {data_path}")

    # Use Whisper filtered CSV files
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
    fusion = fusion.cuda()
    fusion.eval()

    """Training Setting"""        
    training_epochs = args.epochs
    save_term = int(training_epochs/5)
    max_grad_norm = 10
    lr = args.learning_rate
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler()

    model_train(training_epochs, model_t, audio_s, video_s, fusion, train_loader, dev_loader, test_loader, optimizer, scheduler, max_grad_norm, scaler, save_path)
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args) 