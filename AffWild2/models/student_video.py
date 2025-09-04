
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

import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoProcessor, Data2VecAudioModel
from transformers import AutoImageProcessor, TimesformerModel
import gc

from AffWild2.data.preprocessing import *
from AffWild2.utils.utils import *
from AffWild2.data.dataset import *
from AffWild2.models.model import *
from AffWild2.inference.affwild2_kd import *

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
    parser.add_argument('--teacher_path', default=None, type=str, help='Path to AffWild2 teacher model checkpoint')
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

def CE_Loss(pred_outs, logit_t, hidden_s, hidden_t, labels):
    """
    Compute the cross-entropy loss.

    Args:
        pred_outs (torch.Tensor): Predicted outputs
        logit_t (torch.Tensor): Teacher logits
        hidden_s (torch.Tensor): Student hidden states
        hidden_t (torch.Tensor): Teacher hidden states
        labels (torch.Tensor): True labels
    """
    ori_loss = nn.CrossEntropyLoss()
    ori_loss = ori_loss(pred_outs, labels)
    logit_loss = Logit_Loss(tau=2.0).cuda()  # Temperature tau changed to 2.0 compared to MELD.
    logit_loss = logit_loss(pred_outs, logit_t)
    feature_loss = Feature_Loss().cuda()
    feature_loss = feature_loss(hidden_s, hidden_t)

    # alpha changed to 0.7 compared to MELD.
    alpha = 0.7
    loss_val = ori_loss + (alpha * logit_loss) + feature_loss 
    return loss_val

def model_train(training_epochs, model_t, model_s, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, scaler, save_path):
    """
    Train the video student model.

    Args:
        training_epochs (int): Number of training epochs
        model_t (torch.nn.Module): Teacher model
        model_s (torch.nn.Module): Video student model
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
    model_t.eval()
    
    for epoch in tqdm(range(training_epochs)):
        print(f"\n===== Starting Epoch {epoch+1}/{training_epochs} =====")
        model_s.train() 
        epoch_loss = 0.0
        batch_count = 0
        
        for i_batch, data in enumerate(train_dataloader):
            try:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    """Prediction"""
                    batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels = data
                    batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels = batch_input_tokens.cuda(), batch_attention_masks.cuda(), batch_audio.cuda(), batch_video.cuda(), batch_labels.cuda()
                    
                    hidden_s, logit_s = model_s(batch_video)
                    hidden_t, logit_t = model_t(batch_input_tokens, batch_attention_masks)

                    loss_val = CE_Loss(logit_s, logit_t, hidden_s, hidden_t, batch_labels)
                        
                scaler.scale(loss_val).backward()
                torch.nn.utils.clip_grad_norm_(model_s.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                
                epoch_loss += loss_val.item()
                batch_count += 1
                
                # Debug every 200 batches
                if batch_count % 200 == 0:
                    avg_loss = epoch_loss / batch_count
                    print(f"VISUAL Epoch {epoch}, Batch {batch_count}: Loss={loss_val.item():.4f}, Avg_Loss={avg_loss:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")
                    
            except RuntimeError as e:
                if "All sessions in this batch failed" in str(e):
                    print(f"Warning: Skipping batch {i_batch} due to all sessions failing")
                    continue
                else:
                    raise e

        print(f"Finished training epoch {epoch+1}. Running validation...")
        model_s.eval()   
        dev_pred_list, dev_label_list = evaluation(model_s, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        print(f"visual dev_score : {dev_fbeta}")

        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta
            best_epoch = epoch
            print(f"New best dev F1: {dev_fbeta:.4f} at epoch {epoch+1}. Saving model...")
            _SaveModel(model_s, save_path)

            model_s.eval()
            print(f"Running test evaluation for best model at epoch {epoch+1}...")
            test_pred_list, test_label_list = evaluation(model_s, test_dataloader)
            test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')                
            print(f"visual test_score : {test_fbeta}")
            print("==================================================")
        else:
            print(f"Dev F1 did not improve (best so far: {best_dev_fscore:.4f} at epoch {best_epoch+1})")

    print(f"Training complete. Best dev F1: {best_dev_fscore:.4f} at epoch {best_epoch+1}")
    return best_epoch, best_dev_fscore

def evaluation(model_s, dataloader):
    """
    Evaluate the video student model.

    Args:
        model_s (torch.nn.Module): Video student model
        dataloader (torch.utils.data.DataLoader): Data loader
    """
    model_s.eval()
    label_list = []
    pred_list = []
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""    
            batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels = data
            batch_video, batch_labels = batch_video.cuda(), batch_labels.cuda()
            hidden_s, logit_s = model_s(batch_video)

            """Calculation"""    
            pred_label = logit_s.argmax(1).detach().cpu().numpy()
            true_label = batch_labels.detach().cpu().numpy()
            
            pred_list.extend(pred_label)
            label_list.extend(true_label)

    return pred_list, label_list

def _SaveModel(model, path):
    """
    Save the video student model.

    Args:
        model (torch.nn.Module): Video student model
        path (str): Path to save the model
    """
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'exp_student_visual_best_affwild2.bin'))

def main(args):
    seed_everything(args.seed)
    @dataclass
    class Config():
        mask_time_length: int = 3

    print("Loading datasets...")
    """Dataset Loading"""

    text_model = "roberta-large"
    video_model = "facebook/timesformer-base-finetuned-k400"

    # Use scratch directory if available (for cluster jobs), otherwise use local data
    if 'SCRATCH_DATA_DIR' in os.environ:
        data_path = os.environ['SCRATCH_DATA_DIR'] + '/AffWild2/data/'
        print(f"Using scratch data directory: {data_path}")
    else:
        data_path = './AffWild2/data/'
        print(f"Using local data directory: {data_path}")

    # Use Whisper filtered CSV files
    train_path = data_path + 'affwild2_train_emotions_whisper_filtered_complete_no_missing.csv'
    dev_path = data_path + 'affwild2_dev_emotions_whisper_filtered_complete_fixed.csv'
    test_path = data_path + 'affwild2_test_emotions_whisper_filtered_complete_fixed.csv'

    print(f"Loading train dataset from: {train_path}")
    train_dataset = affwild2_dataset(preprocessing(train_path))
    print(f"Loading dev dataset from: {dev_path}")
    dev_dataset = affwild2_dataset(preprocessing(dev_path))
    print(f"Loading test dataset from: {test_path}")
    test_dataset = affwild2_dataset(preprocessing(test_path))

    print(f"Creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=0, collate_fn=make_batchs_affwild)
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0, collate_fn=make_batchs_affwild)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0, collate_fn=make_batchs_affwild)

    save_video = os.path.join('./AffWild2/save_model', "student_video")
    
    print("###Save Path### ", save_video)
    if not os.path.exists(save_video):
        os.makedirs(save_video)
  
    clsNum = len(train_dataset.emoList)
    init_config = Config()

    '''teacher model load'''
    teacher_path = args.teacher_path
    if teacher_path is None:
        teacher_path = './AffWild2/save_model/teacher_affwild2_whisper_epoch_8.bin'
    print(f"Loading teacher model from: {teacher_path}")
    model_t = Teacher_model(text_model, clsNum)
    model_t.load_state_dict(torch.load(teacher_path))
    for para in model_t.parameters():
        para.requires_grad = False
    model_t = model_t.cuda()
    model_t.eval()
    print("Teacher model loaded successfully!")

    '''student model'''
    video_s = Student_Video(video_model, clsNum)
    video_s = video_s.cuda()

    """Training Setting"""        
    training_epochs = args.epochs
    save_term = int(training_epochs/5)
    max_grad_norm = 10
    lr = args.learning_rate
    num_training_steps = len(train_dataset) * training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer_video = torch.optim.AdamW(video_s.parameters(), lr=lr)
    scheduler_video = get_linear_schedule_with_warmup(optimizer_video, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler()

    print("Starting video student training...")
    best_video_epoch, best_video_fscore = model_train(training_epochs, model_t, video_s, train_loader, dev_loader, test_loader, optimizer_video, scheduler_video, max_grad_norm, scaler, save_video)
    
    print("---------------Done--------------")
    print(f"Training completed!")
    print(f"Video student: Best epoch {best_video_epoch + 1}, Dev F1: {best_video_fscore:.4f}")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args) 