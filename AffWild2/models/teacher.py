import os
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
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
    parser.add_argument('--batch_size', default=16, type=int, help='batch for training.')
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
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

def model_train(training_epochs, model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, save_path):
    best_dev_fscore, best_test_fscore = 0, 0   
    best_epoch = 0

    for epoch in tqdm(range(training_epochs)):
        model.train() 
        epoch_loss = 0.0
        batch_count = 0
        
        for i_batch, data in enumerate(train_dataloader):
            try:
                optimizer.zero_grad()

                """Prediction"""
                batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels = data
                batch_input_tokens, batch_attention_masks, batch_labels = batch_input_tokens.cuda(),batch_attention_masks.cuda(), batch_labels.cuda()
                last_hidden, pred_logits = model(batch_input_tokens, batch_attention_masks)

                loss_val = CELoss(pred_logits, batch_labels)
                
                loss_val.backward()
                

                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss_val.item()
                batch_count += 1
                
                if batch_count % 200 == 0:
                    avg_loss = epoch_loss / batch_count
                    print(f"Epoch {epoch}, Batch {batch_count}: Loss={loss_val.item():.4f}, Avg_Loss={avg_loss:.4f}, Grad_Norm={total_norm:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")
                    
                    grad_norms = []
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            grad_norms.append(grad_norm)
                    
                    if grad_norms:
                        print(f"Gradient stats - Min: {min(grad_norms):.4f}, Max: {max(grad_norms):.4f}, Mean: {np.mean(grad_norms):.4f}")
                    
                    pred_labels = pred_logits.argmax(1).detach().cpu().numpy()
                    true_labels = batch_labels.detach().cpu().numpy()
                    
                    emotion_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
                    
                    print(f"Sample predictions (first 4 in batch):")
                    for i in range(min(4, len(pred_labels))):
                        pred_emotion = emotion_names[pred_labels[i]]
                        true_emotion = emotion_names[true_labels[i]]
                        confidence = torch.softmax(pred_logits[i], dim=0).max().item()
                        print(f"  Sample {i}: Pred={pred_emotion} (conf={confidence:.3f}), True={true_emotion}")
                    
                    pred_counts = np.bincount(pred_labels, minlength=len(emotion_names))
                    true_counts = np.bincount(true_labels, minlength=len(emotion_names))
                    
                    print(f"Batch prediction distribution:")
                    for i, emotion in enumerate(emotion_names):
                        if pred_counts[i] > 0 or true_counts[i] > 0:
                            print(f"  {emotion}: Pred={pred_counts[i]}, True={true_counts[i]}")
                    
                    batch_accuracy = (pred_labels == true_labels).mean()
                    print(f"Batch accuracy: {batch_accuracy:.3f}")
                    
            except RuntimeError as e:
                if "All sessions in this batch failed" in str(e):
                    print(f"Warning: Skipping batch {i_batch} due to all sessions failing")
                    continue
                else:
                    raise e
  
        model.eval()   
        dev_pred_list, dev_label_list = evaluation(model, dev_dataloader)
        
        emotion_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        
        dev_pre_per_class, dev_rec_per_class, dev_f1_per_class, dev_support = precision_recall_fscore_support(
            dev_label_list, dev_pred_list, average=None, labels=range(len(emotion_names))
        )
        
    
        
        dev_accuracy = (np.array(dev_pred_list) == np.array(dev_label_list)).mean()
        
        print(f"=== EPOCH {epoch} DEV RESULTS ===")
        print(f"Overall Weighted F1: {dev_fbeta:.4f}")
        print(f"Overall Accuracy: {dev_accuracy:.4f}")
        print(f"Per-class F1 scores:")
        for i, emotion in enumerate(emotion_names):
            print(f"  {emotion:8s}: F1={dev_f1_per_class[i]:.4f}, Precision={dev_pre_per_class[i]:.4f}, Recall={dev_rec_per_class[i]:.4f}, Support={dev_support[i]}")
        
        
        cm = confusion_matrix(dev_label_list, dev_pred_list, labels=range(len(emotion_names)))
        print(f"Confusion Matrix:")
        print("Pred\\True", end="")
        for emotion in emotion_names:
            print(f"{emotion:>8s}", end="")
        print()
        for i, emotion in enumerate(emotion_names):
            print(f"{emotion:8s}", end="")
            for j in range(len(emotion_names)):
                print(f"{cm[i,j]:>8d}", end="")
            print()
        print("=" * 50)
        
        print(f"dev_score : {dev_fbeta}")

        _SaveModel(model, save_path, f"epoch_{epoch + 1}")

        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta
            best_epoch = epoch
            print(f"New best model found! Epoch {epoch + 1}, Dev F1: {dev_fbeta:.4f}")
            _SaveModel(model, save_path, "best")

            model.eval()
            test_pred_list, test_label_list = evaluation(model, test_dataloader)
            
            test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
            test_accuracy = (np.array(test_pred_list) == np.array(test_label_list)).mean()
            
            test_pre_per_class, test_rec_per_class, test_f1_per_class, test_support = precision_recall_fscore_support(
                test_label_list, test_pred_list, average=None, labels=range(len(emotion_names))
            )
            
            print(f"=== EPOCH {epoch} TEST RESULTS ===")
            print(f"Overall Weighted F1: {test_fbeta:.4f}")
            print(f"Overall Accuracy: {test_accuracy:.4f}")
            print(f"Per-class F1 scores:")
            for i, emotion in enumerate(emotion_names):
                print(f"  {emotion:8s}: F1={test_f1_per_class[i]:.4f}, Precision={test_pre_per_class[i]:.4f}, Recall={test_rec_per_class[i]:.4f}, Support={test_support[i]}")
            print("=" * 50)
            
            print(f"test_score : {test_fbeta}")
    
    return best_epoch, best_dev_fscore

def evaluation(model, dataloader):
    """
    Evaluate the model.

    Args:
        model (torch.nn.Module): Model
        dataloader (torch.utils.data.DataLoader): Data loader
    """
    model.eval()
    label_list = []
    pred_list = []
    
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):
            try:
                """Prediction"""
                batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels = data
                batch_input_tokens, batch_attention_masks, batch_labels = batch_input_tokens.cuda(),batch_attention_masks.cuda(), batch_labels.cuda()

                last_hidden, pred_logits = model(batch_input_tokens, batch_attention_masks)
                
                """Calculation"""  
                
                pred_label = pred_logits.argmax(1).detach().cpu().numpy() 
                true_label = batch_labels.detach().cpu().numpy()
                
                pred_list.extend(pred_label)
                label_list.extend(true_label)
            except RuntimeError as e:
                if "All sessions in this batch failed" in str(e):
                    print(f"Warning: Skipping evaluation batch {i_batch} due to all sessions failing")
                    continue
                else:
                    raise e

    emotion_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    pred_unique, pred_counts = np.unique(pred_list, return_counts=True)
    label_unique, label_counts = np.unique(label_list, return_counts=True)
    
    print(f"Evaluation predictions distribution: {dict(zip(pred_unique, pred_counts))}")
    print(f"Evaluation labels distribution: {dict(zip(label_unique, label_counts))}")
    
    print(f"Sample evaluation predictions (first 10):")
    for i in range(min(10, len(pred_list))):
        pred_emotion = emotion_names[pred_list[i]]
        true_emotion = emotion_names[label_list[i]]
        print(f"  Sample {i}: Pred={pred_emotion}, True={true_emotion}")
    
    accuracy = (np.array(pred_list) == np.array(label_list)).mean()
    print(f"Evaluation accuracy: {accuracy:.3f}")

    return pred_list, label_list


def _SaveModel(model, path, model_type="best"):
    """
    Save the model.

    Args:
        model (torch.nn.Module): Model
        path (str): Path to save the model
        model_type (str, optional): Model type. Defaults to "best".
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
    if model_type == "best":
        filename = 'exp_new_teacher_affwild2_whisper_best.bin'
        print(f"Saving BEST model to: {os.path.join(path, filename)}")
    elif model_type.startswith("epoch_"):
        filename = f'exp_new_teacher_affwild2_whisper_{model_type}.bin'
        print(f"Saving {model_type.upper()} model to: {os.path.join(path, filename)}")
    else:
        filename = f'exp_new_teacher_affwild2_whisper_{model_type}.bin'
        print(f"Saving {model_type.upper()} model to: {os.path.join(path, filename)}")
    
    torch.save(model.state_dict(), os.path.join(path, filename))
    print(f"Model saved successfully!")

def main(args):
    
    seed_everything(args.seed)
    """Dataset Loading"""

    text_model = "roberta-large"

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
    model = Teacher_model(text_model, clsNum)
    model = model.cuda()
    model.eval()

    """Training Setting"""        
    training_epochs = args.epochs
    save_term = int(training_epochs/5)
    max_grad_norm = 1.0
    lr = args.learning_rate
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-06, weight_decay=0.01) # , eps=1e-06, weight_decay=0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    best_epoch, best_dev_fscore = model_train(training_epochs, model, train_loader, dev_loader, test_loader, optimizer, scheduler, max_grad_norm, save_path)
    print("---------------Done--------------")
    print(f"Training completed! Best model was from epoch {best_epoch + 1} with dev F1: {best_dev_fscore:.4f}")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args) 