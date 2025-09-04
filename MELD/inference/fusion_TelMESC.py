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

from MELD.data.preprocessing import *
from MELD.utils.utils import *
from MELD.data.dataset import *
from MELD.models.model import *
from MELD.models_new.TelMESC import ASF_SCMMTelMESC


def parse_args():
    """Parse arguments.

    Returns:
        argparse.Namespace: Arguments
    """
    parser = argparse.ArgumentParser(description='TelMESC Fusion Training')
    parser.add_argument('--epochs', default=10, type=int, help='epoch for training.')
    parser.add_argument('--asf_lr', default=1e-6, type=float, help='learning rate for ASF fine-tuning.')
    parser.add_argument('--scmm_lr', default=1e-5, type=float, help='learning rate for SCMM training.')
    parser.add_argument('--batch_size', default=4, type=int, help='batch for training.')
    parser.add_argument('--seed', default=42, type=int, help='random seed fix')
    parser.add_argument('--scmm_max_context', default=12, type=int, help='maximum context size for SCMM')
    parser.add_argument('--scmm_dropout', default=0.1, type=float, help='dropout for SCMM components')
    parser.add_argument('--scmm_path_dropout', default=0.0, type=float, help='path dropout probability for SCMM (0.0-0.3)')
    parser.add_argument('--scmm_total_epochs', default=10, type=int, help='total epochs for path dropout annealing')
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

def CELoss(pred_outs, labels):
    """Cross entropy loss.

    Args:
        pred_outs (torch.Tensor): Predicted outputs
        labels (torch.Tensor): Labels
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

def model_train(training_epochs, model_t, audio_s, video_s, fusion, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, scaler, save_path):
    """Model train.

    Args:
        training_epochs (int): Training epochs
        model_t (nn.Module): Teacher model
        audio_s (nn.Module): Audio model
        video_s (nn.Module): Video model
        fusion (nn.Module): Fusion model
        train_dataloader (DataLoader): Train dataloader
        dev_dataloader (DataLoader): Dev dataloader
        test_dataloader (DataLoader): Test dataloader
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler.LambdaLR): Scheduler
        max_grad_norm (float): Max gradient norm
        scaler (torch.cuda.amp.GradScaler): Scaler
        save_path (str): Save path
    """
    best_dev_fscore, best_test_fscore = 0, 0   
    best_epoch = 0

    for epoch in tqdm(range(training_epochs)):
        fusion.train() 
        

        fusion.update_epoch(epoch)
        
        epoch_loss = 0
        num_batches = 0
        

        asf_lr = optimizer.param_groups[0]['lr']
        scmm_lr = optimizer.param_groups[1]['lr']
        print(f"Epoch {epoch+1}, ASF learning rate: {asf_lr:.2e}, SCMM learning rate: {scmm_lr:.2e}")
        
        for i_batch, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
                batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), batch_labels.cuda()
                
                text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
                audio_hidden, audio_logits = audio_s(audio_inputs)
                video_hidden, video_logits = video_s(video_inputs)
                
                if (i_batch % 200 == 0) or (i_batch == 0):
                    fusion_output = fusion(text_hidden, audio_hidden, video_hidden, return_scmm_info=True)
                    pred_logits, _, gate_info = fusion_output
                else:
                    fusion_output = fusion(text_hidden, audio_hidden, video_hidden, return_scmm_info=False)
                    pred_logits, _ = fusion_output
                loss_val = CELoss(pred_logits, batch_labels)

            scaler.scale(loss_val).backward()
            
            if (i_batch % 200 == 0) or (i_batch == 0):
                print(f"\nEpoch {epoch+1}, Batch {i_batch}")
                
                pred_labels = pred_logits.argmax(1).detach().cpu().numpy()
                true_labels = batch_labels.detach().cpu().numpy()
                print(f"Sample predictions: {pred_labels}")
                print(f"Sample labels: {true_labels}")
                
                
                for name, param in fusion.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if 'scmm' in name:
                            scmm_grad_norm += grad_norm ** 2
                            scmm_param_count += 1
                        else:
                            asf_grad_norm += grad_norm ** 2
                            asf_param_count += 1
                
                asf_grad_norm = asf_grad_norm ** 0.5 if asf_grad_norm > 0 else 0.0
                scmm_grad_norm = scmm_grad_norm ** 0.5 if scmm_grad_norm > 0 else 0.0
                
                print(f"ASF gradient norm: {asf_grad_norm:.6f} ({asf_param_count} params)")
                print(f"SCMM gradient norm: {scmm_grad_norm:.6f} ({scmm_param_count} params)")
                
                asf_lr = optimizer.param_groups[0]['lr']
                scmm_lr = optimizer.param_groups[1]['lr']
                print(f"Current ASF LR: {asf_lr:.2e}, SCMM LR: {scmm_lr:.2e}")
                
                print(f"Current loss: {loss_val.item():.4f}")
                print()
            
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            epoch_loss += loss_val.item()
            num_batches += 1

        print(f"Average training loss for epoch {epoch+1}: {epoch_loss / num_batches:.4f}")
        
        asf_final_lr = optimizer.param_groups[0]['lr']
        scmm_final_lr = optimizer.param_groups[1]['lr']
        print(f"End of epoch {epoch+1}, Final ASF learning rate: {asf_final_lr:.2e}, Final SCMM learning rate: {scmm_final_lr:.2e}")

        fusion.eval()   
        dev_pred_list, dev_label_list = evaluation(model_t, audio_s, video_s, fusion, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        print(f"Epoch {epoch+1}, dev_score: {dev_fbeta:.4f}")

        test_pred_list, test_label_list = evaluation(model_t, audio_s, video_s, fusion, test_dataloader)
        test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
        print(f"Epoch {epoch+1}, test_score: {test_fbeta:.4f}")

        _SaveModel(fusion, save_path, epoch)
        
        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta
            best_test_fscore = test_fbeta
            best_epoch = epoch
            print(f"New best model at epoch {epoch+1} (dev_fbeta: {dev_fbeta:.4f})")
        
        print(f"Best dev F1-score so far: {best_dev_fscore:.4f} (epoch {best_epoch + 1})")
        print(f"Best test F1-score so far: {best_test_fscore:.4f}")


def evaluation(model_t, audio_s, video_s, fusion, dataloader):
    """Evaluation.

    Args:
        model_t (nn.Module): Teacher model
        audio_s (nn.Module): Audio model
        video_s (nn.Module): Video model
        fusion (nn.Module): Fusion model
        dataloader (DataLoader): Dataloader
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
                
            fusion_output = fusion(text_hidden, audio_hidden, video_hidden)
            
            if isinstance(fusion_output, tuple):
                pred_logits = fusion_output[0]
            else:
                pred_logits = fusion_output
            
            pred_label = pred_logits.argmax(1).detach().cpu().numpy() 
            true_label = batch_labels.detach().cpu().numpy()
            
            pred_list.extend(pred_label)
            label_list.extend(true_label)

    return pred_list, label_list


def _SaveModel(model, path, epoch=None):
    """Save model.

    Args:
        model (nn.Module): Model
        path (str): Path
        epoch (int, optional): Epoch. Defaults to None.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
    if epoch is not None:
        filename = f'TelMESC_dropout0.0_5e-6_epoch_{epoch+1}.bin'
    else:
        filename = 'TelMESC_dropout0.0_5e-6.bin'
    
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

    data_path = './dataset/MELD.Raw/'

    train_path = data_path + 'train_meld_emo.csv'
    dev_path = data_path + 'dev_meld_emo.csv'
    test_path = data_path + 'test_meld_emo.csv'

    train_dataset = meld_dataset(preprocessing(train_path))
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=16, collate_fn=make_batchs)

    dev_dataset = meld_dataset(preprocessing(dev_path))
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs)

    test_dataset = meld_dataset(preprocessing(test_path))
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs)

    save_path = os.path.join('./MELD/save_model')
    
    print("###Save Path### ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    clsNum = len(train_dataset.emoList)
    init_config = Config()

    # Teacher model load
    model_t = Teacher_model(text_model, clsNum)
    model_t.load_state_dict(torch.load('./MELD/save_model/teacher.bin'))
    for para in model_t.parameters():
        para.requires_grad = False
    model_t = model_t.cuda()
    model_t.eval()

    # Student models
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

    # TelMESC fusion
    hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
    fusion = ASF_SCMMTelMESC(
        clsNum, hidden_size, beta_shift, dropout_prob, num_head,
        scmm_hidden_size=768,
        scmm_max_context=args.scmm_max_context,
        scmm_dropout=args.scmm_dropout,
        scmm_path_dropout=args.scmm_path_dropout,
        scmm_total_epochs=args.scmm_total_epochs
    )
    
    if os.path.exists('./MELD/save_model/total_fusion.bin'):
        asf_state_dict = torch.load('./MELD/save_model/total_fusion.bin')
        fusion.load_asf_weights(asf_state_dict)
        print("Loaded pre-trained ASF weights")
    
    fusion = fusion.cuda()
    fusion.train()

    training_epochs = args.epochs
    max_grad_norm = 10
    
    param_groups = fusion.get_parameter_groups(asf_lr=args.asf_lr, scmm_lr=args.scmm_lr)
    print(f"Parameter groups:")
    for i, group in enumerate(param_groups):
        print(f"  Group {i}: {group['name']} (lr={group['lr']:.2e})")
        print(f"    Parameters: {len(group['params'])}")
    
    optimizer = torch.optim.AdamW(param_groups)
    
    num_training_steps = len(train_loader) * training_epochs
    num_warmup_steps = len(train_loader)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    scaler = torch.cuda.amp.GradScaler()

    print(f"Starting TelMESC training with {training_epochs} epochs")
    print(f"ASF learning rate: {args.asf_lr:.2e}")
    print(f"SCMM learning rate: {args.scmm_lr:.2e}")
    print(f"SCMM path dropout: {args.scmm_path_dropout:.3f} (annealing over {args.scmm_total_epochs} epochs)")
    
    model_train(training_epochs, model_t, audio_s, video_s, fusion, train_loader, dev_loader, test_loader, optimizer, scheduler, max_grad_norm, scaler, save_path)
    print("---------------TelMESC Training Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args) 