import torch
import torch.nn as nn
import torch.nn.functional as F

from MELD.models_new.scmm_context_TelMESC import SCMMContextTelMESC


class ASF_SCMMTelMESC(nn.Module):
    """ASF_SCMMTelMESC.

    Args:
        nn (nn.Module): nn module

    Returns:
        ASF_SCMMTelMESC: ASF_SCMMTelMESC
    """
    def __init__(self, clsNum, hidden_size, beta_shift, dropout_prob, num_head, 
                 scmm_hidden_size=768, scmm_max_context=12, scmm_dropout=0.1, scmm_path_dropout=0.0, scmm_total_epochs=10):
        super(ASF_SCMMTelMESC, self).__init__()

        self.TEXT_DIM = 768
        self.VISUAL_DIM = 768
        self.ACOUSTIC_DIM = 768
        
        # Original ASF components
        self.multihead_attn = nn.MultiheadAttention(self.VISUAL_DIM + self.ACOUSTIC_DIM, num_head, batch_first=True)
        self.W_hav = nn.Linear(self.VISUAL_DIM + self.ACOUSTIC_DIM + self.TEXT_DIM, self.TEXT_DIM)
        self.W_av = nn.Linear(self.VISUAL_DIM + self.ACOUSTIC_DIM, self.TEXT_DIM)
        self.beta_shift = beta_shift
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.AV_LayerNorm = nn.LayerNorm(self.VISUAL_DIM + self.ACOUSTIC_DIM)
        self.dropout = nn.Dropout(dropout_prob)
        
        # SCMM Context module
        self.scmm = SCMMContextTelMESC(
            hidden_size=scmm_hidden_size,
            max_context_size=scmm_max_context,
            dropout=scmm_dropout,
            path_dropout=scmm_path_dropout,
            total_epochs=scmm_total_epochs
        )
        
        self.W = nn.Linear(self.TEXT_DIM, clsNum)
        
    def forward(self, text_embedding, visual, acoustic, labels=None, return_scmm_info=False, gate_mode="learned"):
        """Forward pass.

        Args:
            text_embedding (torch.Tensor): Text embedding
            visual (torch.Tensor): Visual embedding
            acoustic (torch.Tensor): Acoustic embedding
            labels (torch.Tensor, optional): Labels. Defaults to None.
            return_scmm_info (bool, optional): Return SCMM info. Defaults to False.
            gate_mode (str, optional): Gate mode. Defaults to "learned".

        Returns:
            torch.Tensor: Logits
            torch.Tensor: Total loss
        """
        batch_size = text_embedding.size(0)
        
        # Original ASF fusion logic
        av_concat = torch.cat([visual, acoustic], dim=-1)
        
        av_concat_expanded = av_concat.unsqueeze(1)
        av_attended, _ = self.multihead_attn(av_concat_expanded, av_concat_expanded, av_concat_expanded)
        av_attended = av_attended.squeeze(1)
        
        av_attended = self.AV_LayerNorm(av_attended)
        
        hav_concat = torch.cat([av_attended, text_embedding], dim=-1)
        
        hav = self.W_hav(hav_concat)
        av = self.W_av(av_attended)
        
        z_k = self.LayerNorm(hav + self.beta_shift * av)
        z_k = self.dropout(z_k)
        
        # Apply SCMM Context
        if return_scmm_info:
            h_k, scmm_info = self.scmm(z_k, return_gate_info=True, gate_mode=gate_mode)
        else:
            h_k = self.scmm(z_k, return_gate_info=False, gate_mode=gate_mode)
        
        # Final classification
        logits = self.W(h_k)
        

        total_loss = None
        if labels is not None:
            total_loss = F.cross_entropy(logits, labels)
        
        if return_scmm_info:
            if labels is not None:
                scmm_info['total_loss'] = total_loss
                scmm_info['main_loss'] = total_loss
            return logits, total_loss, scmm_info
        
        return logits, total_loss

    def load_asf_weights(self, asf_state_dict):
        """Load ASF weights.

        Args:
            asf_state_dict (dict): ASF state dictionary
        """
        model_dict = self.state_dict()
        asf_keys = [k for k in asf_state_dict.keys() if k in model_dict and 'scmm' not in k]
        pretrained_dict = {k: asf_state_dict[k] for k in asf_keys}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print(f"Loaded ASF weights for keys: {list(pretrained_dict.keys())}")
    
    def get_parameter_groups(self, asf_lr=1e-6, scmm_lr=1e-4):
        """Get parameter groups.

        Args:
            asf_lr (float, optional): ASF learning rate. Defaults to 1e-6.
            scmm_lr (float, optional): SCMM learning rate. Defaults to 1e-4.

        Returns:
            list: Parameter groups
        """
       
        asf_params = []
        scmm_params = []
        
        for name, param in self.named_parameters():
            if 'scmm' in name:
                scmm_params.append(param)
            else:
                asf_params.append(param)
        
        param_groups = [
            {'params': asf_params, 'lr': asf_lr, 'name': 'asf'},
            {'params': scmm_params, 'lr': scmm_lr, 'name': 'scmm'}
        ]
        
        return param_groups 
    
    def update_epoch(self, epoch):
       
        self.scmm.current_epoch = epoch 