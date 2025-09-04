import torch
import torch.nn as nn
import torch.nn.functional as F

from MELD.models_new.scmm_encoders_TelMESC import GlobalPathEncoderTelMESC, LocalPathEncoderTelMESC, DirectPathEncoderTelMESC
from MELD.models_new.scmm_gate_TelMESC import SCMMGateTelMESC




class SCMMContextTelMESC(nn.Module):
    """
    SCMM Context TelMESC.

    Args:
        nn (nn.Module): nn module

    Returns:
        SCMMContextTelMESC: SCMM Context TelMESC
    """
    def __init__(self, hidden_size=768, max_context_size=12, dropout=0.1, path_dropout=0.0, total_epochs=10):
        """
        Initialize SCMM Context TelMESC.

        Args:
            hidden_size (int, optional): Hidden size. Defaults to 768.
            max_context_size (int, optional): Maximum context size. Defaults to 12.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            path_dropout (float, optional): Path dropout rate. Defaults to 0.0.
            total_epochs (int, optional): Total epochs. Defaults to 10.
        """
        super(SCMMContextTelMESC, self).__init__()
        
        self.hidden_size = hidden_size
        self.max_context_size = max_context_size
        self.path_dropout = path_dropout
        self.total_epochs = total_epochs
        self.current_epoch = 0  
        self.context_sizes = [1, 5, 12]  # Direct, Local, Global
        self.num_routes = len(self.context_sizes)
        
        # Three separate encoders
        self.global_encoder = GlobalPathEncoderTelMESC(
            hidden_size=hidden_size,
            num_heads=8,
            ff_dim=2048,
            dropout=dropout,
            max_context=12
        )
        
        self.direct_encoder = DirectPathEncoderTelMESC(
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # Gating network with learnable query vector
        self.gate = SCMMGateTelMESC(
            hidden_size=hidden_size,
            num_routes=self.num_routes
        )
        
        # Rolling buffers for context
        self.register_buffer('context_buffer', torch.zeros(1, max_context_size, hidden_size))
        self.register_buffer('buffer_mask', torch.zeros(1, max_context_size, dtype=torch.bool))
        self.register_buffer('buffer_indices', torch.zeros(1, dtype=torch.long))
        
    def update_context_buffer(self, z_k, batch_size):
        """ Adds to buffer, if full, oldest utterance is replaced.

        Args:
            z_k (torch.Tensor): z_k
            batch_size (int): Batch size
        """
        z_k_expanded = z_k.unsqueeze(1)
        
        if self.context_buffer.size(0) != batch_size:
            self.context_buffer = torch.zeros(batch_size, self.max_context_size, self.hidden_size, 
                                             device=self.context_buffer.device)
            self.buffer_mask = torch.zeros(batch_size, self.max_context_size, dtype=torch.bool,
                                          device=self.buffer_mask.device)
            self.buffer_indices = torch.zeros(batch_size, dtype=torch.long, device=self.buffer_indices.device)
        
        for b in range(batch_size):
            idx = self.buffer_indices[b]
            self.buffer_mask[b, idx] = False
            self.context_buffer[b, idx] = z_k_expanded[b, 0]
            self.buffer_mask[b, idx] = True
            self.buffer_indices[b] = (idx + 1) % self.max_context_size
    
    def get_context_sequences(self, batch_size, context_size):
        """ Gets context sequences from buffer.

        Args:
            batch_size (int): Batch size
            context_size (int): Context size
        """
        current_buffer = self.context_buffer[:batch_size]
        current_mask = self.buffer_mask[:batch_size]
        
        sequences = []
        lengths = []
        
        for b in range(batch_size):
            # Get non-zero positions in buffer mask
            real_positions = current_mask[b].nonzero(as_tuple=True)[0]
            
            if len(real_positions) == 0:
                sequences.append(torch.zeros(context_size, self.hidden_size, device=current_buffer.device))
                lengths.append(0)
            else:
                current_idx = self.buffer_indices[b]
                # sort positions in chronological order
                def chronological_order(pos):
                    age = (current_idx - pos) % self.max_context_size
                    return age
                
                sorted_positions = sorted(real_positions, key=chronological_order, reverse=True)
                
                selected_positions = sorted_positions[:context_size]
                
                # If not enough positions, pad with zeros
                if len(selected_positions) < context_size:
                    sequence = current_buffer[b, selected_positions]
                    padding = torch.zeros(context_size - len(selected_positions), self.hidden_size, 
                                        device=current_buffer.device)
                    sequence = torch.cat([sequence, padding], dim=0)
                    lengths.append(len(selected_positions))
                else:
                    sequence = current_buffer[b, selected_positions]
                    lengths.append(context_size)
                
                sequences.append(sequence)
        
        sequences = torch.stack(sequences, dim=0)
        lengths = torch.tensor(lengths, device=current_buffer.device)
        
        padding_mask = (torch.arange(context_size, device=current_buffer.device)
                       .expand(batch_size, context_size)
                       >= lengths.unsqueeze(1))
        
        return sequences, lengths, padding_mask
    
    def get_context_sequences_with_current(self, batch_size, context_size, z_k):
        """ Gets context sequences with current utterance.

        Args:
            batch_size (int): Batch size
            context_size (int): Context size
            z_k (torch.Tensor): z_k

        Returns:
            tuple: Sequences with current utterance, lengths with current utterance, padding mask
        """
        # get old context sequences
        historical_sequences, historical_lengths, historical_padding_mask = self.get_context_sequences(batch_size, context_size - 1)
        
        sequences_with_current = []
        lengths_with_current = []
        
        for b in range(batch_size):
            hist_seq = historical_sequences[b]  # [context_size-1, hidden_size]
            hist_len = historical_lengths[b]
            
            new_seq = torch.zeros(context_size, self.hidden_size, device=z_k.device)
            
            if hist_len > 0:
                copy_len = min(hist_len, context_size - 1)
                new_seq[:copy_len] = hist_seq[:copy_len]
            
            new_seq[context_size - 1] = z_k[b]  # Current utterance
            
            sequences_with_current.append(new_seq)
            lengths_with_current.append(min(hist_len + 1, context_size))  # +1 for current utterance
        
        # stack sequences and lengths
        sequences_with_current = torch.stack(sequences_with_current, dim=0)
        lengths_with_current = torch.tensor(lengths_with_current, device=z_k.device)
        
        # create padding mask
        padding_mask = (torch.arange(context_size, device=z_k.device)
                       .expand(batch_size, context_size)
                       >= lengths_with_current.unsqueeze(1))
        
        return sequences_with_current, lengths_with_current, padding_mask

    def ensure_buffer_size(self, batch_size):
        """ Ensures buffer size.

        Args:
            batch_size (int): Batch size
        """
        if self.context_buffer.size(0) < batch_size:
            new_context_buffer = torch.zeros(batch_size, self.max_context_size, self.hidden_size, 
                                           device=self.context_buffer.device)
            new_buffer_mask = torch.zeros(batch_size, self.max_context_size, dtype=torch.bool,
                                        device=self.buffer_mask.device)
            new_buffer_indices = torch.zeros(batch_size, dtype=torch.long, device=self.buffer_indices.device)
            
            current_buffer_size = self.context_buffer.size(0)
            if current_buffer_size > 0:
                # copy old buffer to new buffer
                new_context_buffer[:current_buffer_size] = self.context_buffer
                new_buffer_mask[:current_buffer_size] = self.buffer_mask
                new_buffer_indices[:current_buffer_size] = self.buffer_indices
            
            self.context_buffer = new_context_buffer
            self.buffer_mask = new_buffer_mask
            self.buffer_indices = new_buffer_indices

    def forward(self, z_k, return_gate_info=False, gate_mode="learned"):
        """ Forward pass.

        Args:
            z_k (torch.Tensor): z_k
            return_gate_info (bool, optional): Return gate info. Defaults to False.
            gate_mode (str, optional): Gate mode. Defaults to "learned".

        Returns:
            torch.Tensor: h_k
        """
        batch_size = z_k.size(0)
        
        self.ensure_buffer_size(batch_size)
        
        global_sequences, global_lengths, global_padding_mask = self.get_context_sequences_with_current(batch_size, 12, z_k)
        
        x_g = self.global_encoder(global_sequences, global_padding_mask)
        
        direct_input = z_k
        x_d = self.direct_encoder(direct_input)
        
        local_sequences, local_lengths, local_padding_mask = self.get_context_sequences_with_current(batch_size, 5, z_k)
        x_l = self.local_encoder(local_sequences, local_lengths)
        path_outputs = torch.stack([x_d, x_l, x_g], dim=1)

        
        # Apply path dropout during training with annealing
        if self.training and self.path_dropout > 0.0:
            epoch_frac = self.current_epoch / self.total_epochs
            p_drop = self.path_dropout * (1 - epoch_frac)
            if torch.rand(1) < p_drop:
                # Randomly zero out one entire path across the batch
                i = torch.randint(0, self.num_routes, (1,)).item()
                path_outputs[:, i] = 0.0
        
        gate_decisions, gate_logits = self.gate(path_outputs, gate_mode)
        
        h_k = torch.sum(gate_decisions.unsqueeze(-1) * path_outputs, dim=1)
        
        z_k_detached = z_k.detach()
        self.update_context_buffer(z_k_detached, batch_size)
        
        if return_gate_info:
            gate_info = {
                'gate_decisions': gate_decisions,
                'gate_logits': gate_logits,
                'path_outputs': path_outputs,
                'context_sizes': self.context_sizes,
                'global_lengths': global_lengths,
            }
            gate_info['local_lengths'] = local_lengths
            return h_k, gate_info
        
        return h_k