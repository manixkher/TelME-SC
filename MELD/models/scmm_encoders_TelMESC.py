import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GlobalPathEncoderTelMESC(nn.Module):
    """ Global Path Encoder TelMESC.

    Args:
        nn (nn.Module): nn module

    Returns:
        GlobalPathEncoderTelMESC: Global Path Encoder TelMESC
    """
    def __init__(self, hidden_size=768, num_heads=8, ff_dim=2048, dropout=0.1, max_context=12):
        super(GlobalPathEncoderTelMESC, self).__init__()
        
        self.hidden_size = hidden_size
        self.max_context = max_context
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
    def forward(self, sequence, padding_mask=None):
        """ Forward pass.

        Args:
            sequence (torch.Tensor): Sequence
            padding_mask (torch.Tensor, optional): Padding mask. Defaults to None.
        """
        transformer_output = self.transformer(
            sequence,
            src_key_padding_mask=padding_mask
        )
        
        if padding_mask is not None:
            last_positions = (~padding_mask).sum(dim=1) - 1
            output = transformer_output[torch.arange(transformer_output.size(0)), last_positions]
        else:
            output = transformer_output[:, -1]
            
        return output


class LocalPathEncoderTelMESC(nn.Module):
    """ Local Path Encoder TelMESC.

    Args:
        nn (nn.Module): nn module

    Returns:
        LocalPathEncoderTelMESC: Local Path Encoder TelMESC
    """
    def __init__(self, hidden_size=768, gru_hidden=384, dropout=0.1, max_context=12, local_window=5):
        super(LocalPathEncoderTelMESC, self).__init__()
        
        self.hidden_size = hidden_size
        self.gru_hidden = gru_hidden
        self.max_context = max_context
        self.local_window = local_window
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, sequence, lengths=None):
        """ Forward pass.

        Args:
            sequence (torch.Tensor): Sequence
            lengths (torch.Tensor, optional): Lengths. Defaults to None.
        """
        batch_size, seq_len, _ = sequence.shape
        
        # Apply local window masking - only keep the last 5 utterances
        if seq_len > self.local_window:
            sequence = sequence[:, -self.local_window:, :]
            seq_len = self.local_window
            
            if lengths is not None:
                lengths = torch.clamp(lengths, max=self.local_window)
        
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed_sequence = pack_padded_sequence(
                sequence, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            gru_output, hidden = self.gru(packed_sequence)
        else:
            gru_output, hidden = self.gru(sequence)
        
        # Process hidden states
        num_dirs = 2 if self.gru.bidirectional else 1
        if hidden.shape[0] == num_dirs:
            if num_dirs == 2:
                fw, bw = hidden[0], hidden[1]
                ctx = torch.cat([fw, bw], dim=-1)
            else:
                ctx = hidden.squeeze(0)
        else:
            fw = hidden[0]
            zero_bw = torch.zeros_like(fw)
            ctx = torch.cat([fw, zero_bw], dim=-1)
        
        ctx = self.dropout(ctx)
        ctx = self.layer_norm(ctx)
        return ctx


class DirectPathEncoderTelMESC(nn.Module):
    """ Direct Path Encoder TelMESC.

    Args:
        nn (nn.Module): nn module

    Returns:
        DirectPathEncoderTelMESC: Direct Path Encoder TelMESC
    """
    def __init__(self, hidden_size=768, dropout=0.1):
        super(DirectPathEncoderTelMESC, self).__init__()
        
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self._initialized = False
        
    def _initialize_weights(self, device):
        if not self._initialized:
            with torch.no_grad():
                self.linear.weight.copy_(
                    torch.eye(self.hidden_size, device=device) + 0.01 * torch.randn(self.hidden_size, self.hidden_size, device=device)
                )
                self.linear.bias.zero_()
            self._initialized = True
        
    def forward(self, current_utterance):
        """ Forward pass.

        Args:
            current_utterance (torch.Tensor): Current utterance
        """
        self._initialize_weights(current_utterance.device)
        
        output = self.linear(current_utterance)
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        return output 