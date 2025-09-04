import torch
import torch.nn as nn
import torch.nn.functional as F


class SCMMGateTelMESC(nn.Module):
    """ SCMM Gate TelMESC.

    Args:
        nn (nn.Module): nn module

    Returns:
        SCMMGateTelMESC: SCMM Gate TelMESC
    """
    def __init__(self, hidden_size=768, num_routes=3):
        super(SCMMGateTelMESC, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_routes = num_routes
        
        # Single learnable query vector q_p
        self.query_vector = nn.Parameter(torch.randn(hidden_size))
        

        nn.init.normal_(self.query_vector, mean=0.0, std=0.02)
        
    def forward(self, path_outputs, gate_mode="learned"):
        """ Forward pass.

        Args:
            path_outputs (torch.Tensor): Path outputs
            gate_mode (str, optional): Gate mode. Defaults to "learned".
        """
        if gate_mode == "uniform":
            batch_size = path_outputs.size(0)
            gate_decisions = torch.ones(batch_size, self.num_routes, device=path_outputs.device) / self.num_routes
            gate_logits = torch.zeros(batch_size, self.num_routes, device=path_outputs.device)
        else:
            scores = torch.matmul(path_outputs, self.query_vector) / (self.hidden_size ** 0.5)
            
            gate_logits = scores
            gate_decisions = F.softmax(scores, dim=-1)
        
        return gate_decisions, gate_logits 