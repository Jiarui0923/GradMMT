import torch
from torch import nn

class BertLastPooler(nn.Module):
    """ Policy for pooling the last (EOS) hidden states of a model into a single vector."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, targetind) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the LAST token.
        ele = torch.arange(0, hidden_states.shape[0])

        first_token_tensor = hidden_states[ele.long(), targetind.long()]#.gather(1, targetind.view(-1,1))#hidden_states[:, -1]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output