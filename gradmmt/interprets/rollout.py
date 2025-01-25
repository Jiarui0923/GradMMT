import torch
from torch import nn
from .multitrack import ModelMultiTrack

class RolloutMultiTrack(ModelMultiTrack):
    
    @torch.no_grad()
    def grad_rollout_bert(self, attentions, gradients, discard_ratio=0.9):

        attentions = torch.stack([i for i in attentions if i.size(-1) == i.size(-2)])
        gradients = torch.stack([i for i in gradients if i.size(-1) == i.size(-2)])
        # constants
        n_layers, batch_size, n_heads, n_tokens, _ = attentions.shape
        I = torch.eye(n_tokens, device=attentions.device)

        attn_fused_layers = (attentions * gradients).mean(axis=2)
        attn_fused_layers[attn_fused_layers < 0] = 0
        flats = attn_fused_layers.view(n_layers, batch_size, -1)
        k = int(flats.size(-1) * discard_ratio)
        _, indices = flats.topk(k, dim=-1, largest=False)
        
        _b_indices = torch.arange(batch_size, device=attentions.device).reshape(1, -1, 1)
        _l_indices = torch.arange(n_layers, device=attentions.device).reshape(-1, 1, 1)

        flats[_l_indices, _b_indices, indices] = 0
            
        a = (attn_fused_layers + 1.0 * I) / 2
        a /= a.sum(dim=-1, keepdim=True)
        import functools
        mask = functools.reduce(torch.bmm, a.flip(0))
        
        # norm_mask = mask / mask.max(-1, True).values
        return mask[:, 0]
    
    def rollouts(self, discard_ratio=0.9):
        _rollouts = []
        for _forward, _backward in zip(self.forwards, self.backwards):
            _rollouts.append(self.grad_rollout_bert(_forward, _backward, discard_ratio))
        return _rollouts
        
