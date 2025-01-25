from torch.nn import functional as F
from .rollout import RolloutMultiTrack

class TulipMultiTrack(RolloutMultiTrack):
        
    
    def criterion(self, output, groundtruths):
        out_a, out_b, out_e = output
        alpha_input, beta_input, peptide_input = groundtruths
        m_a = F.one_hot(alpha_input, out_a.shape[-1])
        loss_a = (out_a*m_a).sum()
        m_b = F.one_hot(beta_input, out_b.shape[-1])
        loss_b = (out_b*m_b).sum()
        m_e = F.one_hot(peptide_input, out_e.shape[-1])
        loss_e = (out_e*m_e).sum()
        loss = loss_a + loss_b + loss_e
        return loss