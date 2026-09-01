import math
import torch
import torch.nn.functional as F

_LN2 = math.log(2.0)

def calculate_jsd(p_final, p_inter):
    """Equation 2: JSD in bits (paper g=0.5 is on [0, 1])."""
    p_final = p_final.to(torch.float32)
    p_inter = p_inter.to(torch.float32)
    m = 0.5 * (p_final + p_inter)
    log_m = m.clamp(min=1e-10).log()
    kl = lambda dist: F.kl_div(log_m, dist, reduction="none", log_target=False).sum(dim=-1).mean()
    return 0.5 * (kl(p_final) + kl(p_inter)) / _LN2

def get_settling_depth(all_hidden_states, lm_head, final_norm, threshold=0.5):
    """
    Finds the first layer where the prediction stabilizes[cite: 113, 155].
    """
    L = len(all_hidden_states) - 1 # Layers are 0-indexed in hidden_states
    
    # Final layer distribution is our 'gold standard' for this token [cite: 108]
    final_h = final_norm(all_hidden_states[-1][:, -1, :])
    p_final = F.softmax(lm_head(final_h), dim=-1)
    
    for l in range(L + 1):
        # Project intermediate hidden state to vocabulary space (Logit Lens) [cite: 105, 106]
        inter_h = final_norm(all_hidden_states[l][:, -1, :])
        p_inter = F.softmax(lm_head(inter_h), dim=-1)
        
        jsd = calculate_jsd(p_final, p_inter)
        if jsd.item() <= threshold: # Threshold g [cite: 113, 115]
            return l + 1 # Return 1-based layer index
            
    return L + 1