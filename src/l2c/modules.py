import torch
import torch.nn as nn

class Sample_Categorical(nn.Module):
    def __init__(self, tau):
        super(Sample_Categorical, self).__init__()
        self.tau = tau
  
    def forward(self, logits, smoothing=None):
        logits = logits.squeeze(-1)
        c = logits.size(-1)
        
        if self.training:
            if smoothing:
                pr = torch.softmax(logits, -1)
                noise = torch.ones_like(pr) / c
                pr = torch.mul(1-smoothing, pr) + torch.mul(smoothing, noise)
                logits = torch.log(pr.clamp(min=1e-12))
            sample = nn.functional.gumbel_softmax(logits, hard=False, tau=self.tau, dim=-1)
        else:
            choice = torch.distributions.categorical.Categorical(logits=logits)
            s = choice.sample()
            sample = torch.eye(c, device=logits.device)[s]
        return sample

class Sample_Bernoulli(nn.Module):
    def __init__(self, tau):
        super(Sample_Bernoulli, self).__init__()
        self.tau = tau

    def forward(self, probs):
        if self.training:
            eps = 1e-7
            probs_clamped = probs.clamp(eps, 1.0 - eps)
            logits = torch.log(probs_clamped) - torch.log(1.0 - probs_clamped)
            u1 = torch.rand(probs.shape, device=probs.device).clamp(eps, 1.0 - eps)
            u2 = torch.rand(probs.shape, device=probs.device).clamp(eps, 1.0 - eps)
            g1 = -torch.log(-torch.log(u1))
            g2 = -torch.log(-torch.log(u2))
            sample = torch.sigmoid((logits + g1 - g2) / self.tau)
        else:
            sample = torch.bernoulli(probs)
        return sample

class Perturber(nn.Module):
    def __init__(self, D, HP):
        super(Perturber, self).__init__()
        self.perturber = nn.Sequential(
            nn.Linear(D, HP), 
            nn.ReLU(),
            nn.Linear(HP, HP),
            nn.ReLU(), 
            nn.Linear(HP, HP),
            nn.ReLU(), 
            nn.Linear(HP, D)
        )
  
    def forward(self, x):
        return self.perturber(x)
        
class Model(nn.Module):
    def __init__(self, D, HP, HS, metadata, tau=0.7):
        super(Model, self).__init__()
        
        self.metadata = metadata
        self.N = metadata['num_features']
        self.C = Sample_Categorical(tau)
        self.B = Sample_Bernoulli(tau)
        
        self.selector = nn.Sequential(
            nn.Linear(D, HS), 
            nn.ReLU(),
            nn.Linear(HS, self.N),
            nn.Sigmoid()          
        )
        self.perturber = Perturber(D, HP)
    
    def search_non_increasing_samples(self, parent_index, x, truth_x):
        """Finds instances where the 'cause' feature did NOT increase."""
        start, end = self.metadata['feature_indices'][parent_index]
        
        if self.metadata['feature_types'][parent_index] == 'cont':
            orig_parent = truth_x[:, start:end]
            new_parent = self.output[parent_index]
        else:
            orig_parent = x[:, start:end].max(dim=-1).indices.float().unsqueeze(-1)
            new_parent = self.output[parent_index].max(dim=-1).indices.float().unsqueeze(-1)

        # Mask where parent remained constant or decreased
        constant_mask = (new_parent <= orig_parent).flatten()
        constant_samples = torch.where(constant_mask)[0].tolist()
        samples = torch.where(~constant_mask)[0].tolist()
        
        return constant_samples, samples

    def perturb(self, col_index, x, truth_x, W, probs):
        start, end = self.metadata['feature_indices'][col_index]
        p_i = probs[:, col_index : col_index+1].clone()
        input_i = x[:, start:end]
        
        # 1. Check if this feature is the "effect" of a cause
        parent_index = None
        force_increase = False
        for (cause, effect, delta) in self.metadata['correlated_features']:
            if col_index == effect:
                parent_index = cause
                force_increase = (delta > 0) # If delta > 0, we assume native L2C forcing increase
                break

        # 2. Modify Probability Mask (p_i) based on Parent State
        if parent_index is not None:
            if self.output[parent_index] is None:
                self.perturb(parent_index, x, truth_x, W, probs)
            constant_samples, samples = self.search_non_increasing_samples(parent_index, x, truth_x)
            
            # If parent didn't increase, child MUST NOT change
            p_i_masks = torch.ones_like(p_i, device=x.device)
            p_i_masks[constant_samples, :] = 0.0
            p_i = torch.mul(p_i, p_i_masks)

            # If parent increased, child MUST change
            p_i_increase = torch.zeros_like(p_i, device=x.device)
            p_i_increase[samples, :] = 1.0
            p_i = p_i + p_i_increase
            p_i = torch.clamp(p_i, 0.0, 1.0)
        else:
            samples = list(range(truth_x.size(0)))
                
        # 3. Apply perturbations (Categorical vs Continuous)
        if self.metadata['feature_types'][col_index] == 'cat':
            logits = W[:, start:end]
            curr_states = input_i.max(dim=-1).indices
            
            # Monotonicity: Mask lower logits to strictly prevent decreasing
            if col_index in self.metadata['monotonic_indices'] or force_increase:
                logit_masks = torch.zeros_like(logits, device=x.device)
                for batch_index in samples:
                    curr = curr_states[batch_index].item()
                    # If forcing increase, mask out current and lower states
                    mask_limit = curr + 1 if force_increase else curr
                    logit_masks[batch_index, :mask_limit] = -100
                logits = logits + logit_masks
                
            x_tilde = self.C(logits, self.smoothing)
            
        else:
            # Continuous feature: W outputs raw values. 
            # Apply Sigmoid to bound the predictions to [0, 1] (MinMaxScaler bounds),
            # removing the discretized matrix multiplication from the original paper.
            x_tilde = torch.sigmoid(W[:, start:end])
            
            # Continuous Monotonicity: clamp new value to be >= original value
            if col_index in self.metadata['monotonic_indices'] or force_increase:
                # We only enforce this clamp for the samples where it's supposed to increase
                orig_vals = truth_x[:, start:end]
                clamped_tilde = torch.max(x_tilde, orig_vals)
                
                # Apply clamp selectively
                mask_tensor = torch.zeros_like(x_tilde)
                mask_tensor[samples, :] = 1.0
                x_tilde = (mask_tensor * clamped_tilde) + ((1 - mask_tensor) * x_tilde)

        # 4. Final Interpolation (Change vs Keep)
        x_i = truth_x[:, start:end] if self.metadata['feature_types'][col_index] == 'cont' else input_i
        x_tilde = torch.mul(p_i, x_tilde) + torch.mul((1-p_i), x_i) 
        self.output[col_index] = x_tilde

    def forward(self, x, truth_x, **kwargs): 
        self.smoothing = kwargs.get('smoothing', 1.0)
        mask_locations = kwargs.get('mask_locations', [])
        L = kwargs.get('L', 1)

        if L > 1: 
            x = torch.repeat_interleave(x, L, dim=0)
            truth_x = torch.repeat_interleave(truth_x, L, dim=0)
        
        P = self.selector(x)
        probs = self.B(P)   
        
        # Apply immutable mask
        if bool(mask_locations):
            mask = torch.ones_like(probs, device=x.device)
            for loc in mask_locations:
                mask[:, loc] = 0.0
            probs = torch.mul(probs, mask)
        
        W = self.perturber(x)
        self.output = {i: None for i in range(self.N)}
        
        # Note: Features must be topologically sorted so 'cause' is processed before 'effect'
        for i in range(self.N):
            if self.output[i] is None:
                self.perturb(i, x, truth_x, W, probs)
                
        output = list(self.output.values())
        xcf = torch.cat(output, dim=1)

        if self.training:
            return truth_x, xcf, P, W
        else:
            return truth_x, xcf, probs, W
