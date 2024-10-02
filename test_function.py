import numpy as np
import torch

def generate_exponential_decay_sequence(N, lambd):
    """
    input: 
        N -> num of elements
        lambda -> decay rate
    x_i = e^{- \lambda i}
    """
    # generate exponential decay sequence
    sequence = np.exp(-lambd * np.arange(N))
    # sum the elements in sequence to 1
    normalized_sequence = sequence / np.sum(sequence)
    normalized_sequence = torch.Tensor(normalized_sequence)
    return normalized_sequence

Q_seq = torch.Tensor([[0.6843, 0.5677, 0.8486, 0.2289, 0.1225, 0.3201, 0.3244, 0.9536],
                 [0.7465, 0.3891, 0.7197, 0.1271, 0.6409, 0.2238, 0.5569, 0.7439],
                 [0.0678, 0.9707, 0.8825, 0.0082, 0.0159, 0.8627, 0.2404, 0.8099]])

actor_horizon_comp_factor = generate_exponential_decay_sequence(8, 0.25)
expanded_actor_horizon_comp_factor = actor_horizon_comp_factor.expand(Q_seq.shape[0], -1)
weighted_Q_seq = expanded_actor_horizon_comp_factor * Q_seq
Q_seq_mean = weighted_Q_seq.sum(dim=1, keepdim=True) 

print("Normalized sequence:", actor_horizon_comp_factor)
print("Expanded Normalized sequence:", expanded_actor_horizon_comp_factor)
print("Target Q:", Q_seq)
print("Weighted Target Q:", weighted_Q_seq)
print("Weighted sum:", Q_seq_mean)
print("Weighted sum:", Q_seq_mean.mean())