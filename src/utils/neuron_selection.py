import torch
import numpy as np

def select_by_threshold(counts, threshold_pct):
    """
    Algorithm 1: Simple Frequency Threshold
    Math: C >= Quantile(C, tau)
    """
    flat = counts.float().view(-1)
    active = flat[flat > 0]
    if len(active) == 0: return set()
    thresh = torch.quantile(active, threshold_pct)
    return set(torch.where(flat >= thresh)[0].tolist())
import torch
import numpy as np

def select_by_entropy(all_activations, n_list, lang_names, top_rate=0.02, filter_rate=0.95):
    """
    Triple-Thresholding LAPE yang dioptimalkan untuk A100 & Data Besar.
    Menggunakan NumPy sebagai fallback untuk menghitung threshold global.
    """
    num_layers, inter_size, num_langs = all_activations.shape
    all_activations = all_activations.float()
    
    # n_list to tensor for broadcasting [1, 1, num_langs]
    n_tensor = torch.tensor(n_list, device=all_activations.device).view(1, 1, -1)
    
    # --- STAGE 1: Calculate Activation Probabilities (P = over_zero / n) ---
    activation_probs = all_activations / n_tensor
    print(f"Menghitung threshold global dari {activation_probs.numel()} elemen...")
    
    probs_cpu = activation_probs.cpu()
    active_values = probs_cpu[probs_cpu > 0].numpy()

    if active_values.size > 0:
        prob_thresh = np.percentile(active_values, filter_rate * 100)
    else:
        prob_thresh = 0.0
        
    print(f"Global Activation Threshold ({filter_rate*100}th): {prob_thresh:.8f}")
    prob_thresh_tensor = torch.tensor(prob_thresh, device=all_activations.device)

    # --- STAGE 2: Entropy Calculation ---
    # Normalisasi untuk Shannon Entropy: p_lang = P / sum(P)
    normed_probs = activation_probs / (activation_probs.sum(dim=-1, keepdim=True) + 1e-9)
    normed_probs = torch.nan_to_num(normed_probs)
    
    log_probs = torch.where(normed_probs > 0, normed_probs.log(), torch.tensor(0.0, device=normed_probs.device))
    entropy = -torch.sum(normed_probs * log_probs, dim=-1) # [Layers, Inter]

    # Filter out quiet neurons (Entropy set to infinity)
    # Neuron yang tidak pernah mencapai threshold di bahasa manapun akan dibuang
    max_p_per_neuron = activation_probs.max(dim=-1).values
    entropy[max_p_per_neuron < prob_thresh_tensor] = torch.inf

    # --- STAGE 3: Selection (Top-K Entropy) ---
    flat_entropy = entropy.flatten()
    top_k = int(len(flat_entropy) * top_rate)
    
    # Mengambil neuron dengan entropi terkecil
    _, top_indices = flat_entropy.topk(top_k, largest=False)

    spec_map = {lang: set() for lang in lang_names}
    
    # Final Mapping menggunakan logika Activation Bar (Stage 3 Paper)
    # Neuron dimasukkan ke mask bahasa jika densitasnya >= threshold
    for idx in top_indices:
        l_idx = idx // inter_size
        n_idx = idx % inter_size
        
        # Cek setiap bahasa: apakah densitasnya di atas threshold?
        for lang_i, lang_name in enumerate(lang_names):
            if activation_probs[l_idx, n_idx, lang_i] >= prob_thresh_tensor:
                spec_map[lang_name].add(idx.item())

    return spec_map