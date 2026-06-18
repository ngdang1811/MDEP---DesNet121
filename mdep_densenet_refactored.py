"""
============================================================================
  MDEP — Microglial-Driven Evidential Pruning
  Single-file Kaggle Notebook version
  
  HOW TO RUN ON KAGGLE:
    1. Create a new Notebook, set Accelerator to GPU (T4 or P100).
    2. Click "Add Data" → search "ISIC 2024" → add the challenge dataset.
    3. Copy-paste this entire file into a single code cell.
    4. Run the cell.
============================================================================
"""

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import os
import shutil
import re
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from PIL import Image
import io
import cv2
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss, f1_score, precision_recall_curve, auc, roc_curve
)
#  SECTION 1 — EDL Core (Evidential Deep Learning foundations)
# ============================================================================

class EvidenceLayer(nn.Module):
    """
    Ensures the output of the network is non-negative evidence (e >= 0).
    Replaces the traditional Softmax layer for EDL.
    """
    def __init__(self, activation='softplus', max_evidence=20.0):
        super(EvidenceLayer, self).__init__()
        self.max_evidence = max_evidence
        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        ev = self.activation(x)
        if self.max_evidence is not None:
            ev = torch.clamp(ev, max=self.max_evidence)
        return ev


def compute_uncertainties(evidence):
    """
    Computes epistemic and aleatoric uncertainties from the evidence.

    Args:
        evidence (torch.Tensor): Output evidence of shape (batch_size, num_classes)

    Returns:
        dict: Contains epistemic (u_e), aleatoric (u_a), alpha, and S.
    """
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    K = evidence.shape[1]

    # Epistemic Uncertainty: u_e = K / S
    u_e = K / S

    # Aleatoric Uncertainty: u_a = sum (alpha_c / S) * (psi(S+1) - psi(alpha_c+1))
    # NOTE: The formula in the original research proposal main (36).pdf incorrectly had a negative sign prefix
    # (u_a = - sum ...). Since psi(S+1) > psi(alpha_c+1), that negative sign would yield negative uncertainty.
    # We omit the negative sign here to ensure mathematical consistency and non-negativity (u_a >= 0).
    digamma_S = torch.digamma(S + 1.0)
    digamma_alpha = torch.digamma(alpha + 1.0)
    u_a_term = (alpha / S) * (digamma_S - digamma_alpha)
    u_a = torch.sum(u_a_term, dim=1, keepdim=True)
    assert torch.all(u_a > -1e-6), f"Sanity check failed: u_a contains negative values {u_a[u_a < -1e-6].tolist()}"
    u_a = torch.clamp(u_a, min=0.0)

    return {
        'epistemic': u_e,
        'aleatoric': u_a,
        'alpha': alpha,
        'S': S,
    }


# ============================================================================
#  SECTION 2 — Loss Functions (Evidential Focal Loss + KL regularization)
# ============================================================================

def kl_divergence(alpha, num_classes):
    """
    KL divergence between a Dirichlet(alpha) and a uniform Dirichlet(1,...,1).
    """
    beta = torch.ones(1, num_classes, dtype=torch.float32, device=alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


class EvidentialFocalLoss(nn.Module):
    def __init__(self, class_counts, total_epochs, warmup_epochs=15, lambda_kl=0.01):
        super().__init__()
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.lambda_kl = lambda_kl
        self.num_classes = len(class_counts)
        
        # Fix CUDA/CPU: Khai báo register_buffer chuẩn
        self.register_buffer('class_weights', torch.ones(self.num_classes, dtype=torch.float32))
        self.register_buffer('kl_asymmetric_weights', torch.tensor([1.0, 5.0], dtype=torch.float32))

    def forward(self, evidence, targets, epoch):
        device = evidence.device
        
        # Ép kiểu an toàn bằng biến cục bộ
        cls_weights = self.class_weights.to(device)
        kl_weights = self.kl_asymmetric_weights.to(device)
        
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        p_hat = alpha / S
        y_onehot_hard = F.one_hot(targets, num_classes=self.num_classes).float()
        
        if epoch < self.warmup_epochs:
            gamma_t = 0.0
        elif epoch < self.warmup_epochs + 3:
            gamma_t = 2.0 * ((epoch - self.warmup_epochs) / 3.0)
        else:
            gamma_t = 2.0
            
        annealing_coef = min(1.0, epoch / 10.0)
        loss_ce = torch.sum(y_onehot_hard * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
        
        p_target = torch.sum(y_onehot_hard * p_hat, dim=1)
        p_target = torch.clamp(p_target, min=1e-5, max=1.0 - 1e-5)
        focal_term = torch.pow(1.0 - p_target, gamma_t)
        
        alpha_tilde = y_onehot_hard + (1.0 - y_onehot_hard) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        
        kl_div = torch.lgamma(S_tilde) - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True) \
                 + torch.sum(torch.lgamma(torch.ones_like(alpha_tilde)), dim=1, keepdim=True) \
                 - torch.lgamma(torch.tensor(self.num_classes, dtype=torch.float32, device=device)) \
                 + torch.sum((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True)
        
        # Nhân với BIẾN CỤC BỘ (kl_weights)
        target_kl_weights = torch.sum(y_onehot_hard * kl_weights, dim=1)
        
        # Giải cứu lớp thiểu số khỏi sức ép của KL Divergence
        kl_mask = torch.where(targets == 1, 0.1, 1.0).to(device)
        kl_div = kl_div.squeeze() * target_kl_weights * kl_mask
        
        # Nhân với BIẾN CỤC BỘ (cls_weights)
        sample_weights = torch.sum(y_onehot_hard * cls_weights, dim=1)
        loss = sample_weights * (focal_term * loss_ce + self.lambda_kl * annealing_coef * kl_div)
        return torch.mean(loss)


# ============================================================================
#  SECTION 3 — MDEP Multi-Agent Sparsity Engine
# ============================================================================

class SmoothedSTE(torch.autograd.Function):
    """
    Sparse-Refined STE (SR-STE) with Local 2:4 Bounds.
    """
    @staticmethod
    def forward(ctx, scores, mask, gamma, weight, lambda_w=2e-4):
        shape = scores.shape
        if scores.numel() % 4 == 0:
            scores_flat = scores.view(-1, 4)
            # Find the 2nd and 3rd largest values in each block
            sorted_scores, _ = torch.sort(scores_flat, dim=-1, descending=True)
            s2 = sorted_scores[:, 1]
            s3 = sorted_scores[:, 2]
            # Local threshold is the midpoint
            tau = ((s2 + s3) / 2.0).unsqueeze(-1) # shape: (N, 1)
            tau = tau.expand_as(scores_flat).reshape(shape)
        else:
            tau = torch.zeros_like(scores)

        ctx.save_for_backward(scores, tau, torch.tensor(gamma), mask, weight, torch.tensor(lambda_w))
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        scores, tau, gamma, mask, weight, lambda_w = ctx.saved_tensors
        gamma_val = gamma.item()
        lambda_w_val = lambda_w.item()
        
        # Luồng Gradient cho S_ij (Giữ nguyên Smoothed STE)
        margin = scores - tau
        sig = torch.sigmoid(margin / gamma_val)
        grad_scores = grad_output * sig * (1.0 - sig) / gamma_val
        
        # BẮT BUỘC TUÂN THỦ: Bộ tối ưu AdamW chỉ được tác động lên các trọng số đang "mở"
        # Trả về None cho grad_weight để tránh Optimizer Hijacking lên các mask=0
        grad_weight = None
        
        # Trả về gradient cho weights thay vì None
        return grad_scores, None, None, grad_weight, None


def generate_2_4_mask(scores):
    """
    Generates an NVIDIA 2:4 structured sparsity mask strictly along the input channel dimension.
    """
    if scores.dim() == 4: # Cho Conv2d (C_out, C_in, K, K)
        C_out, C_in, K1, K2 = scores.shape
        if C_in % 4 != 0:
            return torch.ones_like(scores)
        # Tái định hình để lấy block 4 dọc theo C_in
        scores_reshaped = scores.permute(0, 2, 3, 1).reshape(-1, 4)
        _, indices = torch.topk(scores_reshaped, 2, dim=-1)
        mask_flat = torch.zeros_like(scores_reshaped).scatter_(1, indices, 1.0)
        return mask_flat.view(-1, K1, K2, C_in).permute(0, 3, 1, 2)
    elif scores.dim() == 2: # Cho Linear
        if scores.shape[1] % 4 != 0:
            return torch.ones_like(scores)
        scores_reshaped = scores.view(-1, 4)
        _, indices = torch.topk(scores_reshaped, 2, dim=-1)
        mask_flat = torch.zeros_like(scores_reshaped).scatter_(1, indices, 1.0)
        return mask_flat.view(scores.shape)
    return torch.ones_like(scores)


class MDEPLinear(nn.Linear):
    """Drop-in replacement for nn.Linear with MDEP dynamic sparsity."""
    def __init__(self, in_features, out_features, bias=True):
        super(MDEPLinear, self).__init__(in_features, out_features, bias)
        self.scores = nn.Parameter(torch.abs(self.weight.data).clone().to(torch.float32))
        self.register_buffer('mask', torch.ones_like(self.weight, dtype=torch.float32))
        self.register_buffer('scores_momentum', torch.zeros_like(self.weight, dtype=torch.float32))
        self.gamma = 1.0
        self.warmup = True
        self.freeze_mask = False

    def forward(self, x):
        # Lịch trình Cắt tỉa Sinh học (Biological Progressive Masking)
        epoch = getattr(self, 'current_epoch', 0)
        warmup_epochs = getattr(self, 'warmup_epochs', 15)
        
        if epoch < warmup_epochs:
            alpha = 0.0
        elif epoch <= warmup_epochs + 10:
            alpha = (epoch - warmup_epochs) / 10.0
        else:
            alpha = 1.0

        if alpha == 0.0:
            effective_weight = self.weight
        else:
            if getattr(self, 'freeze_mask', False) and hasattr(self, 'cached_mask'):
                current_mask = self.cached_mask
            else:
                raw_mask = generate_2_4_mask(self.scores)
                self.mask.copy_(raw_mask)
                current_mask = SmoothedSTE.apply(self.scores, self.mask, self.gamma, self.weight, 2e-4)
                self.cached_mask = current_mask.detach()
            
            # W_effective = W * ((1 - alpha)*1 + alpha * Mask)
            effective_weight = self.weight * ((1.0 - alpha) + alpha * current_mask)
            
        return F.linear(x, effective_weight, self.bias)


class MDEPConv2d(nn.Conv2d):
    """Drop-in replacement for nn.Conv2d with MDEP dynamic sparsity."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MDEPConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias
        )
        self.scores = nn.Parameter(torch.abs(self.weight.data).clone().to(torch.float32))
        self.register_buffer('mask', torch.ones_like(self.weight, dtype=torch.float32))
        self.register_buffer('scores_momentum', torch.zeros_like(self.weight, dtype=torch.float32))
        self.gamma = 1.0
        self.warmup = True
        self.freeze_mask = False

    def forward(self, x):
        # Lịch trình Cắt tỉa Sinh học (Biological Progressive Masking)
        epoch = getattr(self, 'current_epoch', 0)
        warmup_epochs = getattr(self, 'warmup_epochs', 15)
        
        if epoch < warmup_epochs:
            alpha = 0.0
        elif epoch <= warmup_epochs + 10:
            alpha = (epoch - warmup_epochs) / 10.0
        else:
            alpha = 1.0

        if alpha == 0.0:
            effective_weight = self.weight
        else:
            if getattr(self, 'freeze_mask', False) and hasattr(self, 'cached_mask'):
                current_mask = self.cached_mask
            else:
                raw_mask = generate_2_4_mask(self.scores)
                self.mask.copy_(raw_mask)
                current_mask = SmoothedSTE.apply(self.scores, self.mask, self.gamma, self.weight, 2e-4)
                self.cached_mask = current_mask.detach()
            
            # W_effective = W * ((1 - alpha)*1 + alpha * Mask)
            effective_weight = self.weight * ((1.0 - alpha) + alpha * current_mask)
            
        return F.conv2d(
            x, effective_weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


def compute_rank(tensor):
    shape = tensor.shape
    flat = tensor.view(-1)
    if flat.numel() <= 1:
        return torch.zeros_like(tensor)
    ranks = flat.argsort().argsort().float()
    return (ranks / (ranks.numel() - 1.0)).view(shape)


def update_scores_agents(model, epoch, total_epochs, beta=1.0):
    """
    Updates latent scores S_ij using Microglia (pruning) and Astrocyte (growing) signals (Dense Gradient Pass).
    Also computes and returns the Mask Flop Rate (structural convergence).
    """
    total_flops = 0
    total_elements = 0
    
    print("\n🔍 [DEBUG - update_scores_agents]")
    print("-" * 75)
    for name, module in model.named_modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            if not hasattr(module, 'grad_L_w'):
                print(f"  Layer {name}: ⚠️ Missing grad_L_w")
                continue

            # Capture old mask before score update
            old_mask = generate_2_4_mask(module.scores.data)

            # --- Feature Flow ---
            # :math: h^{(l)} = \phi( \tilde{W}^{(l)} [h^{(0)}, \dots, h^{(l-1)}] + b^{(l)} )
            # Phép nối tensor (Concatenation) dọc chiều kênh giúp mọi lớp tích chập nhận gradient 
            # trực tiếp từ Loss, chống triệt tiêu đạo hàm.
            # ----------------------------------------------
            w_val = module.weight.data

            # BẮT BUỘC TUÂN THỦ: Microglia cắt tỉa theo Relative Gradient Entropy (Rank-based)
            grad_micro_w = getattr(module, 'grad_micro_w', torch.zeros_like(w_val))
            micro_force = torch.abs(w_val * grad_micro_w)
            C_ij = compute_rank(micro_force)

            # BẮT BUỘC TUÂN THỦ: Astrocyte tối thiểu hóa Reverse KL Divergence đối với Uniform Dir(1)
            # Tuyệt đối không nhân gradient với trọng số hiện tại Wij để rễ mọc lại từ 0
            grad_astro_w = getattr(module, 'grad_astro_w', torch.zeros_like(w_val))
            astro_force = torch.abs(grad_astro_w)
            G_ij = compute_rank(astro_force)

            # BẮT BUỘC TUÂN THỦ: Động lực học Langevin chống Kết tinh (SGLD)
            if G_ij.max().item() <= 1e-8:
                u_e_node = getattr(module, 'u_e_node', None)
                if u_e_node is not None:
                    if isinstance(module, MDEPLinear):
                        g1 = u_e_node.unsqueeze(1).expand_as(w_val)
                    elif isinstance(module, MDEPConv2d):
                        g1 = u_e_node.view(-1, 1, 1, 1).expand_as(w_val)
                    else:
                        g1 = torch.zeros_like(w_val)
                else:
                    g1 = torch.zeros_like(w_val)
                g1_norm = g1 / (g1.max() + 1e-8)
                
                # Temperature schedule (eta_t * T)
                progress = epoch / max(total_epochs - 1, 1)
                eta_t = 0.001 + 0.5 * (0.05 - 0.001) * (1 + math.cos(math.pi * progress))
                T_temp = 1.0 # Temperature
                noise_scale = math.sqrt(2 * eta_t * T_temp)
                
                noise = noise_scale * torch.randn_like(G_ij) * g1_norm
                G_ij = G_ij + torch.clamp(noise, min=0.0)

            # Calculate total driving force Delta S
            delta_S = C_ij + G_ij
            
            # Step 1: Update Velocity (Momentum EMA)
            beta_m = 0.95
            module.scores_momentum.data.mul_(beta_m).add_(delta_S, alpha=1.0 - beta_m)
            
            # Step 2: Apply update to scores via gradient ascent
            # Ensure Multi-Agent Algorithmic Convergence (Eta Decay Schedule)
            import math
            progress = epoch / max(total_epochs - 1, 1)
            eta = 0.001 + 0.5 * (0.05 - 0.001) * (1 + math.cos(math.pi * progress))
            module.scores.data.add_(module.scores_momentum.data, alpha=eta)
            
            # Step 3: Zero-center scores to prevent global positive drift over time
            module.scores.data.sub_(module.scores.data.mean())

            # Step 4: Clamp scores to prevent infinite growth and gradient underflow (dead gradients)
            module.scores.data.clamp_(min=-5.0, max=5.0)
            
            # Compute new mask and count flops
            new_mask = generate_2_4_mask(module.scores.data)
            flops = (old_mask != new_mask).sum().item()
            total_flops += flops
            total_elements += old_mask.numel()

            # Print diagnostic info for each layer
            print(f"  Layer: {name}")
            print(f"    C_ij (Micro): min={C_ij.min().item():.4f}, max={C_ij.max().item():.4f}")
            print(f"    G_ij (Astro): min={G_ij.min().item():.4f}, max={G_ij.max().item():.4f}")
            print(f"    delta_S     : min={delta_S.min().item():.4f}, max={delta_S.max().item():.4f}")
            print(f"    Flips/Total : {flops} / {old_mask.numel()} ({flops / old_mask.numel() * 100:.4f}%)")
            print("-" * 50)

    flop_rate = total_flops / (total_elements + 1e-8)
    print(f"  >>> TOTAL FLOP RATE: {flop_rate*100:.6f}% ({total_flops} / {total_elements})")
    print("-" * 75)
    return flop_rate


# ============================================================================
#  SECTION 4 — Trainer (warm-up, cosine schedules, amortized gradients)
# ============================================================================

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) Optimizer.
    Tối ưu hóa tránh cực tiểu sắc nhọn, tăng cường khả năng tổng quát hóa (OOD).
    """
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # Leo lên đỉnh sắc nhọn \epsilon (w + e)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # Trở về vị trí cũ (w)
        # Note: Bỏ qua self.base_optimizer.step() để bên ngoài dùng Scaler
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't use step() directly, use first_step() and second_step()")


class MDEPTrainer:
    def __init__(self, model, optimizer, criterion, total_epochs, warmup_epochs=None, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.total_epochs = total_epochs
        self.scheduler = scheduler # <-- Thêm biến này
        if warmup_epochs is None:
            self.warmup_epochs = max(1, int(0.20 * total_epochs))
        else:
            self.warmup_epochs = warmup_epochs

        # Smoothed-STE temperature schedule
        self.gamma_initial = 5.0
        self.gamma_final = 0.15
        
        # AMP Scaler for Mixed Precision
        self.scaler = torch.amp.GradScaler('cuda')

    def step_gamma(self, epoch):
        """Cosine-annealed temperature for the Smoothed STE."""
        if epoch < self.warmup_epochs:
            return self.gamma_initial
        progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        gamma = self.gamma_final + 0.5 * (self.gamma_initial - self.gamma_final) * (
            1 + math.cos(math.pi * progress)
        )
        return gamma

    def check_gradient_flow(self, epoch):
        import os
        import matplotlib.pyplot as plt
        
        # Create artifacts folder if not exists
        artifacts_dir = os.path.join(os.getcwd(), "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        
        target_layers = []
        for name, m in self.model.named_modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                target_layers.append((name, m))
                
        if not target_layers:
            print("No MDEP layers found to check gradient flow.")
            return
            
        print(f"\n🔍 [Gradient Flow Check - Epoch {epoch}]")
        print("-" * 75)
        
        # We will check the last layer
        visualize_layers = [target_layers[-1]]
        
        for name, m in visualize_layers:
            w_val = m.weight.data.cpu()
            grad_micro = getattr(m, 'grad_micro_w', None)
            grad_astro = getattr(m, 'grad_astro_w', None)
            
            if grad_micro is None or grad_astro is None:
                print(f"Layer {name}: grad_micro_w or grad_astro_w is None. Cannot perform flow check.")
                continue
                
            grad_micro = grad_micro.cpu()
            grad_astro = grad_astro.cpu()
            
            # Magnitudes: Microglia uses |w * grad_micro|, Astrocyte uses |grad_astro|
            mag_micro = torch.abs(w_val * grad_micro)
            mag_astro = torch.abs(grad_astro)
            
            # Print raw statistics
            print(f"Layer: {name}")
            print(f"  Microglia Force: mean={mag_micro.mean().item():.2e}, std={mag_micro.std().item():.2e}, max={mag_micro.max().item():.2e}")
            print(f"  Astrocyte Force: mean={mag_astro.mean().item():.2e}, std={mag_astro.std().item():.2e}, max={mag_astro.max().item():.2e}")
            
            # Plot histograms
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Flatten to 1D
            mag_micro_flat = mag_micro.numpy().flatten()
            mag_astro_flat = mag_astro.numpy().flatten()
            
            ax1.hist(mag_micro_flat, bins=50, color='blue', alpha=0.7)
            ax1.set_title(f'Microglia Force ({name})')
            ax1.set_xlabel('Magnitude')
            ax1.set_ylabel('Count')
            
            ax2.hist(mag_astro_flat, bins=50, color='green', alpha=0.7)
            ax2.set_title(f'Astrocyte Force ({name})')
            ax2.set_xlabel('Magnitude')
            ax2.set_ylabel('Count')
            
            plt.tight_layout()
            plot_path = os.path.join(artifacts_dir, f"grad_ua_flow_epoch_{epoch}.png")
            plt.savefig(plot_path)
            plt.close(fig) # TRUYỀN ĐỐI TƯỢNG 'fig' VÀO ĐỂ ÉP XÓA RÁC RAM
            print(f"  Saved gradient histogram to: {plot_path}")
            print("-" * 75)
            
        # Dọn sạch toàn bộ rác đồ họa còn sót lại sau khi lặp qua tất cả các layers
        import gc
        plt.close('all')
        gc.collect()

    def set_warmup_state(self, epoch, is_warmup, gamma):
        for module in self.model.modules():
            if isinstance(module, (MDEPLinear, MDEPConv2d)):
                module.current_epoch = epoch
                module.warmup_epochs = self.warmup_epochs
                module.warmup = is_warmup
                module.gamma = gamma

    def set_mask_freeze_state(self, freeze):
        for module in self.model.modules():
            if isinstance(module, (MDEPLinear, MDEPConv2d)):
                module.freeze_mask = freeze

    def compute_amortized_gradients(self, inputs, tab_inputs=None):
        """
        Amortized backward passes that compute:
          • ∂u_a / ∂w      → signal for the Microglia agent
          • ∂u_e / ∂a^(l)  → signal for the Astrocyte agent (per-neuron)
        Called only once per epoch to keep FLOPs low.
        """
        # [FIX: BatchNorm Double-Update Skew] Switch to eval() during forward pass to preserve batch normalization statistics
        self.model.eval()

        # Register forward hooks to capture layer activations for Astrocyte
        activations = {}
        hooks = []
        for name, m in self.model.named_modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                def _hook(module, inp, out, n=name):
                    activations[n] = out
                hooks.append(m.register_forward_hook(_hook))

        # Dùng AMP Autocast để giảm 50% VRAM khi lưu đồ thị tính toán của Astrocyte
        with torch.amp.autocast('cuda'):
            outputs = self.model(inputs, tab_inputs)
        
        # [FIX: BatchNorm Double-Update Skew] Revert to train() immediately after forward pass
        self.model.train()
        
        # Ép kiểu về FP32 (float) để hàm digamma không bị sinh ra NaN do lỗi precision FP16
        uncertainties = compute_uncertainties(outputs.float())

        u_a_sample = uncertainties['aleatoric']
        u_e_sample = uncertainties['epistemic']
        alpha = uncertainties['alpha']
        S = uncertainties['S']
        K = alpha.shape[1]
        device = alpha.device

        # 1. BẮT BUỘC TUÂN THỦ: Microglia agent: L_micro = mean(u_a / (u_e + 1e-8))
        loss_micro = torch.mean(u_a_sample / (u_e_sample + 1e-8))
        self.model.zero_grad()
        loss_micro.backward(retain_graph=True)
        for m in self.model.modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                if m.weight.grad is not None:
                    m.grad_micro_w = m.weight.grad.clone().detach()
                else:
                    m.grad_micro_w = torch.zeros_like(m.weight)

        # 2. BẮT BUỘC TUÂN THỦ: Astrocyte agent: L_astro = Reverse KL(Dir(1) || Dir(alpha))
        term1 = torch.lgamma(torch.tensor(float(K), device=device))
        term3 = -torch.lgamma(S)
        term4 = torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        digamma_1 = torch.digamma(torch.tensor(1.0, device=device))
        digamma_K = torch.digamma(torch.tensor(float(K), device=device))
        term5 = torch.sum((1.0 - alpha) * (digamma_1 - digamma_K), dim=1, keepdim=True)
        loss_astro = torch.mean(term1 + term3 + term4 + term5)
        
        self.model.zero_grad()
        loss_astro.backward(retain_graph=True)
        for m in self.model.modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                if m.weight.grad is not None:
                    m.grad_astro_w = m.weight.grad.clone().detach()
                else:
                    m.grad_astro_w = torch.zeros_like(m.weight)

        # Clear grads for the next step
        self.model.zero_grad()

        # 3. ∂u_e/∂a^(l) → SGLD (Convolutional Activation Gradient Pooling 4D)
        u_e = torch.mean(u_e_sample)
        act_tensors = []
        act_modules = []
        for name, m in self.model.named_modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)) and name in activations:
                act_tensors.append(activations[name])
                act_modules.append(m)

        if act_tensors:
            # Giải phóng bớt áp lực RAM bằng cách chặn gradient backprop quá sâu (Chỉ lấy 20 lớp cuối)
            safe_tensors = act_tensors[-20:] 
            safe_modules = act_modules[-20:]
            grads = torch.autograd.grad(u_e, safe_tensors, allow_unused=True, retain_graph=False)
            
            for m, grad in zip(safe_modules, grads):
                if grad is not None:
                    if isinstance(m, MDEPLinear):
                        m.u_e_node = torch.abs(grad).mean(dim=0).detach()
                    elif isinstance(m, MDEPConv2d):
                        m.u_e_node = torch.abs(grad).mean(dim=(0, 2, 3)).detach() # Pooling 4D
                else:
                    m.u_e_node = None

        # Clean up hooks
        for h in hooks:
            h.remove()
        self.model.zero_grad()

        # [FIX: CUDA OOM / Graph Leak] Explicitly delete heavy tensors holding the graph and clear cache
        # [FIX: NameError] removed alpha and S since they are not local variables
        del outputs, uncertainties, u_a, u_e
        torch.cuda.empty_cache()

    def train_epoch(self, epoch, dataloader, device, print_interval=200):
        self.model.train()

        is_warmup = epoch < self.warmup_epochs
        gamma = self.step_gamma(epoch)
        self.set_warmup_state(epoch, is_warmup, gamma)

        ema_loss = None
        ema_grad = None
        num_batches = len(dataloader)
        epoch_start = time.time()

        for batch_idx, data in enumerate(dataloader):
            # Base LR from optimizer (updated by OneCycleLR)
            current_lr = self.optimizer.param_groups[0]['lr']

            if len(data) == 3:
                inputs, tab_inputs, targets = data
                inputs, tab_inputs, targets = inputs.to(device), tab_inputs.to(device), targets.to(device)
            else:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                tab_inputs = None

            # Amortized uncertainty-gradient pass on the first batch of the epoch (also during warm-up epochs 0 and 1)
            if (not is_warmup or epoch < 2) and batch_idx == 0:
                self.compute_amortized_gradients(inputs, tab_inputs)

            self.model.zero_grad()
            self.set_mask_freeze_state(is_warmup or batch_idx > 0)
            
            # Use Automatic Mixed Precision for Forward Pass
            with torch.amp.autocast('cuda'):
                evidence = self.model(inputs, tab_inputs)
                
            # Ensure Evidential Loss runs strictly in FP32 to avoid digamma/log underflow
            with torch.amp.autocast('cuda', enabled=False):
                loss = self.criterion(evidence.float(), targets, epoch)
            
            # Loss scaling removed to resolve conflict with AMP
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer.base_optimizer)
            
            params_to_clip = [p for group in self.optimizer.param_groups for p in group['params']]
            grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
            
            # --- LÁ CHẮN AMP: KIỂM TRA NAN TRƯỚC KHI CHO PHÉP SAM HOẠT ĐỘNG ---
            is_valid_grad = not (torch.isnan(grad_norm).item() or torch.isinf(grad_norm).item())
            
            if is_valid_grad:
                if ema_grad is None:
                    ema_grad = grad_norm.item()
                else:
                    ema_grad = 0.95 * ema_grad + 0.05 * grad_norm.item()

                is_mdep_batch = (not is_warmup and batch_idx == 0)
                scale_before = self.scaler.get_scale()

                if is_mdep_batch:
                    # Cache primary weight gradient for Microglia
                    for m in self.model.modules():
                        if isinstance(m, (MDEPLinear, MDEPConv2d)):
                            if m.weight.grad is not None:
                                m.grad_L_w = m.weight.grad.clone().detach()
                            else:
                                m.grad_L_w = torch.zeros_like(m.weight)
                    self.check_gradient_flow(epoch)
                    
                    self.scaler.step(self.optimizer.base_optimizer)
                    self.scaler.update()
                    
                    # Multi-agent structure optimization
                    mask_flop_rate = update_scores_agents(self.model, epoch, self.total_epochs)
                    self.last_flop_rate = mask_flop_rate
                else:
                    # BƯỚC 2: Gọi SAM first_step an toàn
                    self.optimizer.first_step(zero_grad=True)
                    
                    # BƯỚC 3: Tính Forward 2 -> Loss 2 -> Backward 2
                    with torch.amp.autocast('cuda'):
                        evidence2 = self.model(inputs, tab_inputs)
                    with torch.amp.autocast('cuda', enabled=False):
                        loss2 = self.criterion(evidence2.float(), targets, epoch)
                    
                    self.scaler.scale(loss2).backward()
                    
                    # BƯỚC 4: Về lại điểm gốc và cập nhật
                    self.optimizer.second_step(zero_grad=False)
                    self.scaler.step(self.optimizer.base_optimizer)
                    self.scaler.update()

                scale_after = self.scaler.get_scale()
                
                # FIX WARNING: Chỉ step scheduler nếu optimizer thực sự đã step
                if scale_before <= scale_after and self.scheduler is not None:
                    self.scheduler.step()
            else:
                # Nếu dính NaN từ AMP, chỉ update scaler để nó tự hạ scale xuống
                self.scaler.update()

            self.model.zero_grad()

            # BẢO VỆ EMA LOSS: Không cộng dồn loss lỗi vào log
            if not torch.isnan(loss) and not torch.isinf(loss):
                if ema_loss is None:
                    ema_loss = loss.item()
                else:
                    ema_loss = 0.95 * ema_loss + 0.05 * loss.item()

            # Progress printing
            if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.time() - epoch_start
                avg_time = elapsed / (batch_idx + 1)
                eta = avg_time * (num_batches - batch_idx - 1)
                avg_loss = ema_loss if ema_loss is not None else 0.0
                avg_grad = ema_grad if ema_grad is not None else 0.0
                
                flop_str = f"| Flop: {self.last_flop_rate*100:.4f}%  " if hasattr(self, 'last_flop_rate') else ""
                
                print(
                    f"    Batch [{batch_idx+1:>5}/{num_batches}]  "
                    f"| Loss: {avg_loss:.4f}  "
                    f"| LR: {current_lr:.2e}  "
                    f"| GradNorm: {avg_grad:.4f}  "
                    f"{flop_str}"
                    f"| Elapsed: {elapsed/60:.1f}m  "
                    f"| ETA: {eta/60:.1f}m",
                    flush=True,
                )
                
                if wandb.run is not None:
                    wandb.log({
                        "train/batch_loss": avg_loss,
                        "train/learning_rate": current_lr,
                        "train/grad_norm": avg_grad,
                        "train/mask_flop_rate": self.last_flop_rate if hasattr(self, 'last_flop_rate') else 0.0
                    })

        return ema_loss if ema_loss is not None else 0.0


# ============================================================================
#  SECTION 5 — ISIC 2024 Dataset + ResNet backbone + main()
# ============================================================================

class ISICDataset(Dataset):
    """PyTorch Dataset for the ISIC 2024 Skin Cancer challenge on Kaggle.
    Supports loading images from individual files OR from an HDF5 archive."""
    def __init__(self, dataframe, image_dir, tabular_cols=None, transform=None, hdf5_path=None):
        self.data_frame = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.hdf5_path = hdf5_path
        self._hdf5_file = None
        self.tabular_cols = tabular_cols or []
        self._error_printed = False

    def _get_hdf5(self):
        """Lazy-open HDF5 file (one handle per worker process)."""
        if self._hdf5_file is None and self.hdf5_path and HAS_H5PY:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r')
        return self._hdf5_file

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        isic_id = self.data_frame.iloc[idx]['isic_id']
        image = None

        if self.hdf5_path and getattr(self, 'HAS_H5PY', globals().get('HAS_H5PY', False)):
            try:
                hf = self._get_hdf5()
                if hf and isic_id in hf:
                    img_bytes = hf[isic_id][()]
                    # [FIX: CPU/GPU Dataloader Bottleneck] Use cv2 to decode bytes directly
                    np_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if np_img is not None:
                        image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
            except Exception:
                pass

        if image is None and self.image_dir:
            img_path = os.path.join(self.image_dir, f"{isic_id}.jpg")
            try:
                # [FIX: CPU/GPU Dataloader Bottleneck] Use cv2 for image reading
                np_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if np_img is not None:
                    image = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                else:
                    image = Image.new('RGB', (384, 384), color='black')
            except Exception:
                image = Image.new('RGB', (384, 384), color='black')
                
        if image is None:
            image = Image.new('RGB', (384, 384), color='black')

        target = self.data_frame.iloc[idx]['target']
        
        # EXTRACT TABULAR METADATA
        if self.tabular_cols:
            tabular = self.data_frame.iloc[idx][self.tabular_cols].values.astype(np.float32)
            tabular_tensor = torch.tensor(tabular)
        else:
            tabular_tensor = torch.zeros(1, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, tabular_tensor, torch.tensor(target, dtype=torch.long)


def get_isic_dataloaders(batch_size=32, val_ratio=0.1, test_ratio=0.2):
    tabular_cols = []
    """
    Returns (train_loader, val_loader, test_loader, num_classes, cw, tabular_dim).
    Uses stratified 70/10/20 split. Falls back to dummy data if not on Kaggle.
    """
    num_classes = 2

    # Auto-detect the ISIC dataset path under /kaggle/input/
    # Competition datasets are mounted under /kaggle/input/competitions/<slug>/
    # Regular datasets are mounted under /kaggle/input/<slug>/
    csv_path = None
    image_dir = None
    kaggle_input = '/kaggle/input'
    
    # Debug: show full tree under /kaggle/input/
    print(f"🔍 Checking Kaggle input dir: {kaggle_input}")
    print(f"   Exists? {os.path.isdir(kaggle_input)}")
    if os.path.isdir(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            depth = root.replace(kaggle_input, '').count(os.sep)
            if depth < 3:  # Only show first 3 levels
                indent = '   ' + '  ' * depth
                print(f"{indent}📁 {os.path.basename(root)}/")
                for f in files[:5]:  # Show first 5 files per dir
                    print(f"{indent}  📄 {f}")
                if len(files) > 5:
                    print(f"{indent}  ... and {len(files)-5} more files")
    
    def _try_find_dataset(base_dir):
        """Search for train-metadata.csv in a directory and return (csv_path, image_dir) or (None, None)."""
        if not os.path.isdir(base_dir):
            return None, None
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            candidate_csv = os.path.join(folder_path, 'train-metadata.csv')
            if os.path.exists(candidate_csv):
                img_dir = None
                for img_sub in ['train-image/image', 'train-image', 'train-images/image', 'train-images']:
                    candidate_img = os.path.join(folder_path, img_sub)
                    if os.path.isdir(candidate_img):
                        img_dir = candidate_img
                        break
                if img_dir is None:
                    img_dir = os.path.join(folder_path, 'train-image')
                print(f"✅ Found ISIC dataset at: {folder_path}/")
                return candidate_csv, img_dir
        return None, None
    
    # Strategy 1: Check directly under /kaggle/input/<slug>/
    csv_path, image_dir = _try_find_dataset(kaggle_input)
    
    # Strategy 2: Check under /kaggle/input/competitions/<slug>/
    if csv_path is None:
        competitions_dir = os.path.join(kaggle_input, 'competitions')
        csv_path, image_dir = _try_find_dataset(competitions_dir)
    
    # Strategy 3: Recursive scan — check ALL subdirectories up to 2 levels deep
    # [DISABLED] action 4: Resolving Kaggle I/O Dataset Hang by skipping os.walk
    # if csv_path is None and os.path.isdir(kaggle_input):
    #     for root, dirs, files in os.walk(kaggle_input):
    #         depth = root.replace(kaggle_input, '').count(os.sep)
    #         if depth > 2:
    #             continue
    #         if 'train-metadata.csv' in files:
    #             csv_path = os.path.join(root, 'train-metadata.csv')
    #             for img_sub in ['train-image/image', 'train-image', 'train-images/image', 'train-images']:
    #                 candidate_img = os.path.join(root, img_sub)
    #                 if os.path.isdir(candidate_img):
    #                     image_dir = candidate_img
    #                     break
    #             if image_dir is None:
    #                 image_dir = os.path.join(root, 'train-image')
    #             print(f"✅ Found ISIC dataset via deep scan at: {root}/")
    #             break
    
    if csv_path:
        print(f"📂 CSV path:   {csv_path}")
        print(f"📂 Image dir:  {image_dir}")
    else:
        print(f"❌ train-metadata.csv not found anywhere under {kaggle_input}")

    train_tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90), # Xoay ngẫu nhiên cực đại
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Kéo giãn nhẹ
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if csv_path is None or not os.path.exists(csv_path):
        raise FileNotFoundError("❌ Không tìm thấy ISIC 2024 Dataset! Script dừng lại để tránh train trên Dummy Data.")

    df = pd.read_csv(csv_path)
    print(f"📊 Loaded CSV with {len(df)} rows, columns: {list(df.columns[:5])}")
    
    # Detect HDF5 archive for images
    hdf5_path = None
    dataset_root = os.path.dirname(csv_path)
    for hdf5_name in ['train-image.hdf5', 'train-image.h5']:
        candidate = os.path.join(dataset_root, hdf5_name)
        if os.path.exists(candidate):
            hdf5_path = candidate
            print(f"📂 HDF5 archive: {hdf5_path}")
            break
    
    # Debug: list available files in dataset root
    if os.path.isdir(dataset_root):
        print(f"📂 Dataset contents: {os.listdir(dataset_root)}")
    
    # Subsample to keep training feasible within Kaggle session limits.
    # Set MAX_SAMPLES = None to use the full dataset.
    MAX_SAMPLES = None  # None = use full dataset (401K samples)
    if MAX_SAMPLES is not None and len(df) > MAX_SAMPLES:
        print(f"📉 Subsampling: {len(df)} → {MAX_SAMPLES} samples (set MAX_SAMPLES=None for full dataset)")
        df = df.groupby('target', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), int(MAX_SAMPLES * len(x) / len(df))), random_state=42)
        ).reset_index(drop=True)
        print(f"   After stratified subsample: {len(df)} samples, target distribution:")
        print(f"   {df['target'].value_counts().to_dict()}")

    # --- Eradicate Data Leakage via Strict Patient-Level Cross Validation ---
    # [FIX: Data Leakage] Implement proper 70/10/20 train/val/test split FIRST
    from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
    
    if 'patient_id' in df.columns:
        print('🧬 Found patient_id. Using GroupShuffleSplit to prevent data leakage.')
        # First split off test set
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
        train_val_idx, test_idx = next(gss_test.split(df, df['target'], groups=df['patient_id']))
        train_val_df = df.iloc[train_val_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        # Then split remaining into train and val
        val_relative_ratio = val_ratio / (1.0 - test_ratio)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative_ratio, random_state=42)
        train_idx, val_idx = next(gss_val.split(train_val_df, train_val_df['target'], groups=train_val_df['patient_id']))
        train_df = train_val_df.iloc[train_idx].copy()
        val_df = train_val_df.iloc[val_idx].copy()
    else:
        print('⚠ patient_id not found. Falling back to StratifiedKFold.')
        skf_test = StratifiedKFold(n_splits=int(1/test_ratio), shuffle=True, random_state=42)
        train_val_idx, test_idx = next(skf_test.split(df, df['target']))
        train_val_df = df.iloc[train_val_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        val_relative_ratio = val_ratio / (1.0 - test_ratio)
        skf_val = StratifiedKFold(n_splits=max(2, int(1/val_relative_ratio)), shuffle=True, random_state=42)
        train_idx, val_idx = next(skf_val.split(train_val_df, train_val_df['target']))
        train_df = train_val_df.iloc[train_idx].copy()
        val_df = train_val_df.iloc[val_idx].copy()

    # --- Multi-Modal Tabular Preprocessing (Safe Extraction) ---
    # [FIX: Tabular Data Leakage] Apply fit() ON TRAIN ONLY, transform() on val/test
    desired_cols = ['age_approx', 'sex', 'anatom_site_general_challenge', 'anatom_site_general', 'clin_size_long_diam_mm']
    available_cols = [c for c in desired_cols if c in df.columns]
    
    if available_cols:
        # Fill missing values based on Train statistics
        if 'age_approx' in available_cols:
            age_med = train_df['age_approx'].median()
            train_df['age_approx'] = train_df['age_approx'].fillna(age_med)
            val_df['age_approx'] = val_df['age_approx'].fillna(age_med)
            test_df['age_approx'] = test_df['age_approx'].fillna(age_med)
            
        if 'clin_size_long_diam_mm' in available_cols:
            size_med = train_df['clin_size_long_diam_mm'].median()
            train_df['clin_size_long_diam_mm'] = train_df['clin_size_long_diam_mm'].fillna(size_med)
            val_df['clin_size_long_diam_mm'] = val_df['clin_size_long_diam_mm'].fillna(size_med)
            test_df['clin_size_long_diam_mm'] = test_df['clin_size_long_diam_mm'].fillna(size_med)
            
        cat_candidates = ['sex', 'anatom_site_general_challenge', 'anatom_site_general']
        cat_cols = [c for c in cat_candidates if c in available_cols]
        for col in cat_cols:
            mode_val = train_df[col].mode()[0] if not train_df[col].mode().empty else 'unknown'
            train_df[col] = train_df[col].fillna(mode_val)
            val_df[col] = val_df[col].fillna(mode_val)
            test_df[col] = test_df[col].fillna(mode_val)
            
        # Get dummies
        if cat_cols:
            train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
            val_df = pd.get_dummies(val_df, columns=cat_cols, drop_first=True)
            test_df = pd.get_dummies(test_df, columns=cat_cols, drop_first=True)
            
            # Align columns for Val and Test
            val_df = val_df.reindex(columns=train_df.columns, fill_value=0)
            test_df = test_df.reindex(columns=train_df.columns, fill_value=0)
            
        num_cols = [c for c in ['age_approx', 'clin_size_long_diam_mm'] if c in available_cols]
        if num_cols:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
            val_df[num_cols] = scaler.transform(val_df[num_cols])
            test_df[num_cols] = scaler.transform(test_df[num_cols])
            
        # Determine tabular columns
        tabular_cols = num_cols + [c for c in train_df.columns if c.startswith(tuple(cat_cols + ['_'])) and c not in df.columns]

    print(f'📊 Train: {len(train_df)} samples  |  Val: {len(val_df)} samples  |  Test: {len(test_df)} samples')
    train_ds = ISICDataset(train_df, image_dir, tabular_cols=tabular_cols, transform=train_tf, hdf5_path=hdf5_path)
    val_ds   = ISICDataset(val_df,   image_dir, tabular_cols=tabular_cols, transform=test_tf,  hdf5_path=hdf5_path)
    test_ds  = ISICDataset(test_df,  image_dir, tabular_cols=tabular_cols, transform=test_tf,  hdf5_path=hdf5_path)
    
    # Clinical Grade: Thread-Safe HDF5 Multiprocessing
    def h5_worker_init_fn(worker_id):
        import h5py
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        if hasattr(dataset, 'hdf5_path') and getattr(dataset, 'HAS_H5PY', globals().get('HAS_H5PY', False)):
            dataset._hdf5_file = h5py.File(dataset.hdf5_path, 'r')

    # [FIX: Kaggle OOM Risk] Reduce num_workers from 4 to 2 to prevent RAM overflow when using HDF5 multiprocessing
    # Fix HDF5 Deadlock & Memory Leak trên Kaggle: Ép chạy Main Thread
    # Tối ưu Bottleneck: Dùng 2 luồng + prefetch để CPU cbi sẵn data cho GPU
    nw = 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True, worker_init_fn=h5_worker_init_fn, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True, worker_init_fn=h5_worker_init_fn, prefetch_factor=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True, worker_init_fn=h5_worker_init_fn, prefetch_factor=2)

    total_samples = len(train_df)
    pos_samples = train_df['target'].sum()
    neg_samples = total_samples - pos_samples
    
    # Khắc phục Bùng nổ Gradient bằng Effective Number of Samples (ENS)
    beta = 0.9999
    # Tính Effective Number of Samples (ENS)
    E_neg = (1.0 - (beta ** neg_samples)) / (1.0 - beta)
    E_pos = (1.0 - (beta ** pos_samples)) / (1.0 - beta)
    
    # Trọng số thô
    W_neg = 1.0 / E_neg
    W_pos = 1.0 / E_pos
    
    # Chuẩn hóa trọng số: tổng trọng số bằng tổng số mẫu
    norm_factor = total_samples / (W_neg * neg_samples + W_pos * pos_samples)
    weight_neg = W_neg * norm_factor
    weight_pos = W_pos * norm_factor
    
    cw = torch.tensor([weight_neg, weight_pos], dtype=torch.float32)
    
    # Mảng đếm số lượng cho Focal Loss
    class_counts = [neg_samples, pos_samples]
    print(f"⚖️  Class weights calculated: Benign={weight_neg:.4f}, Malignant={weight_pos:.4f}")

    return train_loader, val_loader, test_loader, num_classes, cw, class_counts, len(tabular_cols)


def replace_conv2d_with_mdep(model):
    """Recursively swap nn.Conv2d / nn.Linear → MDEPConv2d / MDEPLinear."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            new = MDEPConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding,
                dilation=module.dilation, groups=module.groups,
                bias=(module.bias is not None),
            )
            new.weight.data.copy_(module.weight.data)
            new.scores.data.copy_(torch.abs(module.weight.data))
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(model, name, new)
        elif isinstance(module, nn.Linear):
            new = MDEPLinear(
                module.in_features, module.out_features,
                bias=(module.bias is not None),
            )
            new.weight.data.copy_(module.weight.data)
            new.scores.data.copy_(torch.abs(module.weight.data))
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(model, name, new)
        else:
            replace_conv2d_with_mdep(module)


# ============================================================================
#  SECTION 6 — Evaluation, Metrics & Visualization
# ============================================================================

def sinkhorn_knopp(x, y, epsilon=0.1, n_iters=5):
    """
    Optimal Transport Sinkhorn alignment (Per-Sample, Safe Fusion).
    x: Image tokens [B, N, D]
    y: Tabular tokens [B, M, D]
    Trả về ma trận Coupling P kích thước [B, N, M]
    """
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    
    # Cost Matrix C: Khoảng cách Cosine theo từng sample [B, N, M]
    C = 1.0 - torch.bmm(x_norm, y_norm.transpose(1, 2))
    K = torch.exp(-C / epsilon)
    
    u = torch.ones_like(K[:, :, 0]).unsqueeze(2) / K.size(1) # [B, N, 1]
    v = torch.ones_like(K[:, 0, :]).unsqueeze(1) / K.size(2) # [B, 1, M]
    
    for _ in range(n_iters):
        u = (1.0 / K.size(1)) / (torch.bmm(K, v.transpose(1, 2)) + 1e-8)
        v = (1.0 / K.size(2)) / (torch.bmm(K.transpose(1, 2), u) + 1e-8).transpose(1, 2)
        
    P = u * K * v # [B, N, M]
    return P

class MultimodalDenseNet(nn.Module):
    """Multi-Modal Architecture: Fuses DenseNet-121 image features with tabular metadata."""
    def __init__(self, backbone, tab_features, num_classes):
        super().__init__()
        self.features = backbone.features
        self.use_tabular = tab_features > 0
        in_features = backbone.classifier.in_features
        
        if self.use_tabular:
            self.tab_mlp = nn.Sequential(
                nn.Linear(tab_features, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.4)  # <-- Thêm lớp phân tán nhiễu (Dropout 40%)
            )
            # [FIX: Multi-modal Fusion] Gated Attention / SE block to weight modalities
            self.attn_mlp = nn.Sequential(
                nn.Linear(in_features + 64, (in_features + 64) // 16),
                nn.ReLU(),
                nn.Linear((in_features + 64) // 16, in_features + 64),
                nn.Sigmoid()
            )
            self.classifier = nn.Sequential(
                nn.Linear(in_features + 64, num_classes),
                EvidenceLayer(activation='softplus')
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(in_features, num_classes),
                EvidenceLayer(activation='softplus')
            )

    def forward(self, img_x, tab_x=None):
        features = self.features(img_x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        img_feats = torch.flatten(out, 1)
        
        if self.use_tabular:
            if tab_x is None:
                tab_x = torch.zeros((img_x.size(0), self.tab_mlp[0].in_features), device=img_x.device)
            tab_out = self.tab_mlp(tab_x) # [B, 64]
            
            # --- SINKHORN FUSION: Tokenize features to match dimensions ---
            # img_feats: [B, 1024] -> Reshape thành 16 tokens chiều 64: [B, 16, 64]
            img_seq = img_feats.view(img_feats.size(0), 16, 64)
            # tab_out: [B, 64] -> Reshape thành 1 token chiều 64: [B, 1, 64]
            tab_seq = tab_out.unsqueeze(1)
            
            # Tính ma trận vận tải tối ưu P [B, 16, 1]
            P = sinkhorn_knopp(img_seq, tab_seq, epsilon=0.1, n_iters=5)
            
            # Căn chỉnh Tabular theo Image: [B, 1, 16] x [B, 16, 64] -> [B, 1, 64]
            aligned_tab = torch.bmm(P.transpose(1, 2), img_seq).squeeze(1)
            
            # Dung hợp và chạy qua Attention Gate
            fuse = torch.cat([img_feats, aligned_tab], dim=1)
            attn_weights = self.attn_mlp(fuse)
            fuse = fuse * attn_weights
            return self.classifier(fuse)
            
        return self.classifier(img_feats)

def compute_ece(confidences, accuracies, n_bins=15):
    """Expected Calibration Error with equal-width bins."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_sizes = [], [], []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_sizes.append(0)
            continue
        b_acc  = accuracies[mask].mean()
        b_conf = confidences[mask].mean()
        b_size = mask.sum()
        ece += (b_size / len(confidences)) * abs(b_acc - b_conf)
        bin_accs.append(b_acc)
        bin_confs.append(b_conf)
        bin_sizes.append(b_size)
    return ece, bin_accs, bin_confs, bin_sizes


def plot_reliability_diagram(bin_accs, bin_confs, bin_sizes, n_bins=15):
    """Reliability diagram: accuracy vs confidence per bin."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    x = np.arange(n_bins)
    width = 0.8
    ax.bar(x, bin_accs, width, label='Accuracy', color='#4e79a7', alpha=0.85)
    ax.bar(x, bin_confs, width, label='Confidence', color='#e15759', alpha=0.4)
    ax.plot([-0.5, n_bins - 0.5], [0, 1], 'k--', linewidth=1, label='Perfect')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Value')
    ax.set_title('Reliability Diagram')
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


def plot_uncertainty_histogram(u_e_correct, u_e_incorrect):
    """Overlaid histograms of epistemic uncertainty for correct vs wrong."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if len(u_e_correct) > 0:
        ax.hist(u_e_correct, bins=40, alpha=0.6, label='Correct', color='#59a14f')
    if len(u_e_incorrect) > 0:
        ax.hist(u_e_incorrect, bins=40, alpha=0.6, label='Incorrect', color='#e15759')
    ax.set_xlabel('Epistemic Uncertainty (u_e)')
    ax.set_ylabel('Count')
    ax.set_title('Uncertainty Distribution')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_pr_curve(y_true, probs):
    """Precision-Recall Curve with AUC."""
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(recall, precision, color='#86bcB6', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_risk_coverage_curve(y_true, y_pred, confidences):
    """Risk-Coverage curve and AURC (Area Under Risk-Coverage)."""
    # Sort instances by descending confidence
    sorted_indices = np.argsort(-confidences)
    sorted_true = y_true[sorted_indices]
    sorted_pred = y_pred[sorted_indices]
    
    n_samples = len(y_true)
    errors = (sorted_true != sorted_pred).astype(float)
    cumulative_errors = np.cumsum(errors)
    
    coverages = np.arange(1, n_samples + 1) / n_samples
    risks = cumulative_errors / np.arange(1, n_samples + 1)
        
    aurc = auc(coverages, risks)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(coverages, risks, color='#f28e2b', lw=2, label=f'Risk-Coverage (AURC = {aurc:.4f})')
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Risk (Error Rate)')
    ax.set_title('Risk-Coverage Curve')
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def check_representational_collapse(model):
    """Diagnoses Representational Collapse, with support for Dense/Warmup mode."""
    print("\n🔬 Representational Collapse Diagnostics")
    print("-" * 105)
    print(f"  {'Layer':30s} | {'Score Std':12s} | {'Score Mean':12s} | {'Grad Norm':12s} | {'Status'}")
    print("-" * 105)
    
    all_pass = True
    for name, module in model.named_modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            scores = module.scores.data
            std = scores.std().item() if scores.numel() > 1 else 0.0
            mean = scores.mean().item()
            
            grad_norm = 0.0
            grad_L = getattr(module, 'grad_L_w', None)
            if grad_L is not None:
                grad_norm = grad_L.norm().item()
                
            is_warmup = getattr(module, 'warmup', False)
            
            # Nhận diện chế độ Dense/Warmup
            if is_warmup or grad_L is None:
                status = "🟢 DENSE MODE (No Pruning)"
            else:
                status = "✅ PASS"
                issues = []
                if std <= 1e-4: issues.append("Zero Variance")
                if mean <= -10.0: issues.append("Negative Drift")
                if grad_norm <= 1e-9: issues.append("Dead Gradient")
                    
                if issues:
                    status = "❌ FAIL (" + ", ".join(issues) + ")"
                    all_pass = False
                    
            print(f"  {name:30s} | {std:12.4e} | {mean:12.4f} | {grad_norm:12.4e} | {status}")
            
    print("-" * 105)
    if all_pass:
        print("  🌟 OVERALL STATUS: HEALTHY (No Representational Collapse Detected)")
    else:
        print("  ⚠️ OVERALL STATUS: WARNING (Representational Collapse Detected in some layers)")
    print()


def print_sparsity_report(model):
    """Per-layer and total sparsity stats, recognizing full-dense capacity."""
    print("\n📐 Sparsity & Hardware Metrics Report")
    print("-" * 75)
    total_params = 0
    total_zeros  = 0
    total_macs_dense = 0
    total_macs_sparse = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            mask = getattr(module, 'cached_mask', module.mask)
            n = mask.numel()
            z = (mask == 0).sum().item()
            total_params += n
            total_zeros  += z
            sparsity = z / n * 100 if n > 0 else 0.0
            
            macs_dense = n
            macs_sparse = n - z
            total_macs_dense += macs_dense
            total_macs_sparse += macs_sparse
            
            # Nhận diện 2:4 hoặc Dense
            if n % 4 == 0:
                blocks = mask.view(-1, 4)
                valid = (blocks.sum(dim=1) == 2).all().item()
                if valid:
                    pattern = "✅ 2:4 (TensorCore Ready)"
                elif sparsity == 0.0:
                    pattern = "🟢 Dense (Full Capacity)"
                else:
                    pattern = "❌ Not 2:4"
            else:
                pattern = "⚠ skip (size%4≠0)"
            print(f"  {name:30s} | {sparsity:5.1f}% sparse | {pattern}")
            
    overall = total_zeros / total_params * 100 if total_params > 0 else 0.0
    macs_saved = (total_macs_dense - total_macs_sparse) / total_macs_dense * 100 if total_macs_dense > 0 else 0.0
    print("-" * 75)
    print(f"  {'TOTAL PARAMS':30s} | {overall:5.1f}% sparse")
    print(f"  {'THEORETICAL MACs SAVED':30s} | {macs_saved:5.1f}% reduction")
    print()
    
    check_representational_collapse(model)


@torch.no_grad()
def find_optimal_threshold(model, val_loader, device):
    """Finds the optimal classification threshold using the Validation set to prevent data leakage."""
    model.eval()
    all_targets = []
    all_probs = []
    
    for data in val_loader:
        if len(data) == 3:
            inputs, tab_inputs, targets = data
            inputs, tab_inputs = inputs.to(device), tab_inputs.to(device)
        else:
            inputs, targets = data
            inputs = inputs.to(device)
            tab_inputs = None
            
        evidence = model(inputs, tab_inputs)
        unc = compute_uncertainties(evidence)
        p_hat = (unc['alpha'] / unc['S']).cpu().numpy()
        
        all_targets.append(targets.numpy())
        all_probs.append(p_hat[:, 1])  # Class 1 probabilities
        
    y_true = np.concatenate(all_targets)
    probs = np.concatenate(all_probs)
    
    best_t = 0.5
    best_score = -1.0
    
    for t in np.linspace(0.01, 0.99, 1000):
        preds_t = (probs >= t).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, preds_t, labels=[0,1]).ravel()
        
        sens_t = tp_t / (tp_t + fn_t + 1e-8)
        spec_t = tn_t / (tn_t + fp_t + 1e-8)
        prec_t = tp_t / (tp_t + fp_t + 1e-8)
        
        f2_t = (5 * prec_t * sens_t) / (4 * prec_t + sens_t + 1e-8)
        penalty = 1.0 if spec_t >= 0.80 else (spec_t / 0.80) ** 3
        custom_score = f2_t * penalty
        
        if custom_score > best_score:
            best_score = custom_score
            best_t = t
            
    print(f"\n🔍 Threshold tuned on Validation Set. Optimal best_t = {best_t:.4f}")
    return best_t


def compute_isic_pauc(y_true, y_pred, min_tpr=0.80):
    """
    Hàm tính Raw pAUC chuẩn hệ thống chấm điểm của Kaggle ISIC 2024.
    Mục tiêu: Tính diện tích thực dưới đường cong ROC tại vùng có TPR >= 0.80 (Max = 0.20).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 1. Thủ thuật đảo ngược (Flip Trick) của Kaggle
    v_gt = np.abs(y_true - 1)
    v_pred = -1.0 * y_pred
    max_fpr = np.abs(1 - min_tpr) # Tương đương 0.20
    
    # 2. Dựng ROC từ nhãn và dự đoán đã lật
    fpr, tpr, _ = roc_curve(v_gt, v_pred)
    
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    
    # 3. Tìm điểm cắt (cutoff) tại giới hạn max_fpr và Nội suy tuyến tính
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr_cutoff = np.interp(max_fpr, x_interp, y_interp)
    
    # 4. Khâu lại đường cong & Tính Raw Area
    fpr_trunc = np.append(fpr[:stop], max_fpr)
    tpr_trunc = np.append(tpr[:stop], tpr_cutoff)
    partial_auc = auc(fpr_trunc, tpr_trunc)
    
    return partial_auc


@torch.no_grad()
def evaluate(model, test_loader, device, num_classes, best_t=0.5):
    """Full evaluation: metrics, plots, and uncertainty analysis."""
    model.eval()

    all_targets  = []
    all_preds    = []
    all_confs    = []
    all_probs    = []
    all_u_e      = []
    all_u_a      = []

    for data in test_loader:
        if len(data) == 3:
            inputs, tab_inputs, targets = data
            inputs, tab_inputs, targets = inputs.to(device), tab_inputs.to(device), targets.to(device)
        else:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            tab_inputs = None
            
        evidence = model(inputs, tab_inputs)
        unc = compute_uncertainties(evidence)

        alpha = unc['alpha']
        S     = unc['S']
        p_hat = (alpha / S).cpu().numpy()
        preds = p_hat.argmax(axis=1)
        confs = p_hat.max(axis=1)

        all_targets.append(targets.cpu().numpy())
        all_preds.append(preds)
        all_confs.append(confs)
        all_probs.append(p_hat)
        all_u_e.append(unc['epistemic'].cpu().numpy()[:, 0])
        all_u_a.append(unc['aleatoric'].cpu().numpy()[:, 0])

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    confs  = np.concatenate(all_confs)
    probs  = np.concatenate(all_probs, axis=0)
    u_e    = np.concatenate(all_u_e)
    u_a    = np.concatenate(all_u_a)
    correct = (y_pred == y_true).astype(float)

    # ── Scalar Metrics (Default Threshold = 0.5) ───────────────────
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    if num_classes == 2:
        macro_auroc = roc_auc_score(y_true, probs[:, 1], average='macro')
        pr_auc = average_precision_score(y_true, probs[:, 1])
        brier = brier_score_loss(y_true, probs[:, 1])
        try:
            # Dùng hàm Raw pAUC chuẩn Kaggle thay cho sklearn
            pauc = compute_isic_pauc(y_true, probs[:, 1], min_tpr=0.80)
        except Exception as e:
            print(f"Lỗi khi tính pAUC: {e}")
            pauc = float('nan')
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        f2_score = (5 * tp) / (5 * tp + 4 * fn + fp + 1e-8)
        
        from sklearn.metrics import recall_score, precision_score
        
        # ---------------------------------------------------------
        # THRESHOLD APPLICATION (from Validation Set)
        # ---------------------------------------------------------
        # [FIX: Data Leakage] Apply the frozen best_t evaluated on the Validation Set
        print(f"\n🔍 Using Pre-calculated Optimal Threshold = {best_t:.4f}")
        # Áp dụng ngưỡng tối ưu
        y_pred_opt = (probs[:, 1] >= best_t).astype(int)
        bal_acc_opt = balanced_accuracy_score(y_true, y_pred_opt)
        macro_f1_opt = f1_score(y_true, y_pred_opt, average='macro')
        tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_true, y_pred_opt).ravel()
        sensitivity_opt = tp_opt / (tp_opt + fn_opt + 1e-8)
        specificity_opt = tn_opt / (tn_opt + fp_opt + 1e-8)
        f2_score_opt = (5 * tp_opt) / (5 * tp_opt + 4 * fn_opt + fp_opt + 1e-8)
    else:
        macro_auroc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
        pauc = float('nan')
        pr_auc = float('nan')
        brier = float('nan')
        sensitivity = float('nan')
        specificity = float('nan')
        f2_score = float('nan')
        best_t = 0.5
        bal_acc_opt = bal_acc
        macro_f1_opt = macro_f1
        sensitivity_opt = float('nan')
        specificity_opt = float('nan')
        f2_score_opt = float('nan')
        y_pred_opt = y_pred

    ece_val, bin_accs, bin_confs, bin_sizes = compute_ece(confs, correct)

    # Minority-ECE (class 1 = malignant)
    # :math: \text{ECE}_{minority} = \sum_{m=1}^M \frac{|B_m|}{N_{minority}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
    minority_mask = (y_true == 1)
    if minority_mask.sum() > 0:
        m_ece, _, _, _ = compute_ece(confs[minority_mask], correct[minority_mask])
    else:
        m_ece = float('nan')

    # ── Print Results ──────────────────────────────────────────────
    print("\n📈 Evaluation Results (Threshold = 0.5)")
    print("=" * 50)
    print(f"  Balanced Accuracy     : {bal_acc:.4f}")
    print(f"  Macro F1-Score        : {macro_f1:.4f}")
    if num_classes == 2:
        print(f"  Sensitivity (Recall)  : {sensitivity:.4f}")
        print(f"  Specificity           : {specificity:.4f}")
        print(f"  F2-Score              : {f2_score:.4f}")
        print(f"  PR-AUC                : {pr_auc:.4f}")
        print(f"  Brier Score           : {brier:.4f}")
    print(f"  Macro-AUROC           : {macro_auroc:.4f}")
    print(f"  pAUC (@ 20% FPR)      : {pauc:.4f}")
    print(f"  ECE (15 bins)         : {ece_val:.4f}")
    print(f"  Minority-ECE (cls 1)  : {m_ece:.4f}")
    print(f"  Mean Epistemic u_e    : {u_e.mean():.4f}")
    print(f"  Mean Aleatoric u_a    : {u_a.mean():.4f}")
    print("=" * 50)

    if num_classes == 2:
        print(f"\n📈 Evaluation Results (Optimized Threshold = {best_t:.4f})")
        print("=" * 50)
        print(f"  Balanced Accuracy     : {bal_acc_opt:.4f}")
        print(f"  Macro F1-Score        : {macro_f1_opt:.4f}")
        print(f"  Sensitivity (Recall)  : {sensitivity_opt:.4f}")
        print(f"  Specificity           : {specificity_opt:.4f}")
        print(f"  F2-Score              : {f2_score_opt:.4f}")
        print("=" * 50)

    # ── Plots ──────────────────────────────────────────────────────
    plot_reliability_diagram(bin_accs, bin_confs, bin_sizes)
    plot_uncertainty_histogram(
        u_e[correct.astype(bool)],
        u_e[~correct.astype(bool)],
    )
    if num_classes == 2:
        plot_pr_curve(y_true, probs[:, 1])
    plot_risk_coverage_curve(y_true, y_pred_opt, confs)

    return {
        'balanced_accuracy': bal_acc,
        'balanced_accuracy_opt': bal_acc_opt,
        'sensitivity_opt': sensitivity_opt,
        'specificity_opt': specificity_opt,
        'macro_auroc': macro_auroc,
        'pauc': pauc,
        'ece': ece_val,
        'minority_ece': m_ece,
        'mean_u_e': float(u_e.mean()),
        'mean_u_a': float(u_a.mean()),
    }


def export_to_tensorrt_sparse(model, save_path='model_sparse.pth'):
    """
    [FIX: Sparse Export] Physically converts the dense weights of MDEP layers into 
    actual 2:4 sparse tensors for accelerated inference using torch.sparse.
    Requires PyTorch >= 2.1.
    """
    print("\n📦 Exporting Model to Physical 2:4 Sparse Tensors...")
    
    # [FIX: Hardware Safeguard] Check for PyTorch >= 2.1 and Ampere+ GPU (compute capability >= 8.0)
    device = next(model.parameters()).device
    if device.type == 'cuda':
        major, _ = torch.cuda.get_device_capability(device)
        if major < 8:
            print(f"⚠ WARNING: 2:4 sparse export requires Ampere+ GPU (compute capability >= 8.0). Detected capability: {major}.x")
            print("⚠ Skipping physical sparse export to avoid crash. (Model will still be logically sparse)")
            return
    else:
        print("⚠ WARNING: 2:4 sparse export requires CUDA device. Skipping.")
        return

    try:
        from torch.sparse import to_sparse_semi_structured
        for name, module in model.named_modules():
            if isinstance(module, (MDEPLinear, MDEPConv2d)):
                # Apply the current mask to the weight first
                effective_weight = module.weight * getattr(module, 'cached_mask', module.mask)
                # Convert to physical 2:4 sparsity (semi-structured)
                # Only supported for FP16/BF16/INT8, so we convert to half precision first
                sparse_weight = to_sparse_semi_structured(effective_weight.half())
                # Replace the weight parameter
                module.weight = nn.Parameter(sparse_weight)
                
                # Remove MDEP specific attributes to clean up the exported model
                for attr in ['scores', 'mask', 'scores_momentum', 'cached_mask']:
                    if hasattr(module, attr):
                        delattr(module, attr)
                        
        torch.save(model.state_dict(), save_path)
        print(f"✅ Real 2:4 Sparse Model exported successfully to: {save_path}")
    except Exception as e:
        print(f"⚠️ Failed to export to 2:4 sparse tensor. Error: {e}")

# ============================================================================
#  SECTION 7 — main()
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥  Device: {device}")

    # KÍCH HOẠT CHẾ ĐỘ TĂNG TỐC HARDWARE CHO CONVOLUTION
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # ── Data (stratified train / test split) ────────────────────────
    train_loader, val_loader, test_loader, num_classes, class_weights, class_counts, tabular_dim = get_isic_dataloaders(batch_size=32)
    
    # HÀNH ĐỘNG 1: Xóa WeightedRandomSampler (Chống Double-Dipping)
    # Vẫn giữ nguyên giới hạn epoch_steps = 64,000 để chống Kaggle Timeout
    from torch.utils.data import RandomSampler
    
    train_dataset = train_loader.dataset
    epoch_steps = min(64000, len(train_dataset))
    # Dùng RandomSampler cơ bản, KHÔNG dùng trọng số thiên vị
    sampler = RandomSampler(train_dataset, replacement=True, num_samples=epoch_steps)
    
    def h5_worker_init_fn(worker_id):
        import h5py
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        if hasattr(dataset, 'hdf5_path') and getattr(dataset, 'HAS_H5PY', globals().get('HAS_H5PY', False)):
            dataset._hdf5_file = h5py.File(dataset.hdf5_path, 'r')
            
    # [FIX: Kaggle OOM Risk] Reduce num_workers
    # Fix HDF5 Deadlock & Memory Leak trên Kaggle: Ép chạy Main Thread
    # Tối ưu Bottleneck: Kaggle có 4 CPU cores, dùng 2 luồng là tối ưu nhất.
    nw = 2
    train_loader = DataLoader(
        train_dataset, batch_size=32, sampler=sampler, 
        num_workers=nw, pin_memory=True, drop_last=True, worker_init_fn=h5_worker_init_fn, prefetch_factor=2
    )

    print(f"📊 Classes: {num_classes}")

    # ── Model: DenseNet-121 with Multi-Modal EDL head ──────────────────────────
    base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model = MultimodalDenseNet(base_model, tabular_dim, num_classes)
    
    nn.init.normal_(model.classifier[0].weight, mean=0, std=0.001)
    
    with torch.no_grad():
        model.classifier[0].bias[0] = 0.0
        model.classifier[0].bias[1] = 0.0
        
    replace_conv2d_with_mdep(model)
    model = model.to(device)
    
    # --- Hành động 1: Xóa DataParallel, thêm cảnh báo ---
    print("✅ Running in Single-GPU Safe Mode to prevent Autograd Deadlocks.")
        
    # ── CẤU HÌNH LỘ TRÌNH 40 EPOCH (CHIA ĐÔI CHẶNG ĐƯỜNG) ──
    total_epochs = 40
    warmup_epochs = 5
    
    # ĐIỀU KHIỂN TRẠM DỪNG (CHỈNH SỬA TẠI ĐÂY CHO TỪNG PHIÊN):
    # Phiên 1: Gán = 20 (Chạy từ 0 đến 20 rồi dừng và lưu)
    # Phiên 2: Gán = 40 (Nạp checkpoint phiên 1, chạy tiếp từ 20 đến 40)
    target_stop_epoch = 20  

    start_epoch = 0
    best_pauc = 0.0

    # ── Optimizer & Loss ───────────────────────────────────────────
    criterion = EvidentialFocalLoss(
        class_counts=class_counts,  # Đã gỡ bỏ hardcode [1, 1]
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        lambda_kl=0.001
    )
    # Truyền trọng số vào hàm Loss
    criterion.class_weights = class_weights 
    # Khắc phục lỗi Optimizer Hijacking: chặn 'scores' khỏi AdamW
    trainable_params = [p for name, p in model.named_parameters() if 'scores' not in name]
    base_optimizer = optim.AdamW
    optimizer = SAM(trainable_params, base_optimizer, rho=0.05, lr=1e-4, weight_decay=1e-3)

    # Khởi tạo OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer.base_optimizer,  # <--- BẮT BUỘC DÙNG BASE_OPTIMIZER ĐỂ TRÁNH CẢNH BÁO SAM
        max_lr=3e-4,
        epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0
    )

    trainer = MDEPTrainer(model, optimizer, criterion, total_epochs, warmup_epochs, scheduler=scheduler)

    # ── MLOPS RESUME: Tìm kiếm file checkpoint từ phiên chạy trước ──
    checkpoint_path = None
    possible_paths = [
        '/kaggle/working/model_checkpoint.pth',
        './model_checkpoint.pth',
    ]
    # Tự động quét toàn bộ thư mục Input nếu bạn thêm Output của phiên trước vào Data
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'model_checkpoint.pth' in files:
                possible_paths.append(os.path.join(root, 'model_checkpoint.pth'))

    for path in possible_paths:
        if os.path.exists(path):
            checkpoint_path = path
            print(f"🔄 Tìm thấy file Checkpoint tại: {checkpoint_path}")
            break

    if checkpoint_path is not None:
        print(f"📥 Đang tiến hành phục hồi tri thức từ trạm cũ: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_pauc = checkpoint.get('best_pauc', 0.0)
            print(f"🟢 Phục hồi thành công! Sẽ tiếp tục huấn luyện từ Epoch {start_epoch}")
        except Exception as e:
            print(f"⚠ Không thể nạp toàn bộ trạng thái bộ tối ưu, chỉ nạp trọng số gốc. Lỗi: {e}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
    # -----------------------------------------------------------

    # --- WANDB INITIALIZATION & LOGIN ---
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
        wandb.login(key=wandb_api_key)
    except Exception:
        wandb.login()

    wandb.init(
        project="ISIC-2024-MDEP",
        name="DenseNet121-MDEP-SAM",
        config={
            "architecture": "DenseNet121",
            "epochs": total_epochs,
            "warmup_epochs": warmup_epochs,
            "batch_size": 32,
            "optimizer": "SAM",
            "learning_rate": 3e-4,
            "sparsity": "2:4 Dynamic"
        }
    )

    # ── Training ───────────────────────────────────────────────────
    print("\n🚀 Starting Training (MDEP Framework)")
    print("=" * 60)

    # --- KAGGLE MLOPS: Graceful Timeout & Chunked Training ---
    import time
    session_start_time = time.time()
    max_session_hours = 11.5  # Nới rộng vùng đệm an toàn lên 1.5 tiếng
    

    
    print(f"⏱ Session Timer Started: Max {max_session_hours}h or stop safely at Epoch {target_stop_epoch}.")
    
    # --- CSV Logger ---
    
    csv_log_path = 'training_logs.csv'
    if start_epoch == 0 or not os.path.exists(csv_log_path):
        with open(csv_log_path, 'w') as f:
            f.write("Epoch,Phase,Gamma,Loss\n")

    for epoch in range(start_epoch, total_epochs):
        loss = trainer.train_epoch(epoch, train_loader, device)
        phase = "Warm-up (Dense)" if epoch < warmup_epochs else "Dynamic 2:4 Sparsity"
        gamma = trainer.step_gamma(epoch)
        print(
            f"  Epoch [{epoch+1:>2}/{total_epochs}]  "
            f"| Phase: {phase:<22} "
            f"| γ: {gamma:.4f}  "
            f"| Loss: {loss:.4f}"
        )
        
        # Ghi log ra file CSV
        with open(csv_log_path, 'a') as f:
            f.write(f"{epoch+1},{phase},{gamma:.4f},{loss:.4f}\n")
            
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train/epoch_loss": loss,
                "train/gamma": gamma
            })
        
        # ── LƯU CHECKPOINT ĐÓNG GÓI TOÀN DIỆN ĐỂ TIẾP SỨC ──
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'best_pauc': best_pauc,
        }
        torch.save(checkpoint_data, 'model_checkpoint.pth')
        print(f"💾 Đã đóng gói và lưu trạng thái tại Epoch {epoch} vào model_checkpoint.pth")
            
        # --- KIỂM TRA ĐIỀU KIỆN HẠ CÁNH MỀM ---
        if (epoch + 1) >= target_stop_epoch:
            print(f"\n🛑 Đã đạt mốc Epoch {target_stop_epoch} theo kế hoạch. Đã lưu Checkpoint an toàn!")
            print("Chủ động ngắt phiên để chuẩn bị cho chặng tiếp theo.")
            break
            
        elapsed_hours = (time.time() - session_start_time) / 3600.0
        if elapsed_hours > max_session_hours:
            print(f"\n⏳ CẢNH BÁO: Đã chạy {elapsed_hours:.2f}h (Sắp hết 12h của Kaggle). Tiến hành hạ cánh mềm để bảo toàn Checkpoint!")
            break

    print("=" * 60)
    print("✅ Training complete.\n")

    # ── Evaluation ─────────────────────────────────────────────────
    # [FIX: Data Leakage] Find optimal threshold on the validation set
    best_t = find_optimal_threshold(model, val_loader, device) if num_classes == 2 else 0.5
    eval_metrics = evaluate(model, test_loader, device, num_classes, best_t=best_t)
    print_sparsity_report(model)
    
    if wandb.run is not None:
        wandb.log({"eval/" + k: v for k, v in eval_metrics.items()})

    # ── Saving Model ───────────────────────────────────────────────
    model_save_path = 'model_checkpoint.pth'
    checkpoint_data = {
        'epoch': epoch if 'epoch' in locals() else start_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_pauc': best_pauc,
    }
    torch.save(checkpoint_data, model_save_path)
    print("=" * 60)
    print(f"💾 Tải trọng số mô hình đã được lưu tại: {model_save_path}")
    print("   (Bạn có thể tải file này về từ tab 'Output' trên Kaggle)")
    print("=" * 60)

    # ── Export 2:4 Sparse Model ────────────────────────────────────
    # [FIX: Sparse Export] Physically convert weights to 2:4 sparse tensors
    sparse_save_path = 'model_sparse_2_4.pth'
    export_to_tensorrt_sparse(model, sparse_save_path)
    
    if wandb.run is not None:
        # Tự động tải bản Checkpoint và bản Sparse lên Artifacts của Wandb
        wandb.save('model_checkpoint.pth')
        wandb.save(sparse_save_path)
        wandb.finish()



# ============================================================================
#  SECTION 8 — Ablation Study Harness (uncomment to run)
# ============================================================================
#
# def update_scores_ablation(model, beta=1.0, mode='full'):
#     """
#     Ablation wrapper around update_scores_agents.
#       mode='full'       → both Microglia + Astrocyte (default)
#       mode='prune_only' → only Microglia scoring (G_ij = 0)
#       mode='grow_only'  → only Astrocyte scoring (C_ij = 0)
#     """
#     for module in model.modules():
#         if isinstance(module, (MDEPLinear, MDEPConv2d)):
#             if not hasattr(module, 'grad_L_w'):
#                 continue
#             w_val = module.weight.data
#             # Microglia
#             c1 = torch.abs(w_val * module.grad_L_w)
#             c1_min = c1.min()
#             c1_max = c1.max()
#             c1_norm = (c1 - c1_min) / (c1_max - c1_min + 1e-8)
#             c2 = torch.abs(w_val * getattr(module, 'grad_ua_w', torch.zeros_like(w_val)))
#             c2_min = c2.min()
#             c2_max = c2.max()
#             c2_norm = (c2 - c2_min) / (c2_max - c2_min + 1e-8)
#             C_ij = c1_norm + beta * c2_norm
#             # Astrocyte
#             u_e_node = getattr(module, 'u_e_node', None)
#             if u_e_node is not None:
#                 if isinstance(module, MDEPLinear):
#                     g1 = u_e_node.unsqueeze(1).expand_as(w_val)
#                 elif isinstance(module, MDEPConv2d):
#                     g1 = u_e_node.view(-1, 1, 1, 1).expand_as(w_val)
#                 else:
#                     g1 = torch.zeros_like(w_val)
#             else:
#                 g1 = torch.zeros_like(w_val)
#             g1_norm = g1 / (g1.max() + 1e-8)
#             g2 = torch.abs(module.grad_L_w)
#             g2_norm = g2 / (g2.max() + 1e-8)
#             G_ij = g1_norm * g2_norm
#             # Apply ablation
#             if mode == 'prune_only':
#                 G_ij = torch.zeros_like(G_ij)
#             elif mode == 'grow_only':
#                 C_ij = torch.zeros_like(C_ij)
#             delta_S = C_ij + G_ij
#             beta_m = 0.9
#             module.scores_momentum.data.mul_(beta_m).add_(delta_S, alpha=1.0 - beta_m)
#             eta = 0.1
#             module.scores.data.add_(module.scores_momentum.data, alpha=eta)
#             module.scores.data.sub_(module.scores.data.mean())
#
# # To run an ablation study, uncomment and call:
# # for mode in ['prune_only', 'grow_only', 'full']:
# #     print(f"\n{'='*60}\n  ABLATION: {mode}\n{'='*60}")
# #     <rebuild model, train with update_scores_ablation(..., mode=mode), evaluate>


# ── Run ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
