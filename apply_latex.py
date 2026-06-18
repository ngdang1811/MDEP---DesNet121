import os

files_to_update = [
    "mdep_agents.py",
    "mdep_densenet_notebook.py",
    "mdep_densenet_ablation_grow_only.py",
    "mdep_densenet_ablation_prune_only.py"
]

def replace_in_file(filepath, replacements):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
        else:
            print(f"Warning: could not find target string in {filepath}:\n{old[:100]}...")
            
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# 1. Update update_scores_agents (Feature Flow, Jacobian, Microglia, Astrocyte)
for f in files_to_update:
    reps = []
    
    # Feature Flow Docstring
    old_ff1 = """            # --- Feature Flow Docstring ---
            # Cơ chế truyền thuận của mạng thưa động DenseNet:
            # h^(l) = \phi ( W^(l) [h^(0), h^(1), ..., h^(l-1)] + b^(l) )
            # Phép nối tensor (Concatenation) dọc chiều kênh giúp mọi lớp tích chập nhận gradient 
            # trực tiếp từ Loss, chống triệt tiêu đạo hàm.
            # ----------------------------------------------"""
    new_ff1 = """            # --- Feature Flow ---
            # :math: h^{(l)} = \phi( \tilde{W}^{(l)} [h^{(0)}, \dots, h^{(l-1)}] + b^{(l)} )
            # Phép nối tensor (Concatenation) dọc chiều kênh giúp mọi lớp tích chập nhận gradient 
            # trực tiếp từ Loss, chống triệt tiêu đạo hàm.
            # ----------------------------------------------"""
            
    # For files that didn't get the old docstring yet (like prune_only)
    old_ff2 = """            w_val = module.weight.data

            # --- Microglia agent"""
    new_ff2 = """            # --- Feature Flow ---
            # :math: h^{(l)} = \phi( \tilde{W}^{(l)} [h^{(0)}, \dots, h^{(l-1)}] + b^{(l)} )
            # Phép nối tensor (Concatenation) dọc chiều kênh giúp mọi lớp tích chập nhận gradient 
            # trực tiếp từ Loss, chống triệt tiêu đạo hàm.
            # ----------------------------------------------
            w_val = module.weight.data

            # --- Microglia agent"""
            
    reps.append((old_ff1, new_ff1))
    # Note: apply old_ff2 only if old_ff1 isn't there, handled by order or string check
    
    # Astrocyte Jacobian Docstring
    old_jac = """            # --- Astrocyte Jacobian Docstring ---
            # Tính u_{e,i}^{(node)} = | \partial u_e / \partial z_i^{(l)} |
            # Nhờ cơ chế Concatenation của DenseNet, kích hoạt z_i^{(l)} được nối thẳng vào 
            # lớp Global Average Pooling và đi thẳng vào EvidenceLayer (khoảng cách 1 bước). 
            # Ma trận Jacobian J = \partial u_e / \partial z^{(l)} giữ nguyên được toàn vẹn 
            # giá trị (full rank) mà không bị pha loãng như ResNet. Điều này giúp Astrocyte 
            # cực kỳ nhạy bén với tín hiệu lớp thiểu số.
            # ----------------------------------------------------"""
    new_jac = """            # --- Astrocyte Jacobian ---
            # :math: \mathbf{J} = \\frac{\partial u_e}{\partial z^{(l)}}
            # Nhờ cơ chế Concatenation của DenseNet, kích hoạt z_i^{(l)} được nối thẳng vào 
            # lớp Global Average Pooling và đi thẳng vào EvidenceLayer (khoảng cách 1 bước). 
            # Ma trận Jacobian giữ nguyên được toàn vẹn giá trị (full rank) mà không bị pha loãng như ResNet.
            # Điều này giúp Astrocyte cực kỳ nhạy bén với tín hiệu lớp thiểu số.
            # ----------------------------------------------------"""
    reps.append((old_jac, new_jac))
    
    # Microglia g2 logic (use grad_L_S)
    old_mic1 = """            # c1: importance for prediction = |w * ∂L_EFL/∂w|
            c1 = torch.abs(w_val * module.grad_L_w)"""
    new_mic1 = """            if not hasattr(module, 'grad_L_S'):
                module.grad_L_S = module.scores.grad if module.scores.grad is not None else torch.zeros_like(module.scores)
            
            # c1: importance for prediction = |w * ∂L_EFL/∂S|
            c1 = torch.abs(w_val * module.grad_L_S)"""
    reps.append((old_mic1, new_mic1))
    
    # Astrocyte g2 logic was already partially done, but wait: we defined grad_L_S there. 
    # Since we define it in Microglia now, we don't need to define it again in Astrocyte.
    # Let's clean up Astrocyte g2.
    old_ast1 = """            # g2: per-weight loss gradient magnitude = |∂L_EFL/∂S_ij|
            if not hasattr(module, 'grad_L_S'):
                module.grad_L_S = module.scores.grad if module.scores.grad is not None else torch.zeros_like(module.scores)
            g2 = torch.abs(module.grad_L_S)"""
    new_ast1 = """            # g2: per-weight loss gradient magnitude = |∂L_EFL/∂S_ij|
            g2 = torch.abs(module.grad_L_S)"""
    reps.append((old_ast1, new_ast1))
    
    replace_in_file(f, reps)
    
    # For prune_only, add old_ff2 if needed
    with open(f, 'r', encoding='utf-8') as f_read:
        content = f_read.read()
    if "# --- Feature Flow ---" not in content and "# --- Feature Flow Docstring ---" not in content:
        replace_in_file(f, [(old_ff2, new_ff2)])

# 2. Update losses.py
losses_reps = [
    (
        "        # Cross entropy term: sum_c y_c * (digamma(S) - digamma(alpha_c))\n        loss_ce = torch.sum(targets * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)",
        "        # Cross entropy term: sum_c y_c * (digamma(S) - digamma(alpha_c))\n        # :math: \mathcal{L}_{CE} = \sum_{c=1}^K y_c \\left( \psi(S_\\alpha) - \psi(\\alpha_c) \\right)\n        loss_ce = torch.sum(targets * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)"
    ),
    (
        "        # KL Divergence Regularization\n        alpha_tilde = targets + (1 - targets) * alpha\n        loss_kl = kl_divergence(alpha_tilde, self.num_classes)",
        "        # KL Divergence Regularization\n        # :math: \mathcal{L}_{KL} = \log\Gamma\left(\sum_{c=1}^K \\tilde{\\alpha}_c\\right) - \sum_{c=1}^K \log\Gamma(\\tilde{\\alpha}_c) + \sum_{c=1}^K (\\tilde{\\alpha}_c - 1)\\left[ \psi(\\tilde{\\alpha}_c) - \psi\left(\sum_{c=1}^K \\tilde{\\alpha}_c\\right) \\right]\n        alpha_tilde = targets + (1 - targets) * alpha\n        loss_kl = kl_divergence(alpha_tilde, self.num_classes)"
    )
]
replace_in_file("losses.py", losses_reps)

# 3. Update evaluation metrics in Notebook and Ablation scripts
notebook_files = [
    "mdep_densenet_notebook.py",
    "mdep_densenet_ablation_grow_only.py",
    "mdep_densenet_ablation_prune_only.py"
]

for nf in notebook_files:
    n_reps = [
        # F2-Score Optimization
        (
            "        # Optimize threshold for Balanced Accuracy\n        best_t = 0.5\n        best_bal_acc = 0.0\n        for t in np.linspace(0.01, 0.99, 99):\n            y_pred_t = (probs[:, 1] >= t).astype(int)\n            bal_acc_t = balanced_accuracy_score(y_true, y_pred_t)\n            if bal_acc_t > best_bal_acc:\n                best_bal_acc = bal_acc_t\n                best_t = t",
            "        # Optimize threshold for F2-Score\n        # :math: F_2 = 5 \\times \\frac{\\text{Precision} \\times \\text{Sensitivity}}{4 \\times \\text{Precision} + \\text{Sensitivity}}\n        best_t = 0.5\n        best_f2 = 0.0\n        for t in np.linspace(0.01, 0.99, 99):\n            y_pred_t = (probs[:, 1] >= t).astype(int)\n            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, y_pred_t).ravel()\n            sens_t = tp_t / (tp_t + fn_t + 1e-8)\n            prec_t = tp_t / (tp_t + fp_t + 1e-8)\n            f2_t = (5 * prec_t * sens_t) / (4 * prec_t + sens_t + 1e-8)\n            if f2_t > best_f2:\n                best_f2 = f2_t\n                best_t = t"
        ),
        # Clinical decision docstring
        (
            "        y_pred_opt = (probs[:, 1] >= best_t).astype(int)",
            "        # :math: \hat{y}_{clinical} = 1 \\text{ nếu } \mathbb{E}[\pi_{malignant}] \ge \\tau_{opt} \\text{ ngược lại } 0\n        y_pred_opt = (probs[:, 1] >= best_t).astype(int)"
        ),
        # Minority-ECE docstring
        (
            "    # Minority-ECE (class 1 = malignant)\n    minority_mask = (y_true == 1)",
            "    # Minority-ECE (class 1 = malignant)\n    # :math: \\text{ECE}_{minority} = \sum_{m=1}^M \\frac{|B_m|}{N_{minority}} \left| \\text{acc}(B_m) - \\text{conf}(B_m) \\right|\n    minority_mask = (y_true == 1)"
        ),
        # MACs formula
        (
            "    macs_saved = (total_macs_dense - total_macs_sparse) / total_macs_dense * 100 if total_macs_dense > 0 else 0.0\n    print(\"-\" * 75)",
            "    macs_saved = (total_macs_dense - total_macs_sparse) / total_macs_dense * 100 if total_macs_dense > 0 else 0.0\n    print(\"-\" * 75)\n    print(\"  :math: \\text{MACs}_{sparse} = \\\\frac{1}{2} \sum_{l=1}^L \left( C_{in}^{(l)} \\\\times C_{out}^{(l)} \\\\times K^2 \\\\times H \\\\times W \\right) \\\\approx 50\% \\text{ MACs}_{dense}\")"
        )
    ]
    replace_in_file(nf, n_reps)

print("Done applying LaTeX modifications.")
