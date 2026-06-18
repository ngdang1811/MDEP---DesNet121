import re

with open('c:/Users/ADMIN/Downloads/DesNet121/mdep_densenet_refactored.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Fix the Eta update logic replacement
bad_eta_block = r'''            if G_ij.max().item() <= 1e-8:
                # -- Microglia Pruning Action --
                # Accumulate the pruned signal with high momentum
                beta_m = 0.95
                module.scores_momentum.data.mul_(beta_m).add_(delta_S, alpha=1.0 - beta_m)
                
                # Apply update to scores via gradient ascent (add, not subtract)
                # Ensure Multi-Agent Algorithmic Convergence (Eta Decay Schedule)
                progress = epoch / max(total_epochs - 1, 1)
                eta = 0.001 + 0.5 * (0.05 - 0.001) * (1 + math.cos(math.pi * progress))
                module.scores.data.add_(module.scores_momentum.data, alpha=eta)
            
            # Step 1: Update Velocity (Momentum EMA)
            beta_m = 0.95'''
            
good_eta_block = r'''            if G_ij.max().item() <= 1e-8:
                noise = 0.0316 * torch.randn_like(G_ij) * g1_norm
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
            
            # Step 3: Delete redundant variables'''

code = code.replace(bad_eta_block, good_eta_block)
code = re.sub(
    r'''                module\.scores\.data\.add_\(module\.scores_momentum\.data, alpha=eta\)
            
            # Step 1: Update Velocity \(Momentum EMA\)
            beta_m = 0\.95
            module\.scores_momentum\.data\.mul_\(beta_m\)\.add_\(delta_S, alpha=1\.0 - beta_m\)
            
            # Step 2: Apply update to scores via gradient ascent
            eta = 0\.02
            module\.scores\.data\.add_\(module\.scores_momentum\.data, alpha=eta\)''',
    r'''                module.scores.data.add_(module.scores_momentum.data, alpha=eta)''',
    code, flags=re.DOTALL
)


# Fix scaled_loss issue
bad_loss_block = r'''            # Loss scaling to counteract Focal Loss shrinkage (decayed)
            
            self.scaler.scale(scaled_loss).backward()'''
good_loss_block = r'''            # Loss scaling removed to resolve conflict with AMP
            self.scaler.scale(loss).backward()'''

code = code.replace(bad_loss_block, good_loss_block)

with open('c:/Users/ADMIN/Downloads/DesNet121/mdep_densenet_refactored.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('Repairs complete.')
