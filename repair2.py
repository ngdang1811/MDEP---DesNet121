import re

with open('c:/Users/ADMIN/Downloads/DesNet121/mdep_densenet_refactored.py', 'r', encoding='utf-8') as f:
    code = f.read()

bad_block = r'''            # Step 3: Delete redundant variables
            module.scores_momentum.data.mul_(beta_m).add_(delta_S, alpha=1.0 - beta_m)
            
            # Step 2: Update Latent Scores S
            eta = 0.02
            module.scores.data.add_(module.scores_momentum.data, alpha=eta)'''
            
good_block = r'''            # Step 3: Delete redundant variables
            del c1_norm, c2_norm, C_ij, G_ij, delta_S'''

code = code.replace(bad_block, good_block)

with open('c:/Users/ADMIN/Downloads/DesNet121/mdep_densenet_refactored.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('Repairs complete.')
