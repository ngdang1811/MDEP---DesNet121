import torch
import torch.nn as nn
import torchvision.models as models

class EvidenceLayer(nn.Module):
    def __init__(self, activation='softplus'):
        super().__init__()
        self.activation = activation
    def forward(self, x):
        return F.softplus(x) if self.activation == 'softplus' else x

def test():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    in_features = model.classifier.in_features
    print(f"Original in_features: {in_features}")
    
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 2),
        EvidenceLayer(activation='softplus')
    )
    nn.init.normal_(model.classifier[0].weight, mean=0, std=0.001)
    nn.init.constant_(model.classifier[0].bias, 0)
    print("Classifier successfully modified!")
    
    # Check that replace_conv2d_with_mdep logic works (mocking)
    conv_count = 0
    linear_count = 0
    def count_layers(m):
        nonlocal conv_count, linear_count
        for name, child in m.named_children():
            if isinstance(child, nn.Conv2d):
                conv_count += 1
            elif isinstance(child, nn.Linear):
                linear_count += 1
            else:
                count_layers(child)
    
    count_layers(model)
    print(f"Found {conv_count} Conv2d layers and {linear_count} Linear layers to replace.")

if __name__ == '__main__':
    test()
