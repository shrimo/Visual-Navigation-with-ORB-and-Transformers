import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        x = self.embedding(src)
        x = self.transformer(x, x)
        x = self.fc_out(x)
        return x

def load_model(model_path, input_dim, model_dim, num_heads, num_layers, output_dim, device):
    model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model