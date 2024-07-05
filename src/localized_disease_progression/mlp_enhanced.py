from torch import nn

class MLP_enhanced(nn.Module):
    """Enhanced multi-layer perceptron (MLP) with Batch Normalization and Dropout."""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=128, dropout_rate=0.5, apply_to_output=False):
        super(MLP_enhanced, self).__init__()
        self.num_layers = num_layers
        self.apply_to_output = apply_to_output
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i < num_layers - 1:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, output_dim))
            
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout_rate))
                
            if (i == num_layers - 1 and apply_to_output):
                layers.append(nn.BatchNorm1d(output_dim))
                layers.append(nn.ReLU(inplace=True))
                # layers.append(nn.Dropout(dropout_rate))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # print(f'x: {x.shape}')
        return x