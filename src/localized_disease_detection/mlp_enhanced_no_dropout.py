from torch import nn

class MLP_enhanced_no_dropout(nn.Module):
    """Enhanced multi-layer perceptron (MLP) with Batch Normalization and Dropout."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.5):
        super(MLP_enhanced_no_dropout, self).__init__()
        self.num_layers = num_layers
        
        layers = []
        # Define each layer with BatchNorm and Dropout
        for i in range(num_layers):
            if i == 0:
                # First layer
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i < num_layers - 1:
                # Hidden layers
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                # Output layer
                layers.append(nn.Linear(hidden_dim, output_dim))
            
            if i < num_layers - 1:  # No BatchNorm or ReLU or Dropout on output layer
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                # layers.append(nn.Dropout(dropout_rate))
        
        # Using ModuleList to hold all layers
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x