import torch.nn as nn

class Model(nn.Module):
    # Fully connected MLP 
    def __init__(self, config):
        super(Model, self).__init__()
        n_hidden = config['n_hidden']
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        output_size = config['output_size']
        
        # number of layers: 2 for input, 2 for each hidden, 1 for output
        self.layers = nn.ModuleList([None] * (2 + 2*n_hidden + 1))
        
        # INPUT
        self.layers[0] = nn.Linear(input_size, hidden_size)
        self.layers[1] = nn.Tanh()
        
        # HIDDEN
        for i in range(1,1+n_hidden):
            self.layers[2*i+0] = nn.Linear(hidden_size, hidden_size)
            self.layers[2*i+1] = nn.Tanh()
        
        # OUTPUT
        self.layers[(1+n_hidden)*2] = nn.Linear(hidden_size, output_size)
        
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out) 
        return out
