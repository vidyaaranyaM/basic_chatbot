import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self,
                 inputSize,
                 h1Size,
                 h2Size,
                 outputSize) -> None:
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputSize, h1Size),
            nn.ReLU(),
            nn.Linear(h1Size, h2Size),
            nn.ReLU(),
            nn.Linear(h2Size, outputSize)
        )
    
    def forward(self, input):
        return self.model(input)
