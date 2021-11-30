import torch

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
  """
  A class to construct a RNN model
  Attributes
  ----------
      num_layers: int
            Number of layers RNN should consists of 
      num_classes: int
            Number of classes to predict
      hidden_size: int
            Size of a hidden layer
      input_size: int
            Size of input layer
  """
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        
  def forward(self, x):
        """Forward pass
        Parameters
        ----------
            x: matrix
                  Dataset matrix containing features
        Returns
        --------
            Prediction
        """
        # Set initial hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        out, _ = self.rnn(x, h0)
        # Decode the hidden state of the last time step
        out = out.reshape(-1, self.hidden_size)
         
        out = self.fc(out)
        return out