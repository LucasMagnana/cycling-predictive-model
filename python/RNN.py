import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = 32
        self.i2h = nn.Linear(input_size + self.hidden_size, self.hidden_size)
        self.i2o = nn.Linear(input_size + self.hidden_size, self.hidden_size)

        self.i2h_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.i2o_1 = nn.Linear(self.hidden_size, output_size)

        self.i2h_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.i2o_2 = nn.Linear(self.hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)

        output = self.i2o_1(output)
        #hidden = self.i2h_1(hidden)

        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)



class RNN_LSTM(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional, dropout):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_size.
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional, 
                           dropout=dropout)

        # The linear layer that maps from hidden state space to tag space
        self.out1 = nn.Linear(self.hidden_size+self.bidirectional*self.hidden_size, self.hidden_size//2)
        self.out2 = nn.Linear(self.hidden_size//2, output_size)
        self.out3 = nn.Linear(self.hidden_size//2, output_size)

        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.dropout(self.out1(output.squeeze(0)))
        output = self.out2(output)
        #output = self.out3(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        #return torch.zeros(1, self.hidden_size).unsqueeze(0)
        
        return (torch.randn(self.num_layers+self.bidirectional*self.num_layers, 1, self.hidden_size), torch.randn(self.num_layers+self.bidirectional*self.num_layers, 1, self.hidden_size))