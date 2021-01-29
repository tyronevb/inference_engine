"""
Implements the AnomalyDetectorLSTM Class.

This class implements the LSTM-based Deep Learning Model using PyTorch.

"""

__author__ = "tyronevb"
__date__ = "2021"

import torch
import torch.nn as nn


class AnomalyDetectorLSTM(nn.Module):
    """Implements the LSTM Recurrent Neural Network Architecture for performing anomaly detection on log files."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
    ):
        """
        Initialise the Anomaly DetectorLSTM Class.

        :param input_size: how many features represent each input sample
        :param hidden_size: how many hidden units in the model
        :param output_size: how many features represent the output - should equal number of unique log events
        :param num_layers: how many LSTM layers
        :param batch_first: specify that data is given with the batch dimension first
        :param dropout: dropout parameter applied when using many LSTM layers
        :param bidirectional: specify if a Bidrectional LSTM is to be used
        """
        super(AnomalyDetectorLSTM, self).__init__()

        # create class attributes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size  # how many features in the output, for categorical = num classes
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1  # 2 if bidir LSTM used

        # create layers of the neural network
        # LSTM layers of the network
        self.lstm_layer = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # final linear / fully connected / dense layer that results in the output
        # todo: does the number of features here depend on the number of directions?
        self.linear_output_layer = nn.Linear(
            in_features=self.hidden_size * self.num_directions, out_features=self.output_size
        )

    def forward(self, input_data: torch.tensor, device: torch.device) -> torch.tensor:
        """
        Implement the forward pass function for the model.

        :param input_data: sequence of log keys
        :param device: device to train on
        :return: predicted values based on given input data
        """
        # get the batch size from the input_data
        # input_data is of size (batch_size, seq_len, input_size)
        batch_size = input_data.size(0)

        # move input data to device
        input_data_on_device = input_data.to(device)

        # initialise hidden and cell states
        h_0, c_0 = self.initialise_hidden_and_cell_states(batch_size=batch_size, device=device)

        output, hidden = self.lstm_layer(input_data_on_device, (h_0, c_0))

        # output is of size (batch_size, seq_len, num_directions*hidden_size)

        # only the final output from the LSTM layer(s) is passed to the dense layer
        # hence the slicing of the output below: output[:, -1, :]
        output = self.linear_output_layer(output[:, -1, :])

        return output

    def initialise_hidden_and_cell_states(self, batch_size: int, device: torch.device) -> (torch.tensor, torch.tensor):
        """
        Initialse the hidden and cell states of the LSTM network.

        :param batch_size: how many samples per batch
        :param device: device to train on
        :return: initialised hidden and cell state tensors
        """
        h_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)

        c_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        return h_0, c_0


# end
