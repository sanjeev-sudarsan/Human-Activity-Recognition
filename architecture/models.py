from torch.nn import functional as F
from torch import nn
from .base_model import Classifier


class maxpool_lstm(Classifier):
    """
    This is a child class of the Classifier class.
    It has lstm layers coupled with a max pooling layer and fully connected layers.

    Attributes:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden units of the lstm.
        num_layers (int): The number of recurrent layers.
        num_classes (int): The number of classes.
        win_size (int): The size of the sliding window.
        LEARNING_RATE (float): The learning rate of the model.
        DROPOUT_RATE (float): The dropout rate of the model.
        REGULARIZATION (float): The regularization factor of the model.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        win_size,
        LEARNING_RATE,
        DROPOUT_RATE,
        REGULARIZATION,
        **kwargs
    ):
        """
        The constructor for the maxpool_lstm class.

        Parameters:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden units of the lstm.
            num_layers (int): The number of recurrent layers.
            num_classes (int): The number of classes.
            win_size (int): The size of the sliding window.
            LEARNING_RATE (float): The learning rate of the model.
            DROPOUT_RATE (float): The dropout rate of the model.
            REGULARIZATION (float): The regularization factor of the model.
        """

        super(maxpool_lstm, self).__init__(
            num_classes, LEARNING_RATE, DROPOUT_RATE, REGULARIZATION
        )

        # model paramters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        pool_size = 2
        mid_mlp = 64

        # model
        self.lstm_layer = nn.LSTM(
            input_size, self.hidden_size, self.num_layers, batch_first=True
        )
        self.max_pool = nn.Sequential(
            nn.Dropout(self.dropout_rate), nn.MaxPool1d(pool_size)
        )
        self.mlp = nn.Sequential(
            nn.Linear(int(self.hidden_size / 2) * win_size, mid_mlp),
            nn.ReLU(),
            nn.Linear(mid_mlp, self.num_classes),
        )

    def forward(self, x):
        """
        Computes the output of the neural network for a given input.

        Returns:
            A torch.Tensor containing the output of the neural network.
        """

        out, _ = self.lstm_layer(x)
        out = out.transpose(2, 1)
        out = self.max_pool(out)
        out = out.reshape(x.shape[0], -1)
        last_layer = self.mlp(out)
        return F.log_softmax(last_layer, dim=-1)


class simple_lstm(Classifier):
    """
    This is a child class of the Classifier class.
    It has lstm layers coupled with a fully connected layer.

    Attributes:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden units of the lstm.
        num_layers (int): The number of recurrent layers.
        num_classes (int): The number of classes.
        LEARNING_RATE (float): The learning rate of the model.
        DROPOUT_RATE (float): The dropout rate of the model.
        REGULARIZATION (float): The regularization factor of the model.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        LEARNING_RATE,
        DROPOUT_RATE,
        REGULARIZATION,
        **kwargs
    ):
        """
        The constructor for the simple_lstm class.

        Parameters:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden units of the lstm.
            num_layers (int): The number of recurrent layers.
            num_classes (int): The number of classes.
            LEARNING_RATE (float): The learning rate of the model.
            DROPOUT_RATE (float): The dropout rate of the model.
            REGULARIZATION (float): The regularization factor of the model.
        """

        super(simple_lstm, self).__init__(
            num_classes, LEARNING_RATE, DROPOUT_RATE, REGULARIZATION
        )

        # model paramters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # model
        self.lstm_layer = nn.LSTM(
            input_size,
            self.hidden_size,
            self.num_layers,
            dropout=self.dropout_rate,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        """
        Computes the output of the neural network for a given input.

        Returns:
            A torch.Tensor containing the output of the neural network.
        """

        out, _ = self.lstm_layer(x)
        out = out[:, -1, :]
        last_layer = self.fc(out)
        return F.log_softmax(last_layer, dim=-1)


class bidirectional_lstm(Classifier):
    """
    This is a child class of the Classifier class.
    It has bidirectional lstm layers coupled with a fully connected layer.

    Attributes:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden units of the lstm.
        num_layers (int): The number of recurrent layers.
        num_classes (int): The number of classes.
        LEARNING_RATE (float): The learning rate of the model.
        DROPOUT_RATE (float): The dropout rate of the model.
        REGULARIZATION (float): The regularization factor of the model.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        LEARNING_RATE,
        DROPOUT_RATE,
        REGULARIZATION,
        **kwargs
    ):
        """
        The constructor for the bidirectional_lstm class.

        Parameters:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden units of the lstm.
            num_layers (int): The number of recurrent layers.
            num_classes (int): The number of classes.
            LEARNING_RATE (float): The learning rate of the model.
            DROPOUT_RATE (float): The dropout rate of the model.
            REGULARIZATION (float): The regularization factor of the model.
        """

        super(bidirectional_lstm, self).__init__(
            num_classes, LEARNING_RATE, DROPOUT_RATE, REGULARIZATION
        )

        # model paramters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # model
        self.bi_lstm_layer = nn.LSTM(
            input_size,
            self.hidden_size,
            self.num_layers,
            dropout=self.dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, x):
        """
        Computes the output of the neural network for a given input.

        Returns:
            A torch.Tensor containing the output of the neural network.
        """

        out, _ = self.bi_lstm_layer(x)
        out = out[:, -1, :]
        last_layer = self.fc(out)
        return F.log_softmax(last_layer, dim=-1)


class maxpool_bidirectional_lstm(Classifier):
    """
    This is a child class of the Classifier class.
    It has bidirectional lstm layers coupled with a max pooling layer and fully connected layers.

    Attributes:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden units of the lstm.
        num_layers (int): The number of recurrent layers.
        num_classes (int): The number of classes.
        win_size (int): The size of the sliding window.
        LEARNING_RATE (float): The learning rate of the model.
        DROPOUT_RATE (float): The dropout rate of the model.
        REGULARIZATION (float): The regularization factor of the model.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        win_size,
        LEARNING_RATE,
        DROPOUT_RATE,
        REGULARIZATION,
        **kwargs
    ):
        """
        The constructor for the maxpool_bidirectional_lstm class.

        Parameters:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden units of the lstm.
            num_layers (int): The number of recurrent layers.
            num_classes (int): The number of classes.
            win_size (int): The size of the sliding window.
            LEARNING_RATE (float): The learning rate of the model.
            DROPOUT_RATE (float): The dropout rate of the model.
            REGULARIZATION (float): The regularization factor of the model.
        """

        super(maxpool_bidirectional_lstm, self).__init__(
            num_classes, LEARNING_RATE, DROPOUT_RATE, REGULARIZATION
        )

        # model paramters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        pool_size = 2
        mid_mlp = 128

        # model
        self.bi_lstm_layer = nn.LSTM(
            input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=True,
        )
        self.max_pool = nn.Sequential(
            nn.Dropout(self.dropout_rate), nn.MaxPool1d(pool_size)
        )
        self.mlp = nn.Sequential(
            nn.Linear(int(self.hidden_size / 2) * win_size * 2, mid_mlp),
            nn.ReLU(),
            nn.Linear(mid_mlp, self.num_classes),
        )

    def forward(self, x):
        """
        Computes the output of the neural network for a given input.

        Returns:
            A torch.Tensor containing the output of the neural network.
        """

        out, _ = self.bi_lstm_layer(x)
        out = out.transpose(2, 1)
        out = self.max_pool(out)
        out = out.reshape(x.shape[0], -1)
        last_layer = self.mlp(out)
        return F.log_softmax(last_layer, dim=-1)
