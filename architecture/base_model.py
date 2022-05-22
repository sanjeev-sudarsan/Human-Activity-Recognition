import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import ConfusionMatrix
from torch.nn import functional as F
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier(pl.LightningModule):
    """
    This is a class for training and testing the models.

    Attributes:
        num_classes (int): The number of classes.
        LEARNING_RATE (float): The learning rate of the model.
        DROPOUT_RATE (float): The dropout rate of the model.
        REGULARIZATION (float): The regularization factor of the model.
    """

    def __init__(
        self, num_classes, LEARNING_RATE, DROPOUT_RATE, REGULARIZATION, **kwargs
    ):
        """
        The constructor for the Classifier class.

        Parameters:
            num_classes (int): The number of classes.
            LEARNING_RATE (float): The learning rate of the model.
            DROPOUT_RATE (float): The dropout rate of the model.
            REGULARIZATION (float): The regularization factor of the model.
        """

        super().__init__()
        assert num_classes > 0
        # paramters
        self.dropout_rate = DROPOUT_RATE
        self.num_classes = num_classes
        self.learning_rate = LEARNING_RATE
        self.regularization = REGULARIZATION

        # metrics
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.test_confusion = ConfusionMatrix(num_classes=self.num_classes)

        self.criterion = torch.nn.NLLLoss()

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            A torch.optim.Optimizer object.
        """

        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.regularization
        )

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step and updates the accuracy.

        Returns:
            A torch.Tensor containing the loss.
        """

        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        pred = y_hat.max(-1)[1]
        self.train_acc.update(pred, y)
        return loss

    def training_epoch_end(self, outputs):
        """
        Computes the training accuracy at the end of each epoch.
        Also resets the metrics.
        """

        train_accuracy = self.train_acc.compute()
        self.log("step", float(self.trainer.current_epoch))
        self.log("train_accuracy", train_accuracy, prog_bar=True, logger=True)
        wandb.log(
            {"train_accuracy": train_accuracy, "epoch": self.trainer.current_epoch}
        )
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step and updates the accuracy.

        Returns:
            A torch.Tensor containing the loss.
        """

        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        pred = y_hat.max(-1)[1]
        self.val_acc.update(pred, y)
        return loss

    def validation_epoch_end(self, outputs):
        """
        Computes the validation accuracy at the end of each epoch.
        Also resets the metrics.
        """

        val_accuracy = self.val_acc.compute()
        self.log("step", float(self.trainer.current_epoch))
        self.log("validation_accuracy", val_accuracy, prog_bar=True, logger=True)
        wandb.log(
            {"validation_accuracy": val_accuracy, "epoch": self.trainer.current_epoch}
        )
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step and updates the accuracy and confusion matrix.

        Returns:
            A torch.Tensor containing the loss.
        """

        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        pred = y_hat.max(-1)[1]
        self.test_acc.update(pred, y)
        self.test_confusion.update(y_hat, y)
        return loss

    def test_epoch_end(self, outputs):
        """
        Computes the test accuracy and confusion matrix at the end of each epoch.
        Also resets the metrics.
        """

        test_accuracy = self.test_acc.compute()
        conf_mat = self.test_confusion.compute().detach().cpu().numpy().astype(np.int)
        fig = cm_figure(conf_mat, self.num_classes)
        self.logger.experiment.add_figure("Confusion matrix", fig, self.current_epoch)
        self.log("step", float(self.trainer.current_epoch))
        self.log("test_accuracy", test_accuracy, prog_bar=True, logger=True)
        wandb.log({"test_accuracy": test_accuracy, "epoch": self.trainer.current_epoch})
        self.test_confusion.reset()
        self.test_acc.reset()


def accuracy(output, y):
    """
    Computes the accuracy for the given inputs.

    Returns:
        A torch.Tensor containing the accuracy.
    """

    pred = output.max(-1)[1]
    return torch.sum(pred == y).float() / y.shape[0]


def cm_figure(cm, num_classes):
    """
    Creates a heatmap figure for the provided confusion matrix.

    Returns:
        fig_ (matplotlib.pyplot.figure): The heatmap for the confusion matrix.
    """

    df_cm = pd.DataFrame(
        cm, index=np.arange(num_classes), columns=np.arange(num_classes)
    )
    plt.figure()
    sn_plot = sns.heatmap(
        df_cm,
        annot=True,
        fmt="d",
        cbar=False,
        cmap="Blues",
    )
    plt.xlabel("Predictions")
    plt.ylabel("Labels")
    fig_ = sn_plot.get_figure()
    plt.close(fig_)
    return fig_
