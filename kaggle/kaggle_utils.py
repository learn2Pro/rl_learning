import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Dataset():
    def get_dataloader(self, train=True):
        pass


class Model(nn.Module):

    def loss_fn(self, y_hat, y):
        pass

    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_step(self, batch):
        """train the batch data
        Args:
            batch: tuple(train_features, train_labels)

        Returns:
            y_hat: the predict lables
            loss: the loss tensor from network
        """
        return self.step(batch)[1]

    def validate_step(self, batch):
        """validate the batch data
        Args:
            batch: tuple(train_features, train_labels)

        Returns:
            loss: the loss tensor from network
        """
        return self.step(batch)[1]

    def step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        return y_hat, loss


class Classifier(Model):
    def __init__(self, input_dims, output_dims, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.net = nn.Sequential(
            nn.LayerNorm(input_dims),
            nn.ReLU(),
            nn.LazyLinear(output_dims),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

    def loss_fn(self, y_hat, y):
        """cross entropy"""
        return F.cross_entropy(y_hat, y)


class Regression(Model):
    """regression base class"""

    def __init__(self, input_dims, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.net = nn.Sequential(
            nn.LayerNorm(input_dims),
            nn.ReLU(),
            nn.LazyLinear(1)
        )

    def forward(self, x):
        return self.net(x)

    def loss_fn(self, y_hat, y):
        """logloss"""
        return F.mse_loss(y_hat, y)


class Trainer:
    """convenient object for trainning"""

    def __init__(self, max_epochs):
        self.max_epochs = max_epochs
        self.train_loss = []
        self.validate_loss = []

    def fit(self, model, data):
        self.optim = model.optimizer()
        for epoch in range(self.max_epochs):
            model.train()
            t_loss = []
            for batch in data.get_dataloader(train=True):
                loss = model.train_step(batch)
                self.optim.zero_grad()
                # with torch.no_grad():
                loss.backward()
                t_loss.append(loss.item())
                self.optim.step()
            train_loss = np.array(t_loss).sum()
            model.eval()
            v_loss = []
            for batch in data.get_dataloader(train=False):
                with torch.no_grad():
                    loss = model.validate_step(batch)
                    v_loss.append(loss.item())
            val_loss = np.array(v_loss).sum()

            print(
                f'complete {epoch} epoch train_loss={train_loss} validate_loss={val_loss}')
            self.train_loss.append(train_loss)
            self.validate_loss.append(val_loss)

    def plot(self, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(self.train_loss, linestyle='-', label='train')
        plt.plot(self.validate_loss, linestyle='-.', label='validate')
        plt.legend()
        plt.show()
