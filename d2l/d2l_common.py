import collections
import inspect
from IPython import display
from torch import nn
from d2l import torch as d2l
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import os
import requests
import zipfile
import hashlib
import tarfile
import re


def add_to_class(Class):  # @save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper


class HyperParameters:  # @save
    """The base class of hyperparameters."""

    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented


@add_to_class(HyperParameters)  # @save
def save_hyperparameters(self, ignore=[]):
    """Save function arguments into class attributes."""
    frame = inspect.currentframe().f_back
    _, _, _, local_vars = inspect.getargvalues(frame)
    self.hparams = {k: v for k, v in local_vars.items()
                    if k not in set(ignore+['self']) and not k.startswith('_')}
    for k, v in self.hparams.items():
        setattr(self, k, v)


class ProgressBoard(HyperParameters):  # @save
    """The board that plots data points in animation."""

    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented


@add_to_class(ProgressBoard)  # @save
def draw(self, x, y, label, every_n=1):
    Point = collections.namedtuple('Point', ['x', 'y'])
    if not hasattr(self, 'raw_points'):
        self.raw_points = collections.OrderedDict()
        self.data = collections.OrderedDict()
    if label not in self.raw_points:
        self.raw_points[label] = []
        self.data[label] = []
    points = self.raw_points[label]
    line = self.data[label]
    points.append(Point(x, y))
    if len(points) != every_n:
        return

    def mean(x): return sum(x) / len(x)
    line.append(Point(mean([p.x for p in points]),
                      mean([p.y for p in points])))
    points.clear()
    if not self.display:
        return
    d2l.use_svg_display()
    if self.fig is None:
        self.fig = d2l.plt.figure(figsize=self.figsize)
    plt_lines, labels = [], []
    for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
        plt_lines.append(d2l.plt.plot([p.x for p in v], [p.y for p in v],
                                      linestyle=ls, color=color)[0])
        labels.append(k)
    axes = self.axes if self.axes else d2l.plt.gca()
    if self.xlim:
        axes.set_xlim(self.xlim)
    if self.ylim:
        axes.set_ylim(self.ylim)
    if not self.xlabel:
        self.xlabel = self.x
    axes.set_xlabel(self.xlabel)
    axes.set_ylabel(self.ylabel)
    axes.set_xscale(self.xscale)
    axes.set_yscale(self.yscale)
    axes.legend(plt_lines, labels)
    display.display(self.fig)
    display.clear_output(wait=True)


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.plot_train_per_epoch = 2
        self.plot_valid_per_epoch = 1
        self.board = ProgressBoard()
        self.lr = 0.01

    def loss(self):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'neural network is not defined'
        return self.net(X)

    def plot(self):
        """plot a point in animation"""
        # assert hasattr(self, 'trainer'), 'Trainer is not inited'
        # self.board.xlabel = 'epoch'
        # if train:
        #     x = self.trainer.train_batch_ix / self.trainer.num_train_batches
        #     n = self.trainer.num_train_batches / self.plot_train_per_epoch
        # else:
        #     x = self.trainer.epoch+1
        #     n = self.trainer.num_val_battches / self.plot_valid_per_epoch
        # self.board.draw(x, value.item(),
        #                 ('train_' if train else 'val_')+key, every_n=int(n))
        plt.plot(self.train_loss, label='train')
        plt.plot(self.validate_loss, label='validate')
        plt.xlabel = 'epoch'
        plt.ylabel = 'loss'
        plt.legend()

    def train_step(self, batch):
        y_hat = self.forward(batch[0])
        y = batch[1]
        l = self.loss(y_hat, y)
        return l

    def validate_step(self, batch):
        l = self.loss(self.forward(batch[0]), batch[-1])
        return l

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


class DataModule(HyperParameters):  # @save
    """The base class of data."""

    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def get_tensorloader(self, tensors, train, indices=(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)


class Trainer(HyperParameters):
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'no gpu supported yet'
        self.train_batch_ix = 0
        self.val_batch_idx = 0
        self.train_loss = []
        self.validate_loss = []
        self.acc_array = []

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.t_loss = float('inf')
        self.v_loss = float('inf')
        self.acc = float('inf')
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
            print(
                f'complete {self.epoch} epoch train_loss={self.t_loss} validate_loss={self.v_loss}')

    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.train_step(batch)
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
                self.train_batch_idx += 1
        self.t_loss = loss
        self.train_loss.append(self.t_loss.item())
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                y_hat, self.v_loss = self.model.validate_step(batch)
                if hasattr(self.model, 'accuracy'):
                    self.acc = self.model.accuracy(y_hat, batch[-1])
            self.val_batch_idx += 1
        self.validate_loss.append(self.v_loss.item())
        if isinstance(self.acc, torch.TensorType):
            self.acc_array.append(self.acc.item())

    def plot(self, figsize=(10, 5)):
        plt.figure(1, figsize=figsize)
        plt.plot(self.train_loss, label='train')
        plt.plot(self.validate_loss, linestyle='dashed', label='validate')
        plt.plot(self.acc_array, linestyle='dashdot', label='accuracy')
        plt.xlabel = 'epoch'
        plt.ylabel = 'loss'
        plt.legend()
        plt.show()

    def clip_gradients(self, grad_clip_val, model):
        params = [(name, p)
                  for name, p in model.named_parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2))
                          for name, p in params if p.grad is not None))
        # norm = torch.sqrt(sum(torch.sum((p.grad**2))
        #                   for name, p in params if p.grad is not None))
        if norm > grad_clip_val:
            for name, param in params:
                if param.grad is None:
                    continue
                param.grad[:] *= grad_clip_val / norm


class SyntheticRegressionData(DataModule):  # @save
    """Synthetic data for linear regression."""

    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)


class Classifier(Module):
    def __init__(self):
        super().__init__()

    def validate_step(self, batch):
        y_hat = self.forward(batch[0])
        return y_hat, self.loss(y_hat, batch[1])

    def accuracy(self, y_hat, y, averaged=True):
        cmp = (y_hat.argmax(axis=1) == y).type(torch.float32)
        return cmp.mean() if averaged else cmp

    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def layer_summary(self, x_shape):
        x = torch.randn(*x_shape)
        for layer in self.net:
            x = layer(x)
            print(layer.__class__.__name__, 'output shape:\t', x.shape)


class LinearRegression(Module):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1)
        )

    def forward(self, x):
        return self.net(x)

    def loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def validate_step(self, batch):
        y_hat = self.forward(batch[0])
        return y_hat, self.loss(y_hat, batch[1])


class FasionMNIST(DataModule):
    """The Fashion-MNIST dataset."""

    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose(
            [transforms.Resize(resize), transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root='../data', train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root='../data', train=False, transform=trans, download=True)
        self.num_workers = 1
        self.batch_size = batch_size

    def text_labels(self, indices):
        """Return text labels."""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train, num_workers=self.num_workers)

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)


def pplot(x, y, xlabel, ylabel, figsize):
    plt.figure(figsize=figsize)
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.plot(x, y)
    plt.grid()
    plt.show()


def download(url, folder='../data', sha1_hash=None):  # @save
    """Download a file to folder and return the local filepath."""
    # if not url.startswith('http'):
    #     # For back compatability
    #     url, sha1_hash = DATA_HUB[url]
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # Check if hit cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    # Download
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def extract(filename, folder=None):  # @save
    """Extract a zip/tar file into folder."""
    base_dir = os.path.dirname(filename)
    _, ext = os.path.splitext(filename)
    assert ext in ('.zip', '.tar', '.gz'), 'Only support zip/tar files.'
    if ext == '.zip':
        fp = zipfile.ZipFile(filename, 'r')
    else:
        fp = tarfile.open(filename, 'r')
    if folder is None:
        folder = base_dir
    fp.extractall(folder)


def download_extract(name, folder=None):  # @save
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def corr2d(x, k):
    h, w = k.shape
    y = torch.zeros((x.shape[0]-h+1, x.shape[1]-w+1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i:i+h, j:j+w]*k).sum()
    return y


class Vocab:
    """
    Vocabulary for text.
    """

    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]) -> None:
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(
            counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = list(sorted(set(
            ['<unk>']+reserved_tokens + [token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx for idx,
                             token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[idx] for idx in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):
        return self.token_to_idx['<unk>']


class TimeMachine(DataModule):
    def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000, fname='timemachine.txt', root='../data'):
        self.fname = fname
        self.root = root
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val
        corups, self.vocab = self.build(self._download())
        array = torch.tensor([corups[i:i+num_steps+1]
                             for i in range(len(corups)-num_steps)])
        self.array = array
        self.X, self.Y = array[:, :-1], array[:, 1:]

    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(
            self.num_train, self.num_train+self.num_val)
        return self.get_tensorloader([self.X, self.Y], idx)

    def _download(self):
        with open(self.root+'/'+self.fname) as f:
            return f.read()

    def _preprocess(self, text):
        return re.sub('[^A-Za-z]+', ' ', text).lower()

    def _tokenize(self, text):
        return list(text)

    def build(self, raw_txt, vocab=None):
        tokens = self._tokenize(self._preprocess(raw_txt))
        if vocab is None:
            vocab = Vocab(tokens)
        corups = [vocab[token] for token in tokens]
        return corups, vocab


if __name__ == "__main__":
    print(1)
