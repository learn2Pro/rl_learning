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
import math


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
    def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000, fname='timemachine.txt', root='../data', device='cpu'):
        self.fname = fname
        self.root = root
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val
        corups, self.vocab = self.build(self._download())
        array = torch.tensor([corups[i:i+num_steps+1]
                             for i in range(len(corups)-num_steps)])
        self.array = array.to(device)
        self.X, self.Y = array[:, :-1].to(device), array[:, 1:].to(device)

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


class RNNScratch(Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma
        self.W_xh = nn.Parameter(torch.randn(num_inputs, num_hiddens)*sigma)
        self.W_hh = nn.Parameter(torch.randn(num_hiddens, num_hiddens)*sigma)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))

    def forward(self, x, state=None):
        if state is None:
            state = torch.zeros(
                (x.shape[1], self.num_hiddens), device=x.device)
        else:
            state, = state
        outputs = []
        for X in x:
            # [batch_size, num_inputs]
            state = torch.tanh(X@self.W_xh+state@self.W_hh+self.b_h)
            outputs.append(state)
        return outputs, state


class RNNLMScratch(Classifier):
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.rnn = rnn
        self.lr = lr
        self.vocab_size = vocab_size
        self.init_params()

    def init_params(self):
        self.W_hq = nn.Parameter(torch.randn(
            self.rnn.num_hiddens, self.vocab_size)*self.rnn.sigma)
        self.b_q = nn.Parameter(torch.zeros(self.vocab_size))

    def train_step(self, batch):
        y_hat, y = self(*batch[:-1]), F.one_hot(batch[-1],
                                                self.vocab_size).type(torch.float32)
        l = self.loss(y_hat, y)
        return l

    def validate_step(self, batch):
        y_hat, y = self(*batch[:-1]), F.one_hot(batch[-1],
                                                self.vocab_size).type(torch.float32)
        l = self.loss(y_hat, y)
        return y_hat, l

    def accuracy(self, y_hat, y, averaged=True):
        cmp = (y_hat == F.one_hot(y, self.vocab_size).type(
            torch.float32)).type(torch.float32)
        return cmp.mean() if averaged else cmp

    def one_hot(self, x):
        return F.one_hot(x.T, self.vocab_size).type(torch.float32)

    def output_layer(self, rnn_outputs):
        outputs = [H@self.W_hq for H in rnn_outputs]
        return torch.stack(outputs, dim=1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)  # [num_steps, batch_size, vocab_size]
        rnn_outputs, _ = self.rnn(embs, state)
        return self.output_layer(rnn_outputs)

    def predict(self, prefix, num_preds, vocab, device=None):
        state, outputs = None, [vocab[prefix[0]]]
        for i in range(len(prefix)+num_preds-1):
            X = torch.tensor([[outputs[-1]]], device=device)
            embs = self.one_hot(X)
            rnn_outputs, state = self.rnn(embs, state)
            if i < len(prefix)-1:
                outputs.append(vocab[prefix[i+1]])
            else:
                Y = self.output_layer(rnn_outputs)
                outputs.append(int(Y.argmax(dim=2).reshape(1)))
        return ''.join([vocab.idx_to_token[i] for i in outputs])


class MTFraEng(DataModule):
    """The English-French dataset."""

    def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128, device='cpu'):
        super().__init__()
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val
        self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
            self._download())
        self.arrays[0].to(device)
        self.arrays[1].to(device)
        self.arrays[3].to(device)

    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else (self.num_train, None)
        return self.get_tensorloader(self.arrays, train, idx)

    def build(self, src_sentences, tgt_sentences):
        raw_txt = '\n'.join(
            [src+'\t'+tgt for src, tgt in zip(src_sentences, tgt_sentences)])
        arrays, _, _ = self._build_arrays(
            raw_txt, self.src_vocab, self.tgt_vocab)
        return arrays

    def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
        def _build_array(sentences, vocab, is_tgt=False):
            def pad_or_trim(seq, t): return (seq[:t] if len(
                seq) > t else seq+['<pad>']*(t-len(seq)))
            sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
            if is_tgt:
                sentences = [['<bos>']+s for s in sentences]
            if vocab is None:
                vocab = Vocab(sentences, min_freq=2)
            array = torch.tensor([vocab[s] for s in sentences])
            valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
            return array, vocab, valid_len
        src, tgt = self._tokenize(self._preprocess(
            raw_text), self.num_train+self.num_val)
        src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
        tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
        return ((src_array, tgt_array[:, :-1], src_valid_len, tgt_array[:, 1:]), src_vocab, tgt_vocab)

    def _download(self):
        with open('../data/fra.txt', encoding='utf-8') as f:
            return f.read()

    def _preprocess(self, text: str):
        # Replace non-breaking space with space
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        # Insert space between words and punctuation marks
        def no_space(
            char, prev_char): return char in ',.!?' and prev_char != ' '
        out = [' '+char if i > 0 and no_space(char, text[i-1])
               else char for i, char in enumerate(text.lower())]
        return ''.join(out)

    def _tokenize(self, text: str, max_examples=None):
        src, tgt = [], []
        for i, line in enumerate(text.split('\n')):
            if max_examples and i > max_examples:
                break
            parts = line.split('\t')
            if len(parts) == 2:
                src.append([t for t in parts[0].split(' ') if t]+['<eos>'])
                tgt.append([t for t in parts[1].split(' ') if t]+['<eos>'])
        return src, tgt


class Encoder(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(Module):
    """The base decoder interface for the encoder--decoder architecture."""

    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(Classifier):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_x, dec_X, *args):
        enc_all_outputs = self.encoder(enc_x, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        return self.decoder(dec_X, dec_state)[0]

    def predict_step(self, batch, device, num_steps, save_attention_weights=False):
        batch = [a.to(device) for a in batch]
        src, tgt, src_valid_len, _ = batch
        enc_all_outputs = self.encoder(src, src_valid_len)
        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
        outputs, attention_weights = [tgt[:, 0].unsqueeze(1),], []
        for _ in range(num_steps):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
            outputs.append(Y.argmax(2))
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        return torch.cat(outputs[1:], 1), attention_weights


def bleu(pred_seq, label_seq, k):
    """Compute the BLEU."""
    import math
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label/len_pred))
    # A,B,C,D,E,F
    # A,B,B,C,D
    # len_pred=6, len_label=5
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    """Show heatmaps of matrices."""
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles)
    fig.colorbar(pcm, ax=axes, shrink=0.6)


def masked_softmax(X, valid_lens):  # @save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):  # @save
    """Scaled dot product attention."""

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(Module):  # @save
    """Multi-head attention."""

    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads."""
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # Shape of output X: (batch_size, num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3)
        # Shape of output: (batch_size * num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


if __name__ == "__main__":
    print(1)
