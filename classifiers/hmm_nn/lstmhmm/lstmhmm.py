import numpy as np
import hmmlearn.hmm as hmmlearn
from sklearn.neural_network import MLPClassifier
import torch
from torch.utils.data import Dataset, DataLoader
from classifiers.hmm_nn.lstmhmm import lstm_config as conf
from torch import nn


# dataset
class SequenceDataset(Dataset):
    def __init__(self, train, target, sequence_length=conf.SEQUENCE_LENGTH):
        self.train = train
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(target).float()
        self.X = torch.tensor(train).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[i]


class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_classes):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1
        self.num_classes = num_classes
        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0])
        # out = self.linear(hn[0]).flatten() # First dim of Hn is num_layers, which is set to 1 above
        return out

def train_model(data_loader, model, optimizer, loss_function, device):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    for X, y in data_loader:
        labels = y.to(device).long()

        # Forward pass
        output = model(X)
        loss = loss_function(output, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # sum_loss += loss.item()*y.shape[0]
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")

def validate_model(data_loader, model, device):
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            labels = y.to(device).long()
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on features: {} %'.format(100 * correct / total))


def get_data_loader(features, sequences):
    dataset = SequenceDataset(
        train=features,
        target=sequences
    )
    loader = DataLoader(dataset, batch_size=conf.BATCH_SIZE, shuffle=conf.SHUFFLE_TRAINING)
    return loader


class LSTMHMM:
    def __init__(self, n_mix=2, n_components=4):
        self.n_mix = n_mix
        self.n_components = n_components
        self.hmm = hmmlearn.GaussianHMM(n_components=n_components)
        self.lstm = None
        self.features_shape = None
        self.device = torch.device('cpu') # 'cuda' if torch.cuda.is_available() else

    def fit(self, train_features, validate_features):
        self.features_shape = train_features.shape[1]
        self.train_hmm(train_features)

        train_sequences = self.viterbi_hmm(train_features)
        validate_sequences = self.viterbi_hmm(validate_features)
        train_loader = get_data_loader(train_features, train_sequences)
        validate_loader = get_data_loader(validate_features, validate_sequences)

        self.train_mlp(train_loader, validate_loader)

    def viterbi_hmm(self, features):
        sequences = []
        for feature in features:
            sequences.append(self.hmm.predict(feature))
        return sequences

    def train_hmm(self, features):
        lengths = []
        for n, i in enumerate(features):
            lengths.append(len(i))
        self.hmm.fit(np.concatenate(features), lengths)

    def train_mlp(self, train_loader, validate_loader):

        model = ShallowRegressionLSTM(num_sensors=self.features_shape, hidden_units=conf.NUM_HIDDEN_UNITS,
                                      num_classes=self.n_components)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.LEARNING_RATE)

        device = torch.device('cpu')

        for ix_epoch in range(conf.NUM_EPOCHS):
            print(f"Epoch {ix_epoch}\n---------")
            train_model(train_loader, model, optimizer=optimizer, loss_function=loss_function, device=self.device)
            validate_model(validate_loader, model, device)
            print()






