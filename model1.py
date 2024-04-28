# File for model 1
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.autograd import Variable
from torch.utils.data import Dataset

args = {}
kwargs = {}
args['batch_size'] = 64
args['test_batch_size'] = 64
args['epochs'] = 3
args['lr'] = 0.01
args['momentum'] = 0.5
args['seed'] = 1
args['log_interval'] = 10
args['cuda'] = False


class PhishingDataset(Dataset):
    def __init__(self, file):
        self.data = pandas.read_csv(file, usecols=lambda x: x != 'FILENAME')

        label_encoders = {}
        for column in self.data.select_dtypes(include=['object']).columns:
            label_encoders[column] = LabelEncoder()
            self.data[column] = label_encoders[column].fit_transform(self.data[column])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data.iloc[idx, :-1], dtype=torch.float32)
        target = torch.tensor(self.data.iloc[idx, -1], dtype=torch.float32)

        return sample, target


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(54, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)

        return x


csv_file = 'PhiUSIIL_Phishing_URL_Dataset.csv'
dataset = PhishingDataset(csv_file)

training_X, test_X, training_Y, test_Y = train_test_split(dataset.data.iloc[:, :-1], dataset.data.iloc[:, -1],
                                                          test_size=0.2, random_state=args['seed'])

scaler = MinMaxScaler()
training_X = scaler.fit_transform(training_X.values)
test_X = scaler.fit_transform(test_X.values)

training_tensor_X = torch.from_numpy(training_X).float()
training_tensor_Y = torch.from_numpy(training_Y.values.ravel()).float()
test_tensor_X = torch.from_numpy(test_X).float()
test_tensor_Y = torch.from_numpy(test_Y.values.ravel()).float()

training_tensor_Y = training_tensor_Y.unsqueeze(1)
test_tensor_Y = test_tensor_Y.unsqueeze(1)

training_set = torch.utils.data.TensorDataset(training_tensor_X, training_tensor_Y)
test_set = torch.utils.data.TensorDataset(test_tensor_X, test_tensor_Y)

train_loader = torch.utils.data.DataLoader(training_set, batch_size=args['batch_size'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['test_batch_size'], shuffle=True, **kwargs)
train_losses = []
test_losses = []


def train(epoch):
    model.train()
    final_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args['cuda']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = func.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
        final_loss = loss.data.item()
    train_losses.append(final_loss)


def test():
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            if args['cuda']:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += func.binary_cross_entropy_with_logits(output, target).data.item()
            target = target.squeeze()
            output = output.squeeze()
            output = output > .5
            correct += (target.data == output.data).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    test_losses.append(test_loss)


model = Net()
if args['cuda']:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

for _epoch in range(1, args['epochs'] + 1):
    train(_epoch)
    test()
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Testing loss')
plt.xticks([0, 1, 2], ["1", "2", "3"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Plot for Simple DNN")
plt.legend()
plt.show()
