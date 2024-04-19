# File for model 1
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from torch.utils.data import Dataset

args = {}
kwargs = {}
args['batch_size'] = 64
args['test_batch_size'] = 64
args['epochs'] = 15
args['lr'] = 0.1
args['momentum'] = 0.6
args['seed'] = 1
args['log_interval'] = 10
args['cuda'] = True
device = torch.device('cuda')


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
        target = torch.tensor(self.data.iloc[idx, -1], dtype=torch.long)

        return sample, target


csv_file = 'PhiUSIIL_Phishing_URL_Dataset.csv'
dataset = PhishingDataset(csv_file)

training_X, test_X = train_test_split(dataset.data.iloc[:, :-1], test_size=0.2, random_state=args['seed'])
training_Y, test_Y = train_test_split(dataset.data.iloc[:, -1], test_size=0.2, random_state=args['seed'])

training_set = torch.utils.data.TensorDataset(torch.tensor(training_X.values, dtype=torch.float32),
                                              torch.tensor(training_Y.values, dtype=torch.float32))
test_set = torch.utils.data.TensorDataset(torch.tensor(test_X.values, dtype=torch.long),
                                          torch.tensor(test_Y.values, dtype=torch.long))

train_loader = torch.utils.data.DataLoader(training_set, batch_size=args['batch_size'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['test_batch_size'], shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(54, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)
        if args['cuda']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


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
            test_loss += F.nll_loss(output, target, size_average=False).data.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


model = Net()
if args['cuda']:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

for _epoch in range(1, args['epochs'] + 1):
    train(_epoch)
    test()
