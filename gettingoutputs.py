import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('dataset/alldata_up.csv')
labels = df[['Elbow_tao','ShFE_tao','ShAA_tao']]
features = df.drop(['Elbow_tao','ShFE_tao','ShAA_tao'], axis=1)

class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()

        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, output_size)

    def forward(self, x):
        x1 = nn.functional.relu(self.layer1(x))
        x2 = nn.functional.relu(self.layer2(x1))
        x3 = self.layer3(x2)
        return x3


PATH = "trainedmodels/model2.pth"
model = Network(3, 3)
model.load_state_dict(torch.load(PATH))

# tensor([[1.3823, 1.0556, 0.0565],
#         [0.0000, 0.1634, 0.0000],
#         [1.9666, 1.5834, 1.7342]])


testdata = features
testdata = torch.tensor(labels.values, dtype=torch.float32)
out = model(testdata)

np_input = testdata.cpu().detach().numpy()
np_output = out.cpu().detach().numpy()

plt.plot(np_input[:785,0],np_output[:785,0])
plt.xlabel('tao')
plt.ylabel('theta')
plt.show()



