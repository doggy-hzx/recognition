#%%
import torch
import numpy as np
from torch.autograd import Variable
import json
#%%
f = open("facedata.json")
faceData = json.loads(f.read())
faceXx = np.array(list(faceData['x']), dtype=np.float32)
faceYy = np.array(list(faceData['y']), dtype=np.float32)
# faceXx = []
# faceXx.append(faceX[0:1])
# faceXx.append(faceX[30:31])
# faceYy = []
# faceYy.append(faceY[0:1])
# faceYy.append(faceY[1:])
# print(faceX)
# print(faceY)

#%%
lr = 0.0001
epoch = 10000
Batch_Size = 1
# %%
class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.net1 = torch.nn.LSTM(3, 3, 2)
		self.net2 = torch.nn.Sequential(
			torch.nn.Linear(3, 10),
			torch.nn.ReLU(),
			torch.nn.Linear(10, 5),
			torch.nn.ReLU(),
            torch.nn.Linear(5, 1),
			torch.nn.Sigmoid(),
		)
	def forward(self, input):
		y, w = self.net1(input)
		y = y.view(3)
		res = self.net2(y)
		return res

net = Net()
opt = torch.optim.Adam(net.parameters(), lr = lr)
bceloss = torch.nn.BCELoss()
#%%
for i in range(epoch):

	faceX = Variable(torch.from_numpy(faceXx[i % 8]))
	faceY = Variable(torch.from_numpy(faceYy[i % 8]), requires_grad = True)

	# print(faceX)

	data = net(faceX.view([1, Batch_Size, 3]))
	# print(faceX.view([1, Batch_Size, 3]))
	# loss = -torch.mean(torch.log(faceY) + torch.log(1 - data))
	loss = bceloss(data.view(1), faceY)

	opt.zero_grad()
	loss.backward(retain_graph=True)

	opt.step()

	print(data)

# print(data)
# %%
