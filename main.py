from torch_geometric.datasets import MD17
import gnnff
import torch
import torch.nn.functional as F
from torch.nn import Flatten
from torcheval.metrics import MeanSquaredError as MSE

dataset_train = MD17(root='aspirin_dataset', name='aspirin CCSD', train=True)
dataset_val = MD17(root='aspirin_dataset', name='aspirin CCSD', train=False)

data_train = dataset_train[0]
print(data_train)

z_train = data_train.z
pos_train = data_train.pos
force_train = data_train.force

data_val = dataset_val[0]
z_val = data_val.z
pos_val = data_val.pos
force_val = data_val.force

print(z_train.size())
print(pos_train.size())
print(force_train.size())
print(force_train)
print(z_train)

model = gnnff.GNNFF(10,10,3)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(z_train,pos_train)
    loss = F.mse_loss(out, force_train)
    loss.backward()
    optimizer.step()


model.eval()
pred = model(z_val, pos_val)
mse = MSE()
metric = mse.update(pred,force_val)
metric = mse.compute()
print(f'Accuracy: {metric:.4f}')
