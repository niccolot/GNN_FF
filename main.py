from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader
import gnnff
import torch
from torcheval.metrics import MeanSquaredError as MSE
from matplotlib import pyplot as plt
import time


batch_size = 1  
node_channels = 4
edge_channels = 50
layers = 2
epochs = 3
lr = 1e-2


dataset_train = MD17(root='aspirin_dataset', name='aspirin CCSD', train=True)
dataset_val = MD17(root='aspirin_dataset', name='aspirin CCSD', train=False)

train_loader = DataLoader(dataset_train, batch_size=batch_size)


model = gnnff.GNNFF(hidden_node_channels=node_channels,
                    hidden_edge_channels=edge_channels,
                    num_layers=layers)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

losses = []

model.train()
for epoch in range(epochs):
    start = time.time()
    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(data.z, data.pos)
        loss = criterion(out, data.force)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    end = time.time()
    elapsed = end-start
    print("Epoch: %d done, time: %.1f" %(epoch, elapsed))

plt.plot(losses, label='MSE')
plt.xlabel('steps')
plt.show()