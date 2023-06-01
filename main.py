from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader
import gnnff
import torch
from matplotlib import pyplot as plt
import time
import train_utils

batch_size = 1  
node_channels = 4
edge_channels = 50
layers = 2
epochs = 3
lr = 1e-2


dataset_train = MD17(root='aspirin_dataset', name='aspirin CCSD', train=True)
dataset_val = MD17(root='aspirin_dataset', name='aspirin CCSD', train=False)

train_loader = DataLoader(dataset_train, batch_size=batch_size)
val_loader = DataLoader(dataset_val, batch_size=batch_size)

model = gnnff.GNNFF(hidden_node_channels=node_channels,
                    hidden_edge_channels=edge_channels,
                    num_layers=layers)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

losses = []
val_losses = []

for epoch in range(epochs):
    start = time.time()
    model.train()
    mean_train_loss = train_utils.train_loop(model, train_loader, criterion, optimizer, losses)
    model.eval()
    mean_val_loss  = train_utils.val_loop(model, val_loader, criterion, val_losses)
    end = time.time()
    elapsed = end-start
    print("\nEpoch %d done, time: %.1f s" % (epoch+1, elapsed))
    print("mean_train_loss: %.2f, mean_val_loss: %.2f" % (mean_train_loss, mean_val_loss))


plt.plot(losses, label='Train MSE')
plt.plot(val_losses, label='Validation MSE')
plt.xlabel('steps')
plt.ylabel('MSE')
plt.legend()
plt.show()

