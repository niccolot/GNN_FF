from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader
import GNNFF_uff
import torch
from matplotlib import pyplot as plt
import time
import train_utils

batch_size = 1 
node_channels = 1
edge_channels = 50
layers = 4
epochs = 5
lr = 2e-3

DATA_PATH= '/home/lorenz_gauge/Documenti/Lorenzo/Esami_Magistrale/Computing/Progetto_Esame/data/processed/tensor_dataset2.pt'
dataset_train = torch.load(DATA_PATH)


data1, data2 = torch.utils.data.random_split(dataset_train, [0.5,0.5])
print(len(data1),len(data2))
train_loader = DataLoader(data1, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(data2, batch_size=batch_size, shuffle=True)


#print(dataset_train[0].bounds)
#train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)


#i = 0
#for data in train_loader:
#    print(data.size(),i,data[0].bounds)
#    i+=1
    #break

model = GNNFF_uff.GNNFF(hidden_node_channels=node_channels,
                    hidden_edge_channels=edge_channels,
                    num_layers=layers)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

losses = []
val_losses = []
#"""
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

#"""
