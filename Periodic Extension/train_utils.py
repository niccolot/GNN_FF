import numpy as np

def train_loop(model, train_loader, loss_fn, optimizer, losses_list):

    for iter, data in enumerate(train_loader):

        optimizer.zero_grad()
        out = model(data.z, data.x, dimension = data.bounds)
        #out = model(data.z, data.pos)
        loss = loss_fn(out, data.force)
        loss.backward()
        optimizer.step()
        losses_list.append(loss.item()/len(data))

    return np.mean(losses_list)


def val_loop(model, val_loader, loss_fn, val_losses_list):

    for iter, data in enumerate(val_loader):

        out = model(data.z, data.x, data.bounds)
        #out = model(data.z, data.pos)
        loss = loss_fn(out, data.force)
        val_losses_list.append(loss.item()/len(data))

    return np.mean(val_losses_list)

