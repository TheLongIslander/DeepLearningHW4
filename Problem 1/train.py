# train.py
import torch
from torch.nn import CrossEntropyLoss

def train(model, optimizer, scheduler, train_loader, device, epochs=10):
    model.train()
    criterion = CrossEntropyLoss()
    for epoch in range(epochs):
        for batch in train_loader:
            x, y = batch
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()
