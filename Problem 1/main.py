# main.py
import torch
import platform
from model import MLP
from train import train
from evaluate import evaluate
from utils import get_dataloaders
from torch.optim import SGD, Adam, RMSprop, Adagrad
from torch.optim.lr_scheduler import StepLR

# Device Selection
if torch.cuda.is_available():
    device = torch.device('cuda')
elif platform.system() == 'Darwin' and platform.machine() == 'arm64' and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

# Data Loaders
train_loader, test_loader = get_dataloaders()

# Optimizers Configuration
optimizers_config = {
    "SGD": lambda params: SGD(params, lr=0.01),
    "SGD+Momentum": lambda params: SGD(params, lr=0.01, momentum=0.9),
    "AdaGrad": lambda params: Adagrad(params, lr=0.01),
    "RMSprop": lambda params: RMSprop(params, lr=0.001),
    "Adam": lambda params: Adam(params, lr=0.001),
}

# Training and Evaluation
results = {}

for name, opt_func in optimizers_config.items():
    print(f"Training with {name}...")
    model = MLP().to(device)
    optimizer = opt_func(model.parameters())
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    train(model, optimizer, scheduler, train_loader, device, epochs=10)
    acc = evaluate(model, test_loader, device)
    results[name] = acc
    print(f"{name} Test Accuracy: {acc:.2f}%\n")

# Final Results
print("=== Final Results ===")
for name, acc in results.items():
    print(f"{name}: {acc:.2f}%")
