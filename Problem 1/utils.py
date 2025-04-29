# utils.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    transform = transforms.ToTensor()
    train = datasets.MNIST(root='.', train=True, transform=transform, download=True)
    test = datasets.MNIST(root='.', train=False, transform=transform, download=True)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size)
    )
