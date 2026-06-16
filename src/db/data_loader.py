from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


class CIFAR_10_LOADER:


    def __init__(self):
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

        self.train_datasets = CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

        self.test_datasets = CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )


    @property
    def train_loader(self, batch_size=64):

        loader = DataLoader(
            dataset=self.train_datasets,
            shuffle=True,
            batch_size=batch_size
        )
        return loader
    

    @property
    def test_loader(self, batch_size=64):

        loader = DataLoader(
            dataset=self.test_datasets,
            batch_size=batch_size,
            shuffle=False
        )
        return loader