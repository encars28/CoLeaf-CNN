from torch import nn, optim
from torch.utils.data import DataLoader
from torch import device
from tqdm.notebook import tqdm
import torch

class CoLeafTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
        device: device = torch.device("cpu"),
        epochs=50,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs

        self.metrics = []

    def train(self):
        total_data = len(self.train_loader.dataset.dataset)  # type: ignore
        batches = len(self.train_loader)

        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            self.model.train()
            running_loss, running_acc = 0.0, 0.0

            for i, data in enumerate(self.train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_acc += (outputs.argmax(dim=1) == labels).sum().item()

            train_loss = running_loss / batches
            train_acc = running_acc / total_data

            self.model.eval()
            running_val_loss, running_val_acc = 0.0, 0.0

            with torch.no_grad():
                for data in self.val_loader:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)

                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)

                    running_val_loss += loss.item()
                    running_val_acc += (outputs.argmax(dim=1) == labels).sum().item()

            val_loss = running_val_loss / batches
            val_acc = running_val_acc / total_data
            
            self.metrics.append(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
            
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
