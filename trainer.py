from torch import nn, optim
from torch.utils.data import DataLoader
from torch import device
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
        patience=5,
        delta=0.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model = None

        self.metrics = []

    def train(self):
        for epoch in range(self.epochs):
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

            train_loss = running_loss / len(self.train_loader)
            train_acc = running_acc / len(self.train_loader.dataset)  # type: ignore

            self.model.eval()
            running_val_loss, running_val_acc = 0.0, 0.0

            with torch.no_grad():
                for data in self.val_loader:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)

                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)

                    running_val_loss += loss.item()
                    running_val_acc += (outputs.argmax(dim=1) == labels).sum().item()

            val_loss = running_val_loss / len(self.val_loader)
            val_acc = running_val_acc / len(self.val_loader.dataset)  # type: ignore
            
            self.metrics.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
            
            # Early Stopping
            if self.best_score is None:
                self.best_score = val_loss
                self.best_model = self.model.state_dict()
            elif val_loss > self.best_score - self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    break
            else:
                self.best_score = val_loss
                self.best_model = self.model.state_dict()
                self.counter = 0
                
                
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
            
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)