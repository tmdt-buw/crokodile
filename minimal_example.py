import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset


class Submodule(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.layer.parameters())

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.layer(x), y)
        self.log("train_loss", loss)


dataloader = DataLoader(
    TensorDataset(torch.rand(10, 10), torch.rand(10, 1)),
)

Trainer(max_epochs=1).fit(Submodule(), dataloader, dataloader)


class Architecture(LightningModule):
    def __init__(self):
        super().__init__()
        self.submodule = Submodule()

    def configure_optimizers(self):
        return self.submodule.configure_optimizers()

    def training_step(self, batch, batch_idx):
        self.submodule.training_step(batch, batch_idx)

        self.log("bar", 1)


Trainer(max_epochs=1).fit(Architecture(), dataloader, dataloader)
