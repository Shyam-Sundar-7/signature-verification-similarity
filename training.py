import lightning as l
import torch
from torchvision import transforms
from model import SiameseNetwork,LightningModel,CustomDataModule

torch.manual_seed(123)

if __name__=="__main__":
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])

    dm = CustomDataModule('CEDAR_train.csv',batch_size=64,transform=transform)

    pytorch_model = SiameseNetwork()

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.05)

    trainer = l.Trainer(
        max_epochs=15,
        accelerator="cpu",
        devices="auto",
        deterministic=True,

    )

    trainer.fit(model=lightning_model, datamodule=dm)