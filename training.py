import lightning as l
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchvision import transforms
from model import SiameseNetwork,LightningModel,CustomDataModule

torch.manual_seed(123)

if __name__=="__main__":
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img / 255.0)])

    dm = CustomDataModule(batch_size=64,transform=transform)

    pytorch_model = SiameseNetwork()

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.00005)


    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",filename="best_model",
        save_top_k=1,monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_acc", patience=3, verbose=True, mode="max"
    )

    trainer = l.Trainer(callbacks=[checkpoint_callback,early_stopping_callback],
        max_epochs=6,
        accelerator="gpu",
        devices="auto",
    )

    trainer.fit(model=lightning_model, datamodule=dm,ckpt_path="models/best_model.ckpt")
