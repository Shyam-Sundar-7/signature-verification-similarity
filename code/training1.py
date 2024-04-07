import lightning as l
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torchvision import transforms
from model1 import SiameseNetwork1,LightningModel1,CustomDataModule1

torch.manual_seed(123)

if __name__=="__main__":
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img / 255.0)])

    dm = CustomDataModule1(batch_size=64,transform=transform)

    pytorch_model = SiameseNetwork1()

    lightning_model = LightningModel1(model=pytorch_model, learning_rate=0.00005)


    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",filename="best_model",
        save_top_k=1,monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_acc", patience=3, verbose=True, mode="max"
    )

    trainer = l.Trainer(callbacks=[checkpoint_callback,early_stopping_callback],
        max_epochs=6,
        logger=CSVLogger(save_dir="logs/", name="contro_logs"),
        accelerator="gpu",
        devices="auto",
    )

    trainer.fit(model=lightning_model, datamodule=dm,ckpt_path="models/best_model.ckpt")
    train_acc = trainer.validate(dataloaders=dm.train_dataloader())[0]["val_acc"]
    val_acc = trainer.validate(datamodule=dm)[0]["val_acc"]
    test_acc = trainer.test(datamodule=dm)[0]["test_acc"]
    print(
        f"Train Acc {train_acc*100:.2f}%"
        f" | Val Acc {val_acc*100:.2f}%"
        f" | Test Acc {test_acc*100:.2f}%"
    )
