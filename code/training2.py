import lightning as l
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchvision import transforms
from lightning.pytorch.loggers import CSVLogger
from model2 import SiameseNetwork2,LightningModel2,CustomDataModule2

torch.manual_seed(123)

if __name__=="__main__":
    transformation = transforms.Compose([transforms.Resize((100,100)),
                                     transforms.ToTensor()
                                    ])

    dm = CustomDataModule2(batch_size=64,transform=transformation)

    pytorch_model = SiameseNetwork2()

    lightning_model = LightningModel2(model=pytorch_model, learning_rate=0.00005)


    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",filename="best_model2",
        save_top_k=1,monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_acc", patience=3, verbose=True, mode="max"
    )

    trainer = l.Trainer(callbacks=[checkpoint_callback,early_stopping_callback],
        max_epochs=6,
        accelerator="cpu",
        logger=CSVLogger(save_dir="logs/", name="model2_logs"),
        devices="auto",
    )

    trainer.fit(model=lightning_model, datamodule=dm)

    train_acc = trainer.validate(dataloaders=dm.train_dataloader())[0]["val_acc"]
    val_acc = trainer.validate(datamodule=dm)[0]["val_acc"]
    test_acc = trainer.test(datamodule=dm)[0]["test_acc"]
    print(
        f"Train Acc {train_acc*100:.2f}%"
        f" | Val Acc {val_acc*100:.2f}%"
        f" | Test Acc {test_acc*100:.2f}%"
    )