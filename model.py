#SiameseNetwork model in pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as l
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Define the convolutional neural network
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(32*32*32, 256),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        # Pass input through the convolutional layers
        output = self.cnn(x)
        # print(output.shape)
        # Flatten the output
        return output

    def forward(self, input1, input2):
        # Forward pass for the two inputs separately
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # Return the concatenated output
        #stack output1 and output2
        output1=output1.view(64,-1)
        output2=output2.view(64,-1)
        print(output1.shape,output2.shape)
        output = torch.abs(output2-output1)
        # Pass flattened output through the fully connected layers
        output = self.fc(output)
        print(output.shape)
        return output.squeeze()


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name1 = self.data.iloc[idx, 0]
        image1 = Image.open(img_name1).convert("RGB")

        img_name2 = self.data.iloc[idx, 1]
        image2 = Image.open(img_name2).convert("RGB")

        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        #create a image2 tensor same as the size of image2 with 1 as value
        image_like1 = torch.ones_like(image1)
        image_like2 = torch.ones_like(image2)

        label = int(self.data.iloc[idx, -1])

        return image_like1-image1,image_like2-image2, label

class CustomDataModule(l.LightningDataModule):
    def __init__(self, csv_file, transform=None, batch_size=64):
        super().__init__()
        self.csv_file = csv_file
        self.transform = transform
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = CustomDataset(self.csv_file, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,num_workers=5)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,num_workers=6)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,num_workers=6)

class LightningModel(l.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=["model"])

        self.train_acc = torchmetrics.Accuracy(task="binary")
        # self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")

    def forward(self, x,y):
        return self.model(x,y)

    def _shared_step(self, batch):
        input,original, true_labels = batch
        logits = self(input,original)
        loss = F.binary_cross_entropy(logits, true_labels.float())
        predicted_labels = (logits > 0.5).float()
        # print(true_labels.shape,predicted_labels.shape)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step= True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer