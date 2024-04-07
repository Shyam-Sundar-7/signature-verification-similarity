#SiameseNetwork model in pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.pairwise import pairwise_euclidean_distance
import torchmetrics
import lightning as l
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd


class SiameseNetwork1(nn.Module):
    def __init__(self):
        super(SiameseNetwork1, self).__init__()

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
        # print(output1.shape,output2.shape)
        
        output1=output1.view(output1.shape[0],-1)
        output2=output2.view(output2.shape[0],-1)
        
        emb=pairwise_euclidean_distance(output2,output1,reduction="mean")
        # print(output1.shape,output2.shape)
        output = torch.abs(output2-output1)**2

        # Pass flattened output through the fully connected layers
        output = self.fc(output)
        # print(output.shape)
        return output.squeeze(),emb


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

class CustomDataModule1(l.LightningDataModule):
    def __init__(self,  transform=None, batch_size=64):
        super().__init__()
        self.transform = transform
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = CustomDataset("data/CEDAR_train.csv", transform=self.transform)
        self.val_dataset = CustomDataset("data/CEDAR_val.csv", transform=self.transform)
        self.test_dataset = CustomDataset("data/CEDAR_test.csv", transform=self.transform)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=7)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=7)

class LightningModel1(l.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=['model'])

        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        # self.val_auc=torchmetrics.ROC(task="binary")
        self.val_recall=torchmetrics.Recall(task="binary")
        self.val_precision=torchmetrics.Precision(task="binary")
        # self.test_auc=torchmetrics.ROC(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.test_recall=torchmetrics.Recall(task="binary")
        self.test_precision=torchmetrics.Precision(task="binary")

    def forward(self, x,y):
        return self.model(x,y)

    def _shared_step(self, batch):
        input,original, true_labels = batch
        logits,_= self(input,original)
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
        self.val_recall(predicted_labels, true_labels)
        self.log("val_recall", self.val_recall, prog_bar=True)
        self.val_precision(predicted_labels, true_labels)
        self.log("val_precision", self.val_precision, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)
        self.test_recall(predicted_labels, true_labels)
        self.log("test_recall", self.test_recall)
        self.test_precision(predicted_labels, true_labels)
        self.log("test_precision", self.test_precision)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer