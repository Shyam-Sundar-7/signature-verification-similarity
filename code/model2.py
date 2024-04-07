import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torchmetrics
import lightning as l
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
class SiameseNetwork2(nn.Module):

    def __init__(self):
        super(SiameseNetwork2, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive


class CustomDataset1(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name1 = self.data.iloc[idx, 0]
        image1 = Image.open(img_name1).convert("L")

        img_name2 = self.data.iloc[idx, 1]
        image2 = Image.open(img_name2).convert("L")

        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        

        label = int(self.data.iloc[idx, -1])

        return image1,image2, label

class CustomDataModule2(l.LightningDataModule):
    def __init__(self,  transform=None, batch_size=64):
        super().__init__()
        self.transform = transform
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = CustomDataset1("data\CEDAR_train.csv", transform=self.transform)
        self.val_dataset = CustomDataset1("data\CEDAR_val.csv", transform=self.transform)
        self.test_dataset = CustomDataset1("data\CEDAR_test.csv", transform=self.transform)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=7)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=7)

class LightningModel2(l.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=['model'])
        self.criterion = ContrastiveLoss()
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        # self.val_auc=torchmetrics.ROC(task="binary")
        # self.val_recall=torchmetrics.Recall(task="binary")
        # self.val_precision=torchmetrics.Precision(task="binary")
        # self.test_auc=torchmetrics.ROC(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        # self.test_recall=torchmetrics.Recall(task="binary")
        # self.test_precision=torchmetrics.Precision(task="binary")

    def forward(self, x,y):
        return self.model(x,y)

    def _shared_step(self, batch):
        input,original, true_labels = batch
        input_emb,out_emb= self(input,original)
        loss = self.criterion(input_emb, out_emb, true_labels.float())
        predicted_labels = F.pairwise_distance(input_emb, out_emb)
        # print(predicted_labels.shape)
        predicted_labels = torch.where(predicted_labels < 1, 0, 1)
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
        # self.val_recall(predicted_labels, true_labels)
        # self.log("val_recall", self.val_recall, prog_bar=True)
        # self.val_precision(predicted_labels, true_labels)
        # self.log("val_precision", self.val_precision, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)
        # self.test_recall(predicted_labels, true_labels)
        # self.log("test_recall", self.test_recall)
        # self.test_precision(predicted_labels, true_labels)
        # self.log("test_precision", self.test_precision)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer