import time
import torch
# !pip install -qU wandb
import wandb
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn.functional as F




def criterion(inputs, targets):
    loss = F.cross_entropy(inputs, targets)
    return loss

def create_confusion_matrix(preds, labels):
    cf_matrix = confusion_matrix(labels, preds)
    df_cm = pd.DataFrame(cf_matrix)
    plt.figure(figsize=(12, 7))
    s = sn.heatmap(df_cm, annot=True, fmt='g')
    s.set(xlabel='Estimated', ylabel='Real')
    plt.close('all')
    return s.get_figure()



def train_epoch(model, dataloader, optimizer, scheduler, epoch, CFG, log):
    model.train()
    running_loss = 0.0
    dataset_size = 0
    # use tqdm to track progress
    with tqdm(dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch} train")
        # Iterate over data.
        for inputs, targets in tepoch:
            inputs = inputs.to(CFG['device'])
            targets = targets.to(CFG['device'])
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            # loss
            loss = criterion(outputs, targets)
            # backward
            loss.backward()
            optimizer.step()
            if CFG['lr_scheduler'] == 'OneCycleLR':
                scheduler.step()
            # calculate epoch loss
            dataset_size += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / dataset_size
            # get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            # print statistics
            tepoch.set_postfix(loss=epoch_loss, lr=current_lr)
    log["train_loss"] = epoch_loss
    log["lr"] = current_lr
    return epoch_loss

def val_epoch(model, dataloader, scheduler, epoch, CFG, log):
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    running_corrects = 0
    total_targets_label = []
    total_outputs_label = []
    # use tqdm to track progress
    with tqdm(dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch} val")
        # Iterate over data.
        for inputs, targets in tepoch:
            inputs = inputs.to(CFG['device'])
            targets = targets.to(CFG['device'])
            # predict
            outputs = model(inputs)
            # outputs = self.process_pred(outputs, inputs)
            # loss
            loss = criterion(outputs, targets)
            # calculate epoch loss
            dataset_size += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / dataset_size
            # Get target and predicted labels for metrics
            targets_label = torch.max(targets, dim=1)[1]
            outputs_label = torch.max(outputs, dim=1)[1]
            total_outputs_label += outputs_label.tolist()
            total_targets_label += targets_label.tolist()
            # Accuracy
            running_corrects += torch.sum(outputs_label == targets_label).item()
            epoch_acc = running_corrects / dataset_size
            # print statistics
            tepoch.set_postfix(loss=epoch_loss, acc=epoch_acc)
    log['conf_matrix'] = wandb.Image(create_confusion_matrix(total_outputs_label, total_targets_label))
    log["val_loss"] = epoch_loss
    log['acc'] = epoch_acc
    if scheduler is not None:
        scheduler.step(epoch_loss)
    return epoch_loss

