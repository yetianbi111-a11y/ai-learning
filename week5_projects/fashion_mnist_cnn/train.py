import torch
import torch.nn as nn
import argparse
from model import get_model
from data import get_data_loaders, data_datasets  

def train_one_epoch(model,epoch_loader,optimizer,loss_fn,device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    for batch_idx,(x_batch,y_batch) in enumerate(epoch_loader):
        x_batch,y_batch = x_batch.to(device),y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss_value = loss_fn(logits,y_batch)
        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.item() * x_batch.shape[0]
        pred = logits.argmax(dim=1)
        total_correct += pred.eq(y_batch).sum().item()

    return total_loss / len(epoch_loader.dataset), total_correct / len(epoch_loader.dataset)