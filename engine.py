import torch
from torch.nn import functional
from tqdm.auto import tqdm
from monai.data import DataLoader
from monai.losses import DiceLoss
import monai

def dice_metric(prediction: torch.Tensor,
                label: torch.Tensor):
    
    dice_value = DiceLoss(to_onehot_y = True, sigmoid = True)

def train_step(model: torch.nn.Module,
               dataloader: monai.data.DataLoader,
               loss_fn: monai.losses,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> tuple[float, float]:
    model.train()
    train_loss, train_acc = 0, 0
    
    for batch, data in enumerate(dataloader):
        
        inputs, labels = data['vol'].to(device), data['seg'].to(device)
        y_logits = model(inputs)
        
        loss = loss_fn(y_logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        