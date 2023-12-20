import torch
from torch.nn import functional
from tqdm.auto import tqdm
from monai.data import DataLoader
from monai.losses import DiceLoss
import monai
import numpy as np

def dice_metric(prediction: torch.Tensor,
                label: torch.Tensor) -> float:
    '''
    Calculates the Dice accuracy for segmentation tasks.
    
    Args:
        prediction: The model's segmentation prediction
        label: The ground truth segmentation label
    
    Returns:
        value: Dice accuracy
    '''
    dice_value = DiceLoss(to_onehot_y = True, sigmoid = True, squared_pred = True)
    value = 1 - dice_value(prediction, label).item()
    
    return value

def calculate_weights(class_counts: int):
    '''
    Calculate class weights for imbalanced class distribution in training data.
    
    Args:
        class_counts: Counts of pixels for each class
    Returns:
        torch.Tensor: A tensor containing the calculated weights. These weights can be used as an argument in the weight parameter of the cross-entropy loss function.

    '''
    count = np.array(class_counts)
    summ = count.sum()
    weights = count / summ
    weights = 1 / weights
    summ = weights / summ
    
    return torch.tensor(weights, dtype = torch.float32)

def train_step(model: torch.nn.Module,
               dataloader: monai.data.DataLoader,
               loss_fn: monai.losses,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> tuple[float, float]:
    '''
     Perform a training step using the provided model, dataloader, loss function, optimizer, and device.
     
     Args:
        model: The neural network model
        dataloader: The train dataloader
        loss_fn: The loss function
        optimizer: The optimizer for updating the parameters
        device: cpu or cuda
        
    Returns:
        train_loss: contains the mean loss of the train step
        train_metric: contain the mean dice accuracy of the train_step
    '''
    model.train()
    train_loss, train_acc = 0, 0
    train_loss = 0
    
    for batch, data in enumerate(dataloader):
        
        input, label = data['vol'].to(device), data['seg'].to(device)
        y_logits = model(input)
        y_preds = torch.softmax(y_logits, dim = 1)
        
        loss = loss_fn(y_logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_metric = dice_metric(y_preds, label)
        
    train_loss /= len(dataloader)
    train_metric /= len(dataloader)
    
    print(f'[INFO] Batch: {batch} | Train loss: {train_loss} | Test metric: {train_metric}')
    
    return train_loss, train_metric

def test_step(model: torch.nn.Module,
              dataloader: monai.data.DataLoader,
              loss_fn: monai.losses,
              device: torch.device):
    '''
     Perform a test step using the provided model, dataloader, loss function, optimizer, and device.
     
     Args:
        model: The neural network model
        dataloader: The test dataloader
        loss_fn: The loss function
        device: cpu or cuda
     
     Returns:
        test_loss: contains the mean loss of the test_step
        test_metric: contain the mean dice accuracy of the test step
    '''
    model.eval()
    test_loss = 0
    test_metric = 0
    
    with torch.inference_mode():
        for batch, data in enumerate(dataloader):
            
            input, label = data['vol'].to(device), data['seg'].to(device)            
            y_logits = model(input)
            y_preds =torch.softmax(y_logits, dim = 1)
            
            loss = loss_fn(y_logits, label)
            
            test_loss += loss.item()
            test_metric = dice_metric(y_preds, label)
    
    return test_loss, test_metric

def train(model: torch.nn.Module,
          train_dataloader: monai.data.DataLoader,
          test_dataloader: monai.data.DataLoader,
          loss_fn: monai.losses,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device):
    
    results = {'train_loss': [],
               'test_loss': [],
               'train_metric': [],
               'test_metric': []}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_metric = train_step(model = model,
                                              dataloader = train_dataloader,
                                              loss_fn = loss_fn,
                                              optimizer = optimizer,
                                              device = device                                             
                                             )
        test_loss, test_metric = test_step(model = model,
                                           dataloader = test_dataloader,
                                           loss_fn = loss_fn,
                                           device = device)
        
        results['train_loss'].append(train_loss)
        results['train_metric'].append(train_metric)
        results['test_loss'].append(test_loss)
        results['test_metric'].append(train_metric)
    
    return results
