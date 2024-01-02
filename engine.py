import torch
from torch.nn import functional
from tqdm.auto import tqdm
from monai.data import DataLoader
from monai.losses import DiceLoss
import monai
import numpy as np
import utils

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
    dice_value: DiceLoss = DiceLoss(to_onehot_y = True, sigmoid = True, squared_pred = True)
    value: float = 1 - dice_value(prediction, label).item()
    
    return value

def calculate_weights(class_counts: int) -> torch.Tensor:
    '''
    Calculate class weights for imbalanced class distribution in training data.
    
    Args:
        class_counts: Counts of pixels for each class
    Returns:
        torch.Tensor: A tensor containing the calculated weights. These weights can be used as an argument in the weight parameter of the cross-entropy loss function.

    '''
    count: np.ndarray = np.array(class_counts)
    summ: float = count.sum()
    weights: np.ndarray = count / summ
    weights: np.ndarray = 1 / weights
    summ: float = weights / summ
    
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
    train_loss: float= 0
    train_metric: float = 0
    
    for batch, data in enumerate(dataloader):
        
        input: torch.Tensor
        label: torch.Tensor
        input, label = data['vol'].to(device), data['seg'].to(device)
        y_logits: torch.Tensor = model(input)
        y_preds: torch.Tensor = torch.softmax(y_logits, dim = 1)
        
        loss: torch.Tensor = loss_fn(y_logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss: float
        train_loss += loss.item()
        train_metric: float
        train_metric += dice_metric(y_preds, label)
        
    train_loss /= len(dataloader)
    train_metric /= len(dataloader)
    
    print(f'[INFO] Batch: {batch} | Train loss: {train_loss} | Train metric: {train_metric}')
    
    return train_loss, train_metric

def test_step(model: torch.nn.Module,
              dataloader: monai.data.DataLoader,
              loss_fn: monai.losses,
              device: torch.device) -> tuple[float, float]:
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
    test_loss: float = 0
    test_metric: float = 0
    
    with torch.inference_mode():
        for batch, data in enumerate(dataloader):
            input: torch.Tensor
            label: torch.Tensor
            input, label = data['vol'].to(device), data['seg'].to(device)            
            y_logits: torch.Tensor = model(input)
            y_preds: torch.Tensor = torch.softmax(y_logits, dim = 1)
            
            loss: torch.Tensor = loss_fn(y_logits, label)
            
            test_loss: float
            test_loss += loss.item()
            test_metric: float
            test_metric += dice_metric(y_preds, label)
    print(f'[INFO] Testn loss: {test_loss} | Test metric: {test_metric}')
    return test_loss, test_metric

def train(model: torch.nn.Module,
          train_dataloader: monai.data.DataLoader,
          test_dataloader: monai.data.DataLoader,
          loss_fn: monai.losses,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          target_dir: str,
          model_name: str) -> dict[str, list]:
    '''
    Trains a PyTorch/Monai model using the provided data and configuration
    
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_dataloader (monai.data.DataLoader): DataLoader for the training dataset.
        test_dataloader (monai.data.DataLoader): DataLoader for the testing dataset.
        loss_fn (monai.losses): Loss function for optimization.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        epochs (int): Number of training epochs.
        device (torch.device): Device (cpu or cuda) on which to perform training.
        target_dir (str): Directory for saving the trained model.
        model_name (str): Filename for the saved model (should include '.pth' or '.pt' extension).
        
    Returns:
        dict[str, list]: Dictionary containing lists of training and testing loss/metric values.
    '''
    
    results: dict[str, list] = {'train_loss': [],
               'test_loss': [],
               'train_metric': [],
               'test_metric': []}
    
    for epoch in tqdm(range(epochs)):
        train_loss: float
        train_metric: float
        train_loss, train_metric = train_step(model = model,
                                              dataloader = train_dataloader,
                                              loss_fn = loss_fn,
                                              optimizer = optimizer,
                                              device = device                                             
                                             )
        #####################Save mode#############
        
        utils.save_model(model = model,
                         target_dir = target_dir,
                         model_name = model_name)
        
        ########################################
        
        test_loss: float
        test_metric: float
        test_loss, test_metric = test_step(model = model,
                                           dataloader = test_dataloader,
                                           loss_fn = loss_fn,
                                           device = device)
        
        results['train_loss'].append(train_loss)
        results['train_metric'].append(train_metric)
        results['test_loss'].append(test_loss)
        results['test_metric'].append(test_metric)
    
    return results
