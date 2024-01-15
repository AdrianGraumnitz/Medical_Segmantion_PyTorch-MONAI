import torch
from torch.nn import functional
from tqdm.auto import tqdm
from monai.data import DataLoader
from monai.losses import DiceLoss
import monai
import numpy as np
import utils
from pathlib import Path
from torch.utils import tensorboard


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
               device: torch.device,
               ) -> tuple[float, float]:
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
    train_loss: float= 0
    train_metric: float = 0
    
    model.train()
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
        metric: float = dice_metric(y_preds, label)
        train_metric: float
        train_metric += metric
        print(f'Step {batch + 1} of {len(dataloader)} | train loss: {loss:.4f} | train metric {metric:.4f}')
        
    train_loss /= len(dataloader)
    train_metric /= len(dataloader)
    
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
    test_loss: float = 0
    test_metric: float = 0
    
    model.eval()
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
            metric: float = dice_metric(y_preds, label)
            test_metric: float
            test_metric += metric
            print(f'Step: {batch + 1} of {len(dataloader)} | test loss: {loss:.4f} | test metric: {metric:.4f}')
        test_loss /= len(dataloader)
        test_metric /= len(dataloader)
    
    return test_loss, test_metric

def train(model: torch.nn.Module,
          train_dataloader: monai.data.DataLoader,
          test_dataloader: monai.data.DataLoader,
          loss_fn: monai.losses,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          target_dir: str,
          model_name: str,
          writer: torch.utils.tensorboard.writer.SummaryWriter) -> dict[str, list]:
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
    
    best_metric: float = -1
    best_metric_epoch: int = -1
    save_loss_train: list = []
    save_metric_train: list = []
    save_loss_test: list = []
    save_metric_test: list = []
    step: int = 0
    model_path = Path(target_dir) / model_name
    best_metric_dir: Path = Path(target_dir) / 'best_metric.txt'
    
    if model_path.is_file():
        print(f'[INFO] Loading {model_name} weights from {model_path}.')
        model = utils.load_weights(model = model,
                                   target_dir = model_path)
    if best_metric_dir.is_file():
        print(f'[INFO] Loading best metric from {best_metric_dir}')
        best_metric = utils.load_best_metric(target_dir = best_metric_dir)
    for epoch in tqdm(range(epochs)):
        step += 1
        train_loss: float
        train_metric: float
        train_loss, train_metric = train_step(model = model,
                                              dataloader = train_dataloader,
                                              loss_fn = loss_fn,
                                              optimizer = optimizer,
                                              device = device                                             
                                             )
        print(f'\n[INFO] E: {epoch} | Epoch train loss: {train_loss:.4f} | Epoch train metric: {train_metric:.4f}\n{50 * "-"}\n')
        save_loss_train.append(train_loss)
        save_metric_train.append(train_metric)
        utils.save_metric(name = 'train_loss',
                          target_dir = target_dir,
                          metric_list = save_loss_train)
        utils.save_metric(name = 'train_metric',
                          target_dir = target_dir,
                          metric_list = save_metric_train)
        
        
        if epoch % 5 == 0:
            test_loss: float
            test_metric: float
            test_loss, test_metric = test_step(model = model,
                                            dataloader = test_dataloader,
                                            loss_fn = loss_fn,
                                            device = device)
            print(f'\n[INFO] E: {epoch} | Epoch test loss: {test_loss:.4f} | Epoch test metric: {test_metric:.4f}\n{50 * "-"}\n')
            save_loss_test.append(test_loss)
            save_metric_test.append(test_metric)
            utils.save_metric(name = 'test_loss',
                            target_dir = target_dir,
                            metric_list = save_loss_test)
            utils.save_metric(name = 'test_metric',
                            target_dir = target_dir,
                            metric_list = save_metric_test)
            if test_metric > best_metric:
                best_metric = test_metric
                best_metric_epoch = epoch + 1
                utils.save_best_metric(target_dir = target_dir,
                                       best_metric = best_metric)
                utils.save_best_metric_info(target_dir = target_dir,
                                    best_metric = best_metric,
                                    best_metric_epoch = best_metric_epoch)
                utils.save_model(model = model,
                                target_dir = target_dir,
                                model_name = model_name)
            results['test_loss'].append(test_loss)
            results['test_metric'].append(test_metric)
        
        results['train_loss'].append(train_loss)
        results['train_metric'].append(train_metric)

        writer.add_scalars(main_tag = 'Loss',
                           tag_scalar_dict = {'train_loss': train_loss,
                                              'test_loss': test_loss},
                           global_step = step)
        writer.add_scalars(main_tag = 'Dice metric',
                           tag_scalar_dict = {'train_metric': train_metric,
                                              'test_metric': test_metric},
                           global_step = step)
        #writer.add_graph(model = model,
                         #input_to_model = torch.randn(1, 1, 128, 128, 64))
    writer.flush()
    writer.close()
    return results
