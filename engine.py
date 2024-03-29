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
import data_setup
from monai import inferers
import monai


def dice_metric(prediction: torch.Tensor,
                label: torch.Tensor) -> float:
    '''
    Calculates the Dice accuracy for segmentation tasks.
    
    Args:
        prediction (torch.Tensor): The model's segmentation prediction
        label (torch.Tensor): The ground truth segmentation label
    
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
        class_counts (int): Counts of pixels for each class
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
        model (torch.nn.Module): The neural network model
        dataloader (monai.data.DataLoader): The train dataloader
        loss_fn (monai.losses): The loss function
        optimizer (torch.optim.Optimizer): The optimizer for updating the parameters
        device (torch.device): cpu or cuda
        
    Returns:
        tuple[train_loss (float): contains the mean loss of the train step,
              train_metric (float): contain the mean dice accuracy of the train_step
    '''
    train_loss: float= 0
    train_metric: float = 0
    
    model.train()
    for batch, data in enumerate(dataloader):
        
        input: torch.Tensor
        label: torch.Tensor
        input, label = data['vol'].to(device), data['seg'].to(device)
        
        y_logits: torch.Tensor = model(input)
        loss: torch.Tensor = loss_fn(y_logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss: float
        train_loss += loss.item()
        metric: float = dice_metric(y_logits, label)
        train_metric: float
        train_metric += metric
        print(f'Step {batch + 1} of {len(dataloader)} | train loss: {loss:.4f} | train metric: {100 * metric:.2f}%')
        
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
        model (torch.nn.Module): The neural network model
        dataloader (monai.data.DataLoader): The test dataloader
        loss_fn (monai.losses): The loss function
        device torch.device: cpu or cuda
     
     Returns:
        tuple[test_loss (float): contains the mean loss of the test_step,
              test_metric (float): contain the mean dice accuracy of the test step]
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
            loss: torch.Tensor = loss_fn(y_logits, label)
            test_loss: float
            test_loss += loss.item()
            metric: float = dice_metric(y_logits, label)
            test_metric: float
            test_metric += metric
            print(f'Step: {batch + 1} of {len(dataloader)} | test loss: {loss:.4f} | test metric: {100 * metric:.2f}%')
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
          target_dir: Path,
          model_name: str,
          writer: torch.utils.tensorboard.writer.SummaryWriter,
          manual_seed: bool = False) -> dict[str, list]:
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
        target_dir (str or Path): Directory for saving the trained model.
        model_name (str): Filename for the saved model (should include '.pth' or '.pt' extension).
        writer (torch.utils.tensorboard.writer.SummaryWriter): Creates a summary writer instance for tensorboard visualisation
        manual_seed (bool): If True, sets a manual seed for better reproducibility.
        
    Returns:
        dict[str, list]: Dictionary containing lists of training and testing loss/metric values.
    '''
    
    if manual_seed:
        utils.set_seed()
        
    
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
        print(f'\n[INFO] E: {epoch} | Epoch train loss: {train_loss:.4f} | Epoch train metric: {100 * train_metric:.2f}%\n{50 * "-"}\n')
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
            print(f'\n[INFO] E: {epoch} | Epoch test loss: {test_loss:.4f} | Epoch test metric: {100 * test_metric:.2f}%\n{50 * "-"}\n')
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

def perform_inference(model: torch.nn.Module,
                      test_patient: torch.Tensor,
                      roi_size: tuple = (128, 128, 64),
                      sw_batch_size: int = 4,
                      device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Performs inference using the sliding window technique on the given input volume.

    Args:
        model (torch.nn.Module): The PyTorch model used for prediction.
        test_patient (torch.Tensor): The input volume on which to perform inference.
        roi_size (tuple[int, int, int]): Region of interest size. Default is (128, 128, 64).
        sw_batch_size (int): Sliding window batch size. Default is 4.
        device (torch.device): The device on which to perform inference.

    Returns:
        tuple[torch.Tensor: The output of the model,
            torch.Tensor: The predicted segmentation result.]
    '''
    with torch.inference_mode():
        
        t_volume = test_patient['vol'].to(device)
        test_outputs = inferers.sliding_window_inference(inputs = t_volume,
                                                         roi_size = roi_size,
                                                         sw_batch_size = sw_batch_size,
                                                         predictor = model
                                                        )
        prediction = torch.softmax(test_outputs, dim = 1).argmax(dim = 1).unsqueeze(dim = 0)
        
        label_shape = test_patient["seg"].shape if "seg" in test_patient else "No label shape"
        print(f'[INFO]\nImage shape: {test_patient["vol"].shape}\nLabel shape: {label_shape}\nBinary segmentation shape: {test_outputs.shape}\nMulti segmentation shape: {prediction.shape}')
        
        return prediction, test_outputs

        

def create_prediction_list(model: torch.nn.Module,
                          test_dataloader: monai.data.DataLoader,
                          roi_size: tuple = (128, 128, 64),
                          sw_batch_size: int = 4) -> list:
    """
    Generates a list of predictions and their corresponding labels using the given model and test data loader. 
    If no labels are provided in the test data loader, the label list will be empty.

    Args:
        model (torch.nn.Module): The trained model used for inference.
        test_dataloader (monai.data.DataLoader): The data loader containing the test data.
        roi_size (tuple, optional): The size of the regions of interest (ROI) for sliding window inference. Default is (128, 128, 64).
        sw_batch_size (int, optional): The batch size for sliding window inference. Default is 4.

    Returns:
        tuple[list: list of predictions,
        list: list of corresponding labels]

    """
    prediction_list = []
    label_list = []
    model.eval()
    with torch.inference_mode():
        for data in test_dataloader:
            t_volume = data['vol']
            test_outputs = inferers.sliding_window_inference(inputs = t_volume,
                                                        roi_size = roi_size,
                                                        sw_batch_size = sw_batch_size,
                                                        predictor = model
                                                        )
            prediction = torch.softmax(test_outputs, dim = 1).argmax(dim = 1)
            prediction_list.append(prediction)
            if 'seg' in data:
                label_list.append(data['seg'].squeeze(dim = 0))
                
    return prediction_list, label_list
    

