import torch
from pathlib import Path
import numpy as np
from datetime import datetime
from torch.utils import tensorboard
from datetime import datetime
import nibabel as nib
import torchmetrics
from mlxtend import plotting
from monai import inferers, transforms
import monai
import matplotlib.pyplot as plt
import shutil
import numpy as np

def save_model(model: torch.nn.Module,
               target_dir: str or Path,
               model_name: str):
    '''
    Saves a PyTorch model to a target directory.
    
    Args:
        model (torch.nn.Module): Target PyTorch model to save.
        target_dir (str or Path): Directory for saving the model to
        model_name (str): Filename for the saved model. Should include either '.pth' or '.pt' as the file extension
    '''
    
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents = True, exist_ok = True)
    
    assert model_name.endswith('.pth') or model_name.endswith('.pt'), '[INFO] model_name should end with \'.pth\' or \'.pt\''
    model_save_path = target_dir_path / model_name
    
    print(f'[INFO] Saving model to: {model_save_path}')
    torch.save(obj = model.state_dict(),
               f = model_save_path)

def load_weights(model: torch.nn.Module,
                 target_dir: str or Path) -> torch.nn.Module:
    '''
    Loads pre-trained weights into a PyTorch model from a specified directory
    
    Args:
        model (torch.nn.Module): PyTorch model to which will be loaded
        target_dir (str or Path): Directory containing the pre-trained weights
    
    Returns:
        torch.nn.Module: PyTorch model with loaded weights
    '''
    model.load_state_dict(torch.load(target_dir))
    
    if model is not None:
        print('[INFO] Weights sucessfuly loaded.')
    else:
        print('[INFO] Failed to load weights')
        
    return model

def save_metric(name: str,
                target_dir: str or Path,
                metric_list: list):
    '''
    Save a metric to a NumPy binary file.
    
    Args:
        name (str): Name of the metric, used as a part of the saved file's name
        target_dir (str or Path): Directory path where the metric file will be saved
        metric_list (list): List of metric values to be saved
    '''
    #print(f'[INFO] Saving metric {name} to {target_dir}.')
    file_name: str = name + '.npy'
    target_dir = Path(target_dir) / file_name
    
    np.save(file = target_dir,
            arr = metric_list)
    #print(f'[INFO] Metric saved successfully')

def save_best_metric(target_dir: str or Path,
                     best_metric: float):
    '''
    Save the best metric as a text file
    
    Args:
        target_dir (str or Path): Directory path where the best metric file will be saved.
        best_metric (float): best_metric: Value of the best metric achieved.
    '''
    best_metric_dir: Path = Path(target_dir) / 'best_metric.txt'
    with open(best_metric_dir, 'w') as file:
        file.write(str(best_metric)) 

def save_best_metric_info(target_dir: str or Path,
                     best_metric: float,
                     best_metric_epoch: int):
    '''
    Save the best metric, the epoch at which the best metric was achieved, and the timestamp in a text file to gain insights into the model's performance over time.    
    Args:
        target_dir (str or Path): Directory path where the best_metric_info file will be saved.
        best_metric (float): Value of the best metric achieved.
        best_metric_epoch (int): Epoch at which the best metric was achieved.
    '''
    target_dir = Path(target_dir)
    with open((target_dir / 'best_metric_info.txt'), 'a') as file:
                file.write(f'Best metric: {str(best_metric)} | Best metric epoch: {str(best_metric_epoch)} | Datetime: {datetime.now().strftime("%H:%M:%S %Y-%m-%d")}\n')

def load_best_metric(target_dir: str or Path) -> float:
    '''
    Load the best metric value from a text file
    
    Args:
        target_dir (str or Path): Directory path where best metric file is located.
    '''
    best_metric: int = 0
    
    with open(target_dir, 'r') as file:
        for metric in file:
            best_metric = metric
            best_metric = float(best_metric)
    
    return best_metric
            
def create_writer(model_name: str,
                  extra: str = None) -> tensorboard.SummaryWriter:
    '''
    Create and return a Tensorboard SummaryWriter for logging.
    
    Args:
        model_name (str): Name of the model
        extra str (str): Additional information to append to the log directory.
        
    Returns:
        tensorboard.SummaryWriter: SummaryWriter object
    '''
    
    timestamp: datetime = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path.cwd().parent / 'runs' / timestamp / model_name / extra
    print(f'[INFO Created SummaryWriter saving to {log_dir}]')
    
    return tensorboard.SummaryWriter(log_dir = log_dir)

def save_nifti(prediction_list: list, 
               out_dir: str or Path, 
               name: str = 'prediction'):
    '''
    Save a 3D prediction tensor as a Nifit file.
    
    Args:
        prediction (list): List of 3D tensors to be saved as NIfTI files.
        out_dir (str or Path): Directory where the NIfTI files will be saved.
        name (str): Prefix for the saved NIfTI files. Defaults to 'prediction'.
    '''
    for i, data in enumerate(prediction_list):
        print(f'[INFO] Saving {name}_{i} Nifti file to {out_dir}')
        nib.save(nib.Nifti1Image(data.squeeze().cpu().numpy().astype(float), affine = None), Path(out_dir) / f'{name}_{i}.nii.gz')

def set_seed(seed: int = 42):
    '''
    Set the random seed for PyTorch operations to ensure reproducibility.
    
    Args:
        seed (int): The desired random seed.
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f'[INFO] Set random seed to {seed}')

def plot_confusion_matrix(model: torch.nn.Module,
                          test_dataloader: monai.data.DataLoader,
                          class_names: list,
                          roi_size: tuple = (128, 128, 64),
                          sw_batch_size: int = 4) -> list:
    '''
    Plots a confusion matrix for multiclass segmentation using predicted and true labels.

    Args:
    - model (torch.nn.Module): The PyTorch model used for prediction.
    - test_dataloader (monai.data.DataLoader): Dataloader for the test data set
    - class_names (list): A list of class names corresponding to the categories.
    - roi_size (tuple[int, int, int]): Region of interest size. Default is (128, 128, 64).
    - sw_batch_size (int): Sliding window batch size. Default is 4.
    
    Return:
    - list: Containing the test predictions
    '''
    
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
            label_list.append(data['seg'].squeeze(dim = 0))
        
    prediction_cat_tensor = torch.cat(prediction_list)
    label_cat_tensor = torch.cat(label_list)
    
    confmat = torchmetrics.ConfusionMatrix(task = 'multiclass',
                                        num_classes = len(class_names))
    confmat_tensor = confmat(preds = prediction_cat_tensor,
                            target = label_cat_tensor)

    plotting.plot_confusion_matrix(conf_mat = confmat_tensor.numpy(),
                                    figsize = (10, 7),
                                    show_absolute = True,
                                    show_normed = True,
                                    colorbar = True,
                                    class_names = class_names)
    return prediction_list 

def plot_image_label_prediction(test_patient: torch.Tensor,
                                prediction: torch.Tensor,
                                test_outputs: torch.Tensor,
                                start_image_index: int = 50,
                                threshold: float = 0.53):
    """
    Plots images, labels, binary segmentations, and multi-segmentations for visual inspection.

    Args:
    - test_patient (torch.Tensor): The tensor containing test data, including 'vol' (input volume) and 'seg' (true segmentation).
    - prediction (torch.Tensor): The predicted segmentation result.
    - test_outputs (torch.Tensor): The output tensor from the model.
    - start_image_index (int): Index of the first image for visualization. Default is 50.
    - threshold (float): Threshold for binary segmentation. Default is 0.53.
    """
        
    sigmoid_activation = transforms.Activations(sigmoid = True)
    
    test_outputs = sigmoid_activation(test_outputs)
    test_outputs = 1 - test_outputs
    test_outputs = test_outputs > threshold

    for i in range(5):
            plt.figure('check', (18, 6))
            plt.subplot(1, 4, 1)
            plt.title(f'Image {i}')
            plt.imshow(test_patient['vol'][0, 0, :, :, i + start_image_index], cmap = 'gray')
            plt.subplot(1, 4, 2)
            plt.title(f'Label {i}')
            plt.imshow(test_patient['seg'][0, 0, :, :, i + start_image_index], cmap ='gray')
            plt.subplot(1, 4, 3)
            plt.title(f'Binary segmentation {i}')
            plt.imshow(test_outputs.detach().cpu()[0, 0, :, :, i + start_image_index], cmap = 'gray')
            plt.subplot(1, 4, 4)
            plt.title(f'Multi segmentation {i}')
            plt.imshow(prediction.detach().cpu()[0, 0, :, :, i + start_image_index], cmap = 'gray')
            plt.show()

def number_of_classes(in_dir: str or Path) -> int:
    '''
    """
    Count the total number of unique classes present in Nifti volumes within the specified directory.

    Args:
    - in_dir (str or Path): The path to the directory containing Nifti files.

    Returns:
    - int: The total number of unique classes present in the Nifti volumes.
    """
    '''
    file: Path = Path(in_dir).glob('*.nii.gz')
    for nifti in file:
        image = nib.load(nifti)
        num_classes = torch.as_tensor(image.get_fdata()).unique()
        print(f'[INFO] Number of classes: {int(len(num_classes))}')
        return int(len(num_classes))
    
def remove_directory_recursive(in_dir: str or Path):
    '''
    Removes the target directory and all its contents recursively.

    Args:
    - in_dir (str or Path): Path to the target directory.
    '''
    if in_dir.is_dir():
        shutil.rmtree(in_dir)
        print(f'[INFO] The directory {in_dir} has been recursively removed.')
    else:
        print(f'[INFO] No directory exists under the path: {in_dir}.')

def plot_metric(train_loss: np.ndarray[float],
                train_metric: np.ndarray[float],
                test_loss: np.ndarray[float],
                test_metric: np.ndarray[float]):
    """
    Plots the metric data for training and testing.

    Args:
        train_loss (np.ndarray[float]): Numpy array containing the training loss values.
        train_metric (np.ndarray[float]): Numpy array containing the training metric values.
        test_loss (np.ndarray[float]): Numpy array containing the test loss values.
        test_metric (np.ndarray[float]): Numpy array containing the test metric values.
    """ 
    plt.figure(f'Results of {datetime.now().strftime("%d %B")}')
    plt.figure(figsize = (7, 3))
    plt.subplot(1, 2, 1)
    plt.title('Train Dice loss')
    x = [i + 1 for i in range(len(train_loss))]
    y = train_loss
    plt.xlabel('epoch')
    plt.plot(x, y)

    plt.subplot(1, 2, 2)
    plt.title('Train metric DICE')
    x = [i + 1 for i in range(len(train_metric))]
    y = train_metric
    plt.xlabel('epoch')
    plt.plot(x, y)

    plt.show()

    plt.figure(figsize = (7, 3))
    plt.subplot(1, 2, 1)
    plt.title('Test Dice loss')
    x = [i + 1 for i in range(len(test_loss))]
    y = test_loss
    plt.xlabel('epoch')
    plt.plot(x, y)

    plt.subplot(1, 2, 2)
    plt.title('Test metric DICE')
    x = [i for i in range(len(test_metric))]
    y = test_metric
    plt.xlabel('epoch')
    plt.plot(x, y)

    plt.show()