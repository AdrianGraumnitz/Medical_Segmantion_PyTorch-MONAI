import torch
from pathlib import Path
import numpy as np
from datetime import datetime
from torch.utils import tensorboard
from datetime import datetime
import nibabel as nib

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    '''
    Saves a PyTorch model to a target directory.
    
    Args:
        model: Target PyTorch model to save.
        target_dir: Directory for saving the model to
        model_name: Filename for the saved model. Should include either '.pth' or '.pt' as the file extension
    '''
    
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents = True, exist_ok = True)
    
    assert model_name.endswith('.pth') or model_name.endswith('.pt'), '[INFO] model_name should end with \'.pth\' or \'.pt\''
    model_save_path = target_dir_path / model_name
    
    print(f'[INFO] Saving model to: {model_save_path}')
    torch.save(obj = model.state_dict(),
               f = model_save_path)

def load_weights(model: torch.nn.Module,
                 target_dir: Path) -> torch.nn.Module:
    '''
    Loads pre-trained weights into a PyTorch model from a specified directory
    
    Args:
        model: PyTorch model to which will be loaded
        target_dir: Directory containing the pre-trained weights
    
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
                target_dir: str,
                metric_list: list):
    '''
    Save a metric to a NumPy binary file.
    
    Args:
        name: Name of the metric, used as a part of the saved file's name
        target_dir: Directory path where the metric file will be saved
        metric_list: List of metric values to be saved
    '''
    #print(f'[INFO] Saving metric {name} to {target_dir}.')
    file_name: str = name + '.npy'
    target_dir = Path(target_dir) / file_name
    
    np.save(file = target_dir,
            arr = metric_list)
    #print(f'[INFO] Metric saved successfully')

def save_best_metric(target_dir: str,
                     best_metric: float):
    '''
    Save the best metric as a text file
    
    Args:
        target_dir: Directory path where the best metric file will be saved.
        best_metric: best_metric: Value of the best metric achieved.
    '''
    best_metric_dir: Path = Path(target_dir) / 'best_metric.txt'
    with open(best_metric_dir, 'w') as file:
        file.write(str(best_metric)) 

def save_best_metric_info(target_dir: str,
                     best_metric: float,
                     best_metric_epoch: int):
    '''
    Save the best metric, the epoch at which the best metric was achieved, and the timestamp in a text file to gain insights into the model's performance over time.    
    Args:
        target_dir: Directory path where the best_metric_info file will be saved.
        best_metric: Value of the best metric achieved.
        best_metric_epoch: Epoch at which the best metric was achieved.
    '''
    target_dir = Path(target_dir)
    with open((target_dir / 'best_metric_info.txt'), 'a') as file:
                file.write(f'Best metric: {str(best_metric)} | Best metric epoch: {str(best_metric_epoch)} | Datetime: {datetime.now().strftime("%H:%M:%S %Y-%m-%d")}\n')

def load_best_metric(target_dir: str) -> float:
    '''
    Load the best metric value from a text file
    
    Args:
        target_dir: Directory path where best metric file is located.
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
        model_name: Name of the model
        extra: Additional information to append to the log directory.
        
    Returns:
        tensorboard.SummaryWriter: SummaryWriter object
    '''
    
    timestamp: datetime = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path.cwd().parent / 'runs' / timestamp / model_name / extra
    print(f'[INFO Created SummaryWriter saving to {log_dir}]')
    
    return tensorboard.SummaryWriter(log_dir = log_dir)

def save_nifti(prediction_list: list, 
               out_dir: str, 
               name: str = 'prediction'):
    '''
    Save a 3D prediction tensor as a Nifit file.
    
    Args:
        prediction: List of 3D tensors to be saved as NIfTI files.
        out_dir: Directory where the NIfTI files will be saved.
        name: Prefix for the saved NIfTI files. Defaults to 'prediction'.
    '''
    for i, data in enumerate(prediction_list):
        print(f'[INFO] Saving {name}_{i} Nifti file to {out_dir}')
        nib.save(nib.Nifti1Image(data.squeeze().cpu().numpy().astype(float), affine = None), Path(out_dir) / f'{name}_{i}.nii.gz')

def set_seed(seed: int = 42):
    '''
    Set the random seed for PyTorch operations to ensure reproducibility.
    
    Args:
        seed: The desired random seed.
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f'[INFO] Set random seed to {seed}')