import torch
from pathlib import Path

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