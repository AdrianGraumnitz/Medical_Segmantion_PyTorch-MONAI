from monai.networks.layers import Norm
from monai.networks.nets import UNet
import torch
from torchinfo import summary
def unet(num_classes = 8) -> tuple:
    '''
    Unet model prepared for multisegmentation
    
    Args:
        num_classes (int): Number of classes for multisegmentation. Default is 8.
    
    Returns:
        tuple [model (torch.nn.Module):The neural network model, 
               device (torch.device): cpu or cuda]
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(
        spatial_dims = 3,
        in_channels = 1,
        out_channels = num_classes,
        channels = (16, 32, 64, 128, 256),
        strides = (2, 2, 2, 2),
        num_res_units = 2,
        norm = Norm.BATCH
    ).to(device)
    print(summary(model = model,
                  input_size = (1, 1, 128, 128, 64),
                  col_names = ['input_size', 'output_size', 'num_params', 'trainable'],
                  row_settings = ['var_names']))
    
    return model, device