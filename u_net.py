from monai.networks.layers import Norm
from monai.networks.nets import UNet
import torch
from torchinfo import summary

def unet(spatial_dims: int = 3,
         in_channels: int = 1,
         num_classes: int = 8,
         channels: tuple[int] = (16, 32, 64, 128, 256),
         strides: tuple[int] = (2, 2, 2, 2),
         num_res_units: int = 2,
         norm: Norm.BATCH = Norm.BATCH
         ) -> tuple:
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
        spatial_dims = spatial_dims,
        in_channels = in_channels,
        out_channels = num_classes,
        channels = channels,
        strides = strides,
        num_res_units = num_res_units,
        norm = norm
    ).to(device)
    print(summary(model = model,
                  input_size = (1, 1, 128, 128, 64),
                  col_names = ['input_size', 'output_size', 'num_params', 'trainable'],
                  row_settings = ['var_names']))
    
    return model, device