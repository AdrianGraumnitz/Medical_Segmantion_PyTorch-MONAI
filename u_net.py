from monai.networks.layers import Norm
from monai.networks.nets import UNet
import torch
from torchinfo import summary
def unet() -> tuple:
    '''
    Unet model prepared for multisegmentation
    
    Returns:
        model:The neural network model
        device: cpu or cuda
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(
        spatial_dims = 3,
        in_channels = 1,
        out_channels = 8,
        channels = (16, 32, 64, 128, 256),
        strides = (2, 2, 2, 2),
        num_res_units = 2,
        norm = Norm.BATCH
    ).to(device)
    
    return model, device