import os
from pathlib import Path
from glob import glob
import shutil
from tqdm.auto import tqdm
import dicom2nifti
import nibabel as nib
import numpy as np
import torch
from monai import transforms
from monai.data import DataLoader, Dataset, CacheDataset
import monai



in_dir = Path(__file__).parent
out_dir = in_dir / 'out_dir'

def create_groups(in_dir: str, 
                  out_dir: str, 
                  numb_slices: int):
    '''
    Move files from the input directory into newly created folders
    
    Args:
            in_dir: Path to input directory
            out_dir: Path to target directory
            numb_slices: Number of the size of the data collection
    '''
    for patient_path in Path(in_dir).glob('*'):
        patient_name = patient_path.name
        number_folders = len(list(patient_path.glob('*'))) // numb_slices
        
        for folder_index in range(number_folders):
            output_path = out_dir / f'{patient_name}_{str(folder_index)}'
            output_path.mkdir(parents = True, exist_ok = True)
            
            for i, file in enumerate(patient_path.glob('*')):
                if i == numb_slices + 1:
                    break
            
                shutil.move(file, output_path)
                print(f'[INFO] File {file.name} was moved to {output_path}')
                
#create_groups(in_dir = in_dir,
             # out_dir = out_dir,
              #numb_slices = 20)

def dicom2nifti(in_dir: str, 
                out_dir: str):
    '''
    Converts dicom files into nifti files and move them to an other directory
    
    Args:
        in_dir: Path to input directory
        out_dir: Path to target directory
    '''
    for folder in tqdm(Path(in_dir).glob('*')):
        patient_name = folder.name
        dicom2nifti.dicom_series_to_nifti(folder, Path(out_dir / patient_name) + '.nii.gz')
           
def find_empty(in_dir: str) -> list:
    '''
    Searching for not empty nifti files
    
    Args:
        in_dir: Path to input directory
    
    Returns:
        list_patients: A list with not empty nifti files
    '''

    list_patients = []
    for patient_path in Path(in_dir).glob('*'):
        img = nib.load(patient_path)
        
        if len(np.unique(img.get_fdata())) > 2:
            print(f'[INFO] Nifti file: {patient_path.name} with content')
            list_patients.append(patient_path.name)
        else:
            patient_path.unlink() # Delete empty files
            
    return list_patients

def set_seed(seed: int = 42):
    '''
    Set the random seed for PyTorch operations to ensure reproducibility.
    
    Args:
        seed: The desired random seed. Default is 42.
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def prepare(in_dir: str, 
            pixdim: tuple = (1.5, 1.5, 1.0), 
            a_min: float = - 200, 
            a_max: float = 200, 
            b_min: float = 0.0, 
            b_max: float = 1.0, 
            clip: bool = True , 
            spatial_size: list = [128, 128, 64], 
            cache = False, 
            manual_seed: bool = False, 
            num_workers: int = os.cpu_count()) -> tuple[monai.data.DataLoader, monai.data.DataLoader]:
    '''
    Creates datasets, applies transforms, and sets up dataloaders.
    
    Arg: 
        in_dir: Path to input directory.
        pixdim: Scales the size of the voxels and their spacing, measured from the center of one voxel to the center of the next.
        a_min: Minimum intensity value for rescaling.
        a_max: Maximum intensity value for rescaling.
        b_min: Minimum intensity value after scaling.
        b_max: Maximum intensity value after scaling.
        clip: If True, clips the intensity values to the range [b_min, b_max] after scaling.
        spatial_size: Target spatial_size of the processed data, e.g. [depth, height, width],
        cache: If True, enables caching of processed data to improve loading speed.
        manual_seed: If True, sets a manual seed for reproducibility.
        num_workers: Number of parrale workers for data loading.
        
    Returns: 
        A tuple containing dataloaders
        
    '''
    if manual_seed:
        set_seed()
    
    path_train_volumes = sorted(Path(in_dir / 'nifti_files' / 'train_volumes').glob('*.nii.gz'))
    path_train_segmentations = sorted(Path(in_dir / 'nifti_files' / 'train_segmentations').glob('*.nii.gz'))
    
    path_test_volumes = sorted(Path(in_dir / 'nifti_files' / 'test_volumes').glob('*nii.gz'))
    path_test_segmentations = sorted(Path(in_dir / 'nifti_files' / 'test_segmentations').glob('*.nii.gz'))  
    
    train_files = [{'vol': image_name, 'seg': image_label} for image_name, image_label in zip(path_train_volumes, path_train_segmentations)]
    test_files = [{'vol': image_name, 'seg': image_label} for image_name, image_label in zip(path_test_volumes, path_test_segmentations)]
    
    train_transform = transforms.Compose([
        transforms.LoadImaged(keys = ['vol', 'seg']),
        transforms.EnsureChannelFirstd(keys = ['vol', 'seg']),
        transforms.Spacingd(keys = ['vol', 'seg'], pixdim = pixdim, mode = ('bilinear', 'nearest')),
        transforms.Orientationd(keys = ['vol', 'seg'], axcodes = 'RAS'),
        transforms.ScaleIntensityRange(keys = ['vol', 'seg'], a_min = a_min, a_max = a_max, b_min = b_min, b_max = b_max, clip = clip),
        transforms.CropForegroundd(keys = ['vol', 'seg'], source_key = 'vol'),
        transforms.Resized(keys = ['vol', 'seg'], spatial_size = spatial_size),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.LoadImaged(keys = ['vol', 'seg']),
        transforms.EnsureChannelFirstd(keys = ['vol', 'seg']),
        transforms.Spacingd(keys = ['vol', 'seg'], pixdim = pixdim, mode = ('bilinear', 'nearest')),
        transforms.Orientationd(keys = ['vol', 'seg'], axcodes = 'RAS'),
        transforms.ScaleIntensityRange(keys = ['vol', 'seg'], a_min = a_min, a_max = a_max, b_min = b_min, b_max = b_max, clip = clip),
        transforms.CropForegroundd(keys = ['vol', 'seg'], source_key = 'vol'),
        transforms.Resized(keys = ['vol', 'seg'], spatial_size = spatial_size),
        transforms.ToTensor()
    ])
    
    
    
    if cache:
        train_dataset = CacheDataset(data = train_files,
                                     transform = train_transform,
                                     cache_rate = 0.1)
        train_dataloader = DataLoader(dataset = train_dataset,
                                      batch_size = 1,
                                      shuffle = True,
                                      num_workers = num_workers)
        test_dataset = CacheDataset(data = test_files,
                                    transform = test_transform,
                                    cache_rate = 0.1)
        test_dataloader = DataLoader(dataset = test_dataset,
                                     batch_size = 1,
                                     shuffle = False,
                                     num_workers = num_workers)
        
    else:
        train_dataset = Dataset(data = train_files,
                                transform = train_transform)
        train_dataloader = DataLoader(dataset = train_dataset,
                                      batch_size = 1,
                                      shuffle = True,
                                      num_workers = num_workers)
        test_dataset = Dataset(data = test_files,
                               transform = test_transform)
        test_dataloader = DataLoader(dataset = test_dataset,
                                     batch_size = 1,
                                     Shuffle = False,
                                     num_workers = num_workers)
    
    return train_dataloader, test_dataloader
    