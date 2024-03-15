import os
from pathlib import Path
from glob import glob
import shutil
from tqdm.auto import tqdm
import dicom2nifti
import nibabel as nib
import numpy as np
import torch
from torch.nn import functional
from monai import transforms
from monai.data import DataLoader, Dataset, CacheDataset
import monai
import utils

def create_groups(in_dir: Path, 
                  out_dir: Path, 
                  numb_slices: int):
    '''
    Move files from the input directory into newly created folders
    
    Args:
            in_dir (str or Path): Path to input directory
            out_dir (str or Path): Path to target directory
            numb_slices (str or Path): Number of the size of the data collection
    '''
    
    for patient_path in Path(in_dir).glob('*'):
        patient_name: str = patient_path.name
        number_folders: int = len(list(patient_path.glob('*'))) // numb_slices
        for folder_index in range(number_folders):
            output_path: Path = out_dir / f'{patient_name}_{str(folder_index)}'
            output_path.mkdir(parents = True, exist_ok = True)
            
            for i, file in enumerate(patient_path.glob('*')):
                print('3. forschleife')
                if i == numb_slices + 1:
                    break
            
                shutil.remove(file, output_path)

def dcom2nifti(in_dir: Path, 
                out_dir: Path):
    '''
    Converts dicom files into nifti files and move them to an other directory
    
    Args:
        in_dir (str or Path): Path to input directory
        out_dir (str or Path): Path to target directory
    '''
    for folder in tqdm(Path(in_dir).glob('*')):
        patient_name: str = folder.name
        dicom2nifti.dicom_series_to_nifti(folder, Path(out_dir / f'{patient_name}.nii.gz'))
           
def find_empty(in_dir: Path) -> list:
    '''
    Searching for not empty nifti files
    
    Args:
        in_dir (str or Path): Path to input directory
    
    Returns:
        list_patients (list): A list with not empty nifti files
    '''

    list_patients: list = []
    for patient_path in Path(in_dir).glob('*'):
        img: nib = nib.load(patient_path)
        
        if len(np.unique(img.get_fdata())) > 2:
            print(f'[INFO] Nifti file: {patient_path.name} with content')
            list_patients.append(patient_path.name)
        else:
            patient_path.unlink() # Delete empty files
            
    return list_patients

def edit_label(in_dir: Path,
               out_dir: Path):
    '''
    Process NIfTI segmentation files in the input directory, apply value mapping for normalization, and save results.    Args:
    
    Args:    
        in_dir (str or Path): Path to the input directory containing segmentaion NIfTI files.
        out_dir (str or Path): Path to the output directory where processed segmentations will be saved.
    '''
    file: Path = Path(in_dir).glob('*.nii.gz')
    for nifti in file:
        print(f'[INFO] Mapping {nifti.name} to {out_dir} directory')
        image: nib = nib.load(nifti)
        tensor: torch = torch.as_tensor(image.get_fdata())
        value_mapping = {voxel: i for i, voxel in enumerate(torch.unique(tensor))}
        for original_val, mapped_val in value_mapping.items():
            tensor[tensor == original_val] = mapped_val
        nib.save(nib.Nifti1Image(tensor.numpy(), affine = None), Path(out_dir) / nifti.name)
        if (Path(out_dir) / nifti.name).is_file():
            Path(nifti).unlink()
    
def prepare_train_eval_data(in_dir: Path, 
            pixdim: tuple = (1.5, 1.5, 1.0), 
            a_min: float = - 200, 
            a_max: float = 1000, 
            b_min: float = 0.0, 
            b_max: float = 1.0, 
            clip: bool = True , 
            spatial_size: list = [128, 128, 64], 
            cache = False, 
            manual_seed: bool = False, 
            num_workers: int = os.cpu_count() // 4) -> tuple[monai.data.DataLoader, monai.data.DataLoader]:
    '''
    Creates datasets, applies transforms, and sets up dataloaders.
    
    Arg: 
        in_dir (str or Path): Path to input directory.
        pixdim (tuple[float, float, float]): Scales the size of the voxels and their spacing, measured from the center of one voxel to the center of the next.
        a_min (float): Minimum intensity value for rescaling.
        a_max (float): Maximum intensity value for rescaling.
        b_min (float): Minimum intensity value after scaling.
        b_max (float): Maximum intensity value after scaling.
        clip (bool): If True, clips the intensity values to the range [b_min, b_max] after scaling.
        spatial_size (list[int, int, int]): Target spatial_size of the processed data, e.g. [depth, height, width],
        cache (bool): If True, enables caching of processed data to improve loading speed.
        manual_seed (bool): If True, sets a manual seed for reproducibility.
        num_workers (int): Number of parrale workers for data loading.
        
    Returns: 
        tuple[train_dataloader (monai.data.DataLoader): Contain the Train dataloader,
              test_dataloader (monai.data.DataLoader): Contain the test dataloader]
        
    '''
    if manual_seed:
        utils.set_seed()
    
    path_train_volumes: list[Path] = sorted(Path(in_dir / 'train_volumes').glob('*.nii.gz'))
    path_train_segmentations: list[Path] = sorted(Path(in_dir / 'train_segmentations').glob('*.nii.gz'))
    
    path_test_volumes: list[Path] = sorted(Path(in_dir / 'test_volumes').glob('*nii.gz'))
    path_test_segmentations: list[Path] = sorted(Path(in_dir / 'test_segmentations').glob('*.nii.gz'))
    
    train_files: list[dict[Path, Path]] = [{'vol': image_name, 'seg': image_label} for image_name, image_label in zip(path_train_volumes, path_train_segmentations)]
    test_files: list[dict[Path, Path]] = [{'vol': image_name, 'seg': image_label} for image_name, image_label in zip(path_test_volumes, path_test_segmentations)]
    
    train_transform: transforms = transforms.Compose([
        transforms.LoadImaged(keys = ['vol', 'seg']),
        transforms.EnsureChannelFirstd(keys = ['vol', 'seg']),
        transforms.CropForegroundd(keys = ['vol', 'seg'], source_key = 'vol'),
        transforms.ScaleIntensityRanged(keys = ['vol'], a_min = a_min, a_max = a_max, b_min = b_min, b_max = b_max, clip = clip),
        transforms.Spacingd(keys = ['vol', 'seg'], pixdim = pixdim, mode = ('bilinear', 'nearest')),
        transforms.Orientationd(keys = ['vol', 'seg'], axcodes = 'RAS'),
        transforms.Resized(keys = ['vol', 'seg'], spatial_size = spatial_size, mode = ('bilinear', 'nearest')),
        transforms.RandAffined(keys = ['vol', 'seg'], spatial_size = spatial_size, rotate_range = (0, 0), scale_range = (0.9, 1.1), translate_range = (10, 10), mode = ('bilinear', 'nearest')),
        transforms.ToTensor()
    ])
    
    test_transform: transforms = transforms.Compose([
        transforms.LoadImaged(keys = ['vol', 'seg']),
        transforms.EnsureChannelFirstd(keys = ['vol', 'seg']),
        transforms.CropForegroundd(keys = ['vol', 'seg'], source_key = 'vol'),
        transforms.ScaleIntensityRanged(keys = ['vol'], a_min = a_min, a_max = a_max, b_min = b_min, b_max = b_max, clip = clip),
        transforms.Spacingd(keys = ['vol', 'seg'], pixdim = pixdim, mode = ('bilinear', 'nearest')),
        transforms.Orientationd(keys = ['vol', 'seg'], axcodes = 'RAS'),
        transforms.Resized(keys = ['vol', 'seg'], spatial_size = spatial_size, mode = ('bilinear', 'nearest')),
        transforms.RandAffined(keys = ['vol', 'seg'], spatial_size = spatial_size, rotate_range = (0, 0), scale_range = (0.9, 1.1), translate_range = (10, 10), mode = ('bilinear', 'nearest')),
        transforms.ToTensor()
    ])
    
    
    
    if cache:
        train_dataset: CacheDataset =   CacheDataset(data = train_files,
                                        transform = train_transform,
                                        cache_rate = 0.1)
        train_dataloader: DataLoader =  DataLoader(dataset = train_dataset,
                                        batch_size = 1,
                                        shuffle = True,
                                        num_workers = num_workers)
        test_dataset: CacheDataset = CacheDataset(data = test_files,
                                    transform = test_transform,
                                    cache_rate = 0.1)
        test_dataloader: DataLoader =   DataLoader(dataset = test_dataset,
                                        batch_size = 1,
                                        shuffle = False,
                                        num_workers = num_workers)
        
    else:
        train_dataset: Dataset = Dataset(data = train_files,
                                transform = train_transform)
        train_dataloader: DataLoader =  DataLoader(dataset = train_dataset,
                                        batch_size = 1,
                                        shuffle = True,
                                        num_workers = num_workers)
        test_dataset: Dataset = Dataset(data = test_files,
                                transform = test_transform)
        test_dataloader: DataLoader =   DataLoader(dataset = test_dataset,
                                        batch_size = 1,
                                        shuffle = False,
                                        num_workers = num_workers)
    
    return train_dataloader, test_dataloader
    
def prepare_test_data(in_dir: Path, 
            pixdim: tuple = (1.5, 1.5, 1.0), 
            a_min: float = - 200, 
            a_max: float = 1000, 
            b_min: float = 0.0, 
            b_max: float = 1.0, 
            clip: bool = True , 
            spatial_size: list = [128, 128, 64], 
            cache = False, 
            manual_seed: bool = False, 
            num_workers: int = os.cpu_count() // 4) -> tuple[monai.data.DataLoader, monai.data.DataLoader]:
    '''
    Creates datasets, applies transforms, and sets up dataloaders.
    
    Arg: 
        in_dir (str or Path): Path to input directory.
        pixdim (tuple[float, float, float]): Scales the size of the voxels and their spacing, measured from the center of one voxel to the center of the next.
        a_min (float): Minimum intensity value for rescaling.
        a_max (float): Maximum intensity value for rescaling.
        b_min (float): Minimum intensity value after scaling.
        b_max (float): Maximum intensity value after scaling.
        clip (bool): If True, clips the intensity values to the range [b_min, b_max] after scaling.
        spatial_size (list[int, int, int]): Target spatial_size of the processed data, e.g. [depth, height, width],
        cache (bool): If True, enables caching of processed data to improve loading speed.
        manual_seed (bool): If True, sets a manual seed for reproducibility.
        num_workers (int): Number of parrale workers for data loading.
        
    Returns: 
              test_dataloader (monai.data.DataLoader): Contain the test dataloader]
        
    '''    


    path_test_volumes: list[Path] = sorted(Path(in_dir / 'test_volumes').glob('*nii.gz'))
    print(path_test_volumes)
    test_files: list[dict[Path, Path]] = [{'vol': image_name} for image_name in path_test_volumes]

    test_transform: transforms = transforms.Compose([
            transforms.LoadImaged(keys = ['vol']),
            transforms.EnsureChannelFirstd(keys = ['vol']),
            transforms.CropForegroundd(keys = ['vol'], source_key = 'vol'),
            transforms.ScaleIntensityRanged(keys = ['vol'], a_min = a_min, a_max = a_max, b_min = b_min, b_max = b_max, clip = clip),
            transforms.Spacingd(keys = ['vol'], pixdim = pixdim, mode = ('bilinear')),
            transforms.Orientationd(keys = ['vol'], axcodes = 'RAS'),
            transforms.Resized(keys = ['vol'], spatial_size = spatial_size, mode = ('bilinear')),
            transforms.ToTensor()
        ])
    if cache:
        test_dataset: CacheDataset = CacheDataset(data = test_files,
                                        transform = test_transform,
                                        cache_rate = 0.1)
        test_dataloader: DataLoader =   DataLoader(dataset = test_dataset,
                                            batch_size = 1,
                                            shuffle = False,
                                            num_workers = num_workers)
    else:
        test_dataset: Dataset = Dataset(data = test_files,
                                transform = test_transform)
        test_dataloader: DataLoader =   DataLoader(dataset = test_dataset,
                                        batch_size = 1,
                                        shuffle = False,
                                        num_workers = num_workers)
    return test_dataloader

def rescale_predictions(prediction_list: list[torch.Tensor],
                        file_path: Path.glob):
    '''
    Rescales a list of predictions to match the dimensions of corresponding Nifti files.

    Args:
        prediction_list (list[torch.Tensor]): List of predictions to be rescaled.
        file_path (Path.glob): Generator of file paths to Nifti files.

    Returns:
        list[torch.Tensor]: List of rescaled predictions with dimensions matching the corresponding Nifti files.
    '''
    
    rescaled_prediction_list = []
    images = [image for image in file_path]
    
    if len(prediction_list) != len(images):
        print(f'[INFO] The length of the prediction list: {len(prediction_list)} dont match with the length of the number of files in the directory: {len(images)}')
        if len(prediction_list) == 1:
            print("[INFO] Only one dataset is being reshaped, not the entire set.")
            for image in images:
                image = torch.as_tensor(nib.load(image).get_fdata()).unsqueeze(dim = 0).unsqueeze(dim = 0)
                
                
                rescaled_prediction = functional.interpolate(input = prediction_list.to(torch.float),
                                                             size = image.shape[-3:],
                                                             mode = 'nearest')
                return rescaled_prediction
    else:
        print(f'[INFO] Length of prediction list: {len(prediction_list)}')
        print(f'[INFO] Number of image files in directory: {len(images)}')
        
        for image, prediction in zip(images, prediction_list):
            image = torch.as_tensor(nib.load(image).get_fdata()).unsqueeze(dim = 0).unsqueeze(dim = 0)
            prediction = prediction.unsqueeze(dim = 0)
            rescaled_prediction = functional.interpolate(input = prediction.to(torch.float),
                                                size = image.shape[-3:],
                                                mode = 'nearest')
            rescaled_prediction_list.append(torch.flip(rescaled_prediction, dims = (2,)))
        return rescaled_prediction_list

def move_nifti_files(in_dir: Path,
              out_dir = Path):
    '''
    Moves all files with the extension '.nii.gz' from the input directory
    to the output directory.

    Args:
        in_dir (Path): The input directory from which files will be moved.
        out_dir (Path): The output directory to which files will be moved.

    '''
    file: Path = Path(in_dir).glob('*.nii.gz')
    out_dir.mkdir(parents = True,
                  exist_ok = True)
    
    for nifti in file:
        nifti.rename(out_dir / nifti.name)
        print(f'[INFO] Moving {nifti.name} to {out_dir / nifti.name}')