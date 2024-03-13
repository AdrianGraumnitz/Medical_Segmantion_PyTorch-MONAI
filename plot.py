import torch
import torchmetrics
from monai import transforms
from skimage import measure
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from mlxtend import plotting

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

def plot_image_label_prediction(test_patient: torch.Tensor,
                                prediction: torch.Tensor,
                                test_outputs: torch.Tensor,
                                start_image_index: int = 50,
                                threshold: float = 0.53):
    """
    Plots images, labels, binary segmentations, and multi-segmentations for visual inspection.

    Args:
        test_patient (torch.Tensor): The tensor containing test data, including 'vol' (input volume) and 'seg' (true segmentation).
        prediction (torch.Tensor): The predicted segmentation result.
        test_outputs (torch.Tensor): The output tensor from the model.
        start_image_index (int): Index of the first image for visualization. Default is 50.
        threshold (float): Threshold for binary segmentation. Default is 0.53.
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
            
            if 'seg' in test_patient:
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
            else:
                plt.subplot(1, 4, 2)
                plt.title(f'Binary segmentation {i}')
                plt.imshow(test_outputs.detach().cpu()[0, 0, :, :, i + start_image_index], cmap = 'gray')
                plt.subplot(1, 4, 3)
                plt.title(f'Multi segmentation {i}')
                plt.imshow(prediction.detach().cpu()[0, 0, :, :, i + start_image_index], cmap = 'gray')
                plt.show()

def plot_confusion_matrix(prediction_list: list[torch.Tensor],
                          label_list: list[torch.Tensor],
                          class_names: list,
                          ):
    '''
    Plots a confusion matrix for multiclass segmentation using predicted and true labels.

    Args:
        prediction_list (list[torch.Tensor]): A list containing tensors of predicted segmentations.
        label_list (list[torch.Tensor]): A list containing tensors of true segmentations.
        class_names (list): A list containing the names of the classes for the confusion matrix.
    
    Return:
    - list: Containing the test predictions
    '''
        
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

def generate_mesh(rescaled_prediction: torch.Tensor,
                  threshold: float = 0.5) -> tuple[list, list]:
    '''
    Generates a mesh based on a predicted segmentation.
    
    Args:
        rescaled_prediction (torch.Tensor): A tensor containing the predicted segmentation.
        threshold (float): A threshold indicating at what value a voxel is considered part of the mesh. Default is 0.5.
    
    Returns:
        tuple: A tuple containing two lists:
            - A list of vertices representing the vertices of the generated mesh.
            - A list of faces representing the faces of the generated mesh.
    '''
    vertices_list, faces_list = [], []
    segment_masks = [(rescaled_prediction.squeeze().cpu().numpy() == i) for i in range(1, len(rescaled_prediction.unique()))]
    for mask in segment_masks:
        vertices, faces, _, _ = measure.marching_cubes(mask > threshold)
        vertices_list.append(vertices)
        faces_list.append(faces)
    return vertices_list, faces_list

def plot_mesh(vertices_list: list,
              faces_list: list,
              opacity: float = 0.5,
              class_names = None
                  ):
    '''
    Plots a mesh with multiple segments.
    
    Args:
        vertices_list (list): A list containing the vertices of each segment.
        faces_list (list): A list containing the faces of each segment.
        opacity (float): The transparent appearance of the mesh
    '''
    fig = go.Figure()
    for i in range(1, len(vertices_list) + 1):
        if class_names == None:
            fig.add_trace(
                go.Mesh3d(
                    x = vertices_list[i-1][:, 0],
                    y = vertices_list[i-1][:, 1],
                    z = vertices_list[i-1][:, 2],
                    i = faces_list[i-1][:, 0],
                    j = faces_list[i-1][:, 1],
                    k = faces_list[i-1][:, 2],
                    opacity = opacity,
                    name = f'Segment {i}',
                    showlegend = True
                )
            )
        else:
            fig.add_trace(
                go.Mesh3d(
                    x = vertices_list[i-1][:, 0],
                    y = vertices_list[i-1][:, 1],
                    z = vertices_list[i-1][:, 2],
                    i = faces_list[i-1][:, 0],
                    j = faces_list[i-1][:, 1],
                    k = faces_list[i-1][:, 2],
                    opacity = opacity,
                    name = class_names[i],
                    showlegend = True
                )
            )
    fig.update_layout(
        title = 'Mesh for every segment',
        scene = dict(
            xaxis = dict(title = 'X'),
            yaxis = dict(title = 'Y'),
            zaxis = dict(title = 'Z')
        )
    )
    fig.show()

