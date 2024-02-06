"""
Institution: Institute of Computing - University of Campinas (IC/Unicamp)
Project: White Mold 
Description: Implements the neural network model SSD (Single Shot Detector) for step of inference.
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 05/02/2024
Version: 1.0
This implementation is based on DebuggerCafe: https://debuggercafe.com/custom-backbone-for-pytorch-ssd/
"""

# Download TorchVision repo to use some files from
# !pip install opencv-python
# !pip install torchmetrics
# !pip install torch
# !pip install torchvision
# !pip install albumentations

# Basic python and ML Libraries
import os
from datetime import datetime

import shutil
import random
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# xml library for parsing xml files
from xml.etree import ElementTree as et

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans

# for image augmentation
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Importing python modules
from manage_log import *

# Import python code from debugger_cafe
from debugger_cafe.datasets import * 
from debugger_cafe.model import * 
from debugger_cafe.inference import * 

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'
NEW_FILE = True

# ###########################################
# Application Methods
# ###########################################

# ###########################################
# Methods of Level 1
# ###########################################

def main():
    """
    Main method that perform training of neural network model.

    All values of the parameters used here are defined in the external file "wm_model_ssd_parameters.json".

    """

    # setting dictionary initial parameters for processing
    full_path_project = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-model-ssd'

    # getting application parameters 
    parameters_filename = 'wm_model_ssd_parameters.json'
    parameters = get_parameters(full_path_project, parameters_filename)

    # setting new values of parameters according of initial parameters
    set_input_image_folders(parameters)

    # getting last running id
    get_running_id(parameters)

    # setting output folder results
    set_results_folder(parameters)
    
    # creating log file 
    logging_create_log(
        parameters['inference_results']['log_folder'], 
        parameters['inference_results']['log_filename']
    )

    logging_info('White Mold Research')
    logging_info('Inference of the model SSD (Single Shot Detector)' + LINE_FEED)

    # getting device CUDA
    device = get_device(parameters)

    # creating new instance of parameters file related to current running
    save_processing_parameters(parameters_filename, parameters)

    # loading dataloaders of image dataset for processing
    # train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(parameters)
    
    # copying weights file produced by training step 
    copy_weights_file(parameters)

    # creating neural network model 
    model = get_neural_network_model(parameters, device)

    # inference the neural netowrk model
    inference_neural_network_model(parameters, device, model)

    # printing metrics results 
    

    # finishing model training 
    logging_info('')
    logging_info('Finished the inference of the model SSD (Single Shot Detector)' + LINE_FEED)


# ###########################################
# Methods of Level 2
# ###########################################

def get_parameters(full_path_project, parameters_filename):
    '''
    Get dictionary parameters for processing
    '''    
    # getting parameters 
    path_and_parameters_filename = os.path.join(full_path_project, parameters_filename)
    parameters = Utils.read_json_parameters(path_and_parameters_filename)


    # logging processing parameters 
    # logging_info(Utils.get_pretty_json(parameters) + LINE_FEED)   
    
    # saving current processing parameters in the log folder 
    # path_and_parameters_filename = os.path.join('log', log_filename + "-" + parameters_filename)
    # Utils.save_text_file(path_and_parameters_filename, \
    #                     Utils.get_pretty_json(parameters), 
    #                     NEW_FILE)

    # returning parameters 
    return parameters


def set_input_image_folders(parameters):
    '''
    Set folder name of input images dataset
    '''    
    
    # getting image dataset folder according processing parameters 
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    image_dataset_folder = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['input']['input_dataset']['input_dataset_path'],
        parameters['input']['input_dataset']['annotation_format'],
        input_image_size + 'x' + input_image_size,
    )

    # setting image dataset folder in processing parameters 
    parameters['processing']['image_dataset_folder'] = image_dataset_folder
    parameters['processing']['image_dataset_folder_train'] = \
        os.path.join(image_dataset_folder, 'train')
    parameters['processing']['image_dataset_folder_valid'] = \
        os.path.join(image_dataset_folder, 'valid')
    parameters['processing']['image_dataset_folder_test'] = \
        os.path.join(image_dataset_folder, 'test')

def get_running_id(parameters):
    '''
    Get last running id to calculate the current id
    '''    

    # setting control filename 
    running_control_filename = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        parameters['processing']['running_control_filename'],
    )

    # getting control info 
    running_control = Utils.read_json_parameters(running_control_filename)

    # calculating the current running id 
    running_control['last_running_id'] = int(running_control['last_running_id']) + 1

    # updating running control file 
    running_id = int(running_control['last_running_id'])

    # saving file 
    Utils.save_text_file(running_control_filename, \
                         Utils.get_pretty_json(running_control), 
                         NEW_FILE)

    # updating running id in the processing parameters 
    parameters['processing']['running_id'] = running_id

    # returning the current running id
    # return running_id

def set_results_folder(parameters):
    '''
    Set folder name of output results
    '''

    # creating results folders 
    main_folder = os.path.join(
        parameters['processing']['research_root_folder'],     
        parameters['inference_results']['main_folder']
    )
    parameters['inference_results']['main_folder'] = main_folder
    Utils.create_directory(main_folder)

    # setting and creating action folder of training
    action_folder = os.path.join(
        main_folder,
        parameters['inference_results']['action_folder']
    )
    parameters['inference_results']['action_folder'] = action_folder
    Utils.create_directory(action_folder)

    # setting and creating model folder 
    parameters['inference_results']['model_folder'] = parameters['neural_network_model']['model_name']
    model_folder = os.path.join(
        action_folder,
        parameters['inference_results']['model_folder']
    )
    parameters['inference_results']['model_folder'] = model_folder
    Utils.create_directory(model_folder)

    # setting and creating running folder 
    running_id = parameters['processing']['running_id']
    running_id_text = 'running-' + f'{running_id:04}'
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    parameters['inference_results']['running_folder'] = running_id_text + "-" + input_image_size + 'x' + input_image_size   
    running_folder = os.path.join(
        model_folder,
        parameters['inference_results']['running_folder']
    )
    parameters['inference_results']['running_folder'] = running_folder
    Utils.create_directory(running_folder)

    # setting and creating others specific folders
    processing_parameters_folder = os.path.join(
        running_folder,
        parameters['inference_results']['processing_parameters_folder']
    )
    parameters['inference_results']['processing_parameters_folder'] = processing_parameters_folder
    Utils.create_directory(processing_parameters_folder)

    weights_folder = os.path.join(
        running_folder,
        parameters['inference_results']['weights_folder']
    )
    parameters['inference_results']['weights_folder'] = weights_folder
    Utils.create_directory(weights_folder)

    metrics_folder = os.path.join(
        running_folder,
        parameters['inference_results']['metrics_folder']
    )
    parameters['inference_results']['metrics_folder'] = metrics_folder
    Utils.create_directory(metrics_folder)

    inferenced_image_folder = os.path.join(
        running_folder,
        parameters['inference_results']['inferenced_image_folder']
    )
    parameters['inference_results']['inferenced_image_folder'] = inferenced_image_folder
    Utils.create_directory(inferenced_image_folder)

    log_folder = os.path.join(
        running_folder,
        parameters['inference_results']['log_folder']
    )
    parameters['inference_results']['log_folder'] = log_folder
    Utils.create_directory(log_folder)

def get_device(parameters):
    '''
    Get device CUDA to train models
    '''    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    parameters['processing']['device'] = f'{device}'
    # print(f'Device: {device}')
    logging_info(f'Device: {device}')
    logging_info(f'')

    # print(f'torch.cuda.current_device(): {torch.cuda.current_device()}')
    # print(f'torch.cuda.get_device_name(): {torch.cuda.get_device_name()}')

    # returning current device 
    return device 

def save_processing_parameters(parameters_filename, parameters):
    '''
    Update parameters file of the processing
    '''    
    # setting full path and log folder  to write parameters file 
    path_and_parameters_filename = os.path.join(
        parameters['inference_results']['processing_parameters_folder'], 
        parameters_filename)

    # saving current processing parameters in the log folder 
    Utils.save_text_file(path_and_parameters_filename, \
                        Utils.get_pretty_json(parameters), 
                        NEW_FILE)

def copy_weights_file(parameters):
    '''
    Copying weights file to inference step
    '''

    Utils.copy_file_same_name(
        parameters['input']['inference']['weights_filename'],
        parameters['input']['inference']['weights_folder'],
        parameters['inference_results']['weights_folder']
    )

# def get_dataloaders(parameters):
#     '''
#     Get dataloaders of training, validation and testing from image dataset 
#     '''

#     # getting datasets 
#     train_dataset = create_train_dataset(
#         parameters['processing']['image_dataset_folder_train'], 
#         parameters['neural_network_model']['resize_of_input_image'], 
#         parameters['neural_network_model']['classes'], 
#     )
#     valid_dataset = create_valid_dataset(
#         parameters['processing']['image_dataset_folder_valid'],
#         parameters['neural_network_model']['resize_of_input_image'], 
#         parameters['neural_network_model']['classes'], 
#     )
#     logging.info(f'Getting datasets')
#     logging.info(f'   Number of training images  : {len(train_dataset)}')
#     logging.info(f'   Number of validation images: {len(valid_dataset)}')
#     logging.info(f'   Total                      : {len(train_dataset) + len(valid_dataset)}')
#     logging_info(f'')


#     # getting dataloaders
#     train_dataloader = create_train_loader(
#         train_dataset, 
#         parameters['neural_network_model']['number_workers']
#     )
#     valid_dataloader = create_valid_loader(
#         valid_dataset,
#         parameters['neural_network_model']['number_workers']
#     )
#     test_dataloader = None

#     # logging.info(f'Getting dataloaders')
#     # logging.info(f'Number of training images  : {len(train_dataloader)}')
#     # logging.info(f'Number of validation images: {len(valid_dataloader)}')

#     # returning dataloaders for processing 
#     return train_dataloader, valid_dataloader, test_dataloader 

def get_neural_network_model(parameters, device):
    '''
    Get neural network model
    '''      

    # Initialize the model and move to the computation device.
    # model = create_model(num_classes=NUM_CLASSES, size=RESIZE_TO)
    model = create_model_pytorchvision(
        parameters['neural_network_model']['classes'], 
        size=parameters['neural_network_model']['resize_of_input_image'],         
        nms=parameters['neural_network_model']['non_maximum_suppression'],
        pretrained=Utils.to_boolean_value(
            parameters['neural_network_model']['is_pre_trained_weights']
        )
    )
    
    logging.info(f'Moving model to device: {device}')

    model = model.to(device)

    logging.info(f'Creating neural network model')
    logging.info(f'{model}')
    logging_info(f'')

    # returning neural network model
    return model


# ###########################################
# Methods of Level 3
# ###########################################


# ###########################################
# Main method
# ###########################################
if __name__ == '__main__':
    main()
