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

# Importing python modules$2023#@

from common.manage_log import *
from common.tasks import Tasks
from common.entity.AnnotationsStatistic import AnnotationsStatistic

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

    # creating Tasks object 
    processing_tasks = Tasks()

    # setting dictionary initial parameters for processing
    full_path_project = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-model-ssd'

    # getting application parameters 
    processing_tasks.start_task('Getting application parameters')
    parameters_filename = 'wm_model_ssd_parameters.json'
    parameters = get_parameters(full_path_project, parameters_filename)
    processing_tasks.finish_task('Getting application parameters')

    # setting new values of parameters according of initial parameters
    processing_tasks.start_task('Setting input image folders')
    set_input_image_folders(parameters)
    processing_tasks.finish_task('Setting input image folders')

    # getting last running id
    processing_tasks.start_task('Getting running id')
    running_id = get_running_id(parameters)
    processing_tasks.finish_task('Getting running id')

    # setting output folder results
    processing_tasks.start_task('Setting result folders')
    set_result_folders(parameters)
    processing_tasks.finish_task('Setting result folders')
    
    # creating log file 
    processing_tasks.start_task('Creating log file')
    logging_create_log(
        parameters['test_results']['log_folder'], 
        parameters['test_results']['log_filename']
    )
    processing_tasks.finish_task('Creating log file')

    logging_info('White Mold Research')
    logging_info('Inference of the model SSD (Single Shot Detector)' + LINE_FEED)

    logging_info(f'')
    logging_info(f'>> Set input image folders')
    logging_info(f'')
    logging_info(f'>> Get running id')
    logging_info(f"running id: {str(running_id)}")   
    logging_info(f'')
    logging_info(f'>> Set result folders')

    # getting device CUDA
    processing_tasks.start_task('Getting device CUDA')
    device = get_device(parameters)
    processing_tasks.finish_task('Getting device CUDA')

    # creating new instance of parameters file related to current running
    processing_tasks.start_task('Saving processing parameters')
    save_processing_parameters(parameters_filename, parameters)
    processing_tasks.finish_task('Saving processing parameters')
   
    # copying weights file produced by training step 
    processing_tasks.start_task('Copying weights file used in inference')
    copy_weights_file(parameters)
    processing_tasks.finish_task('Copying weights file used in inference')

    # loading dataloaders of image dataset for processing
    processing_tasks.start_task('Loading dataloaders of image dataset')
    test_dataloader = get_dataloaders(parameters)
    processing_tasks.finish_task('Loading dataloaders of image dataset')
    
    # creating neural network model 
    processing_tasks.start_task('Creating neural network model')
    model = get_neural_network_model(parameters, device)
    processing_tasks.finish_task('Creating neural network model')

    # getting statistics of input dataset
    if parameters['processing']['show_statistics_of_input_dataset']:
        processing_tasks.start_task('Getting statistics of input dataset')
        annotation_statistics = get_input_dataset_statistics(parameters)
        show_input_dataset_statistics(parameters, annotation_statistics)
        processing_tasks.finish_task('Getting statistics of input dataset')  

    # inference the neural netowrk model
    processing_tasks.start_task('Running inference of test images dataset')
    inference_neural_network_model(parameters, device, model)
    processing_tasks.finish_task('Running inference of test images dataset')

    # showing input dataset statistics
    # show_input_dataset_statistics(parameters, annotation_statistics)

    # finishing model training 
    logging_info('')
    logging_info('Finished the test of the model SSD (Single Shot Detector)')
    logging_info('')

    # printing tasks summary 
    processing_tasks.finish_processing()
    logging_info(processing_tasks.to_string())

    # copying processing files to log folder 
    # copy_processing_files_to_log(parameters)

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

    # returning running id 
    return running_id

def set_result_folders(parameters):
    '''
    Set folder name of output results
    '''

    # creating results folders 
    main_folder = os.path.join(
        parameters['processing']['research_root_folder'],     
        parameters['test_results']['main_folder']
    )
    parameters['test_results']['main_folder'] = main_folder
    Utils.create_directory(main_folder)

    # setting and creating model folder 
    parameters['test_results']['model_folder'] = parameters['neural_network_model']['model_name']
    model_folder = os.path.join(
        main_folder,
        parameters['test_results']['model_folder']
    )
    parameters['test_results']['model_folder'] = model_folder
    Utils.create_directory(model_folder)

    # setting and creating action folder of training
    action_folder = os.path.join(
        model_folder,
        parameters['test_results']['action_folder']
    )
    parameters['test_results']['action_folder'] = action_folder
    Utils.create_directory(action_folder)

    # setting and creating running folder 
    running_id = parameters['processing']['running_id']
    running_id_text = 'running-' + f'{running_id:04}'
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    parameters['test_results']['running_folder'] = running_id_text + "-" + input_image_size + 'x' + input_image_size   
    running_folder = os.path.join(
        action_folder,
        parameters['test_results']['running_folder']
    )
    parameters['test_results']['running_folder'] = running_folder
    Utils.create_directory(running_folder)

    # setting and creating others specific folders
    processing_parameters_folder = os.path.join(
        running_folder,
        parameters['test_results']['processing_parameters_folder']
    )
    parameters['test_results']['processing_parameters_folder'] = processing_parameters_folder
    Utils.create_directory(processing_parameters_folder)

    weights_folder = os.path.join(
        running_folder,
        parameters['test_results']['weights_folder']
    )
    parameters['test_results']['weights_folder'] = weights_folder
    Utils.create_directory(weights_folder)

    metrics_folder = os.path.join(
        running_folder,
        parameters['test_results']['metrics_folder']
    )
    parameters['test_results']['metrics_folder'] = metrics_folder
    Utils.create_directory(metrics_folder)

    inferenced_image_folder = os.path.join(
        running_folder,
        parameters['test_results']['inferenced_image_folder']
    )
    parameters['test_results']['inferenced_image_folder'] = inferenced_image_folder
    Utils.create_directory(inferenced_image_folder)

    log_folder = os.path.join(
        running_folder,
        parameters['test_results']['log_folder']
    )
    parameters['test_results']['log_folder'] = log_folder
    Utils.create_directory(log_folder)

def get_device(parameters):
    '''
    Get device CUDA to train models
    ''' 

    logging_info(f'')
    logging_info(f'>> Get device')

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    parameters['processing']['device'] = f'{device}'

    logging_info(f'Device: {device}')

    # returning current device 
    return device 

def save_processing_parameters(parameters_filename, parameters):
    '''
    Update parameters file of the processing
    '''

    logging_info(f'')
    logging_info(f'>> Save processing parameters of this running')

    # setting full path and log folder  to write parameters file 
    path_and_parameters_filename = os.path.join(
        parameters['test_results']['processing_parameters_folder'], 
        parameters_filename)

    # saving current processing parameters in the log folder 
    Utils.save_text_file(path_and_parameters_filename, \
                        Utils.get_pretty_json(parameters), 
                        NEW_FILE)

def copy_weights_file(parameters):
    '''
    Copying weights file to inference step
    '''

    logging_info(f'')
    logging_info(f'>> Copy weights file of the model for inference')
    logging_info(f"Folder name: {parameters['input']['inference']['weights_folder']}")
    logging_info(f"Filename   : {parameters['input']['inference']['weights_filename']}")

    Utils.copy_file_same_name(
        parameters['input']['inference']['weights_filename'],
        parameters['input']['inference']['weights_folder'],
        parameters['test_results']['weights_folder']
    )

def get_dataloaders(parameters):
    '''
    Get dataloaders of testing from image dataset 
    '''

    logging_info(f'')
    logging_info(f'>> Get dataset and dataloaders of the images for processing')

    # getting datasets 
    test_dataset = create_test_dataset(
        parameters['processing']['image_dataset_folder_test'], 
        parameters['neural_network_model']['resize_of_input_image'], 
        parameters['neural_network_model']['classes'], 
    )
   
    logging.info(f'Getting datasets')
    logging.info(f'   Number of testing images: {len(test_dataset)}')
    logging.info(f'   Total                   : {len(test_dataset)}')
    logging_info(f'')

    # getting dataloaders
    test_dataloader = create_test_loader(
        test_dataset, 
        parameters['neural_network_model']['number_workers']
    )

    # returning dataloaders for processing 
    return test_dataloader 

def get_neural_network_model(parameters, device):
    '''
    Get neural network model
    '''      

    logging_info(f'')
    logging_info(f'>> Get neural network model')

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
    
    model = model.to(device)

    logging.info(f'{model}')

    # returning neural network model
    return model


# getting statistics of input dataset 
def get_input_dataset_statistics(parameters):
    
    annotation_statistics = AnnotationsStatistic()
    steps = ['test'] 
    # steps = ['train', 'valid', 'test']
    annotation_statistics.processing_statistics(parameters, steps)
    return annotation_statistics
    
def show_input_dataset_statistics(parameters, annotation_statistics):

    logging_info(f'Input dataset statistic')
    logging_info(annotation_statistics.to_string())
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'],
        parameters['neural_network_model']['model_name'] + '_annotations_statistics.xlsx',
    )
    annotation_format = parameters['input']['input_dataset']['annotation_format']
    input_image_size = parameters['input']['input_dataset']['input_image_size']
    classes = (parameters['neural_network_model']['classes'])[1:5]
    annotation_statistics.save_annotations_statistics(
        path_and_filename,
        annotation_format,
        input_image_size,
        classes
    )

def copy_processing_files_to_log(parameters):
    input_path = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
    )
    output_path = parameters['test_results']['log_folder']    
    input_filename = output_filename = 'ssd_inference_errors'
    # logging_info(f'input_filename: {input_filename}')
    # logging_info(f'input_path: {input_path}')
    # logging_info(f'output_filename: {output_filename}')
    # logging_info(f'output_path: {output_path}')
    Utils.copy_file(input_filename, input_path, output_filename, output_path)

    input_filename = output_filename = 'ssd_inference_output'
    Utils.copy_file(input_filename, input_path, output_filename, output_path)


# ###########################################
# Methods of Level 3
# ###########################################


# ###########################################
# Main method
# ###########################################
if __name__ == '__main__':
    main()
