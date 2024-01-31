"""
Institution: Institute of Computing - University of Campinas (IC/Unicamp)
Project: White Mold 
Description: Implements the neural network model SSD (Single Shot Detector) for step of training.
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 25/01/2024
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
    # creating log file 
    log_filename = logging_create_log()

    logging_info('White Mold Research')
    logging_info('Model SSD (Single Shot Detector) Training' + LINE_FEED)
    print('White Mold Research')
    print('Model SSD (Single Shot Detector) Training' + LINE_FEED)

    # getting last running id
    running_id = get_running_id()

    # setting dictionary initial parameters for processing
    parameters_filename = 'wm_model_ssd_parameters.json'
    logging_info('1) Setting processing parameters' + LINE_FEED)
    print('1) Setting processing parameters' + LINE_FEED)
    processing_parameters = get_processing_parameters(parameters_filename, log_filename, running_id)

    # setting output folder results
    set_folder_of_output_results(processing_parameters)

    # loading dataset 


# ###########################################
# Methods of Level 2
# ###########################################

def get_running_id():
    '''
    Get last running id to calculate the current id
    '''    

    # setting control filename 
    running_control_filename = 'running_control.json'

    # getting control info 
    running_control = Utils.read_json_parameters(running_control_filename)

    # calculating the current running id 
    running_control['last_running_id'] = int(running_control['last_running_id']) + 1

    # updating running control file 
    running_id = int(running_control['last_running_id'])

    # saving file 
    # path_and_parameters_filename = os.path.join('log', log_filename + "-" + parameters_filename)
    Utils.save_text_file(running_control_filename, \
                         Utils.get_pretty_json(running_control), 
                         NEW_FILE) 

    # returning the current running id
    return running_id

def get_processing_parameters(parameters_filename, log_filename, running_id):
    '''
    Get dictionary parameters for processing
    '''    
    # getting parameters 
    processing_parameters = Utils.read_json_parameters(parameters_filename)

    # setting new values of parameters according of initial parameters
    set_folder_of_input_images_dataset(processing_parameters)

    # updating runnig id in processing parameters 
    processing_parameters['processing']['running_id'] = running_id

    # logging processing parameters 
    logging_info(Utils.get_pretty_json(processing_parameters) + LINE_FEED)   
    
    # saving current processing parameters in the log folder 
    path_and_parameters_filename = os.path.join('log', log_filename + "-" + parameters_filename)
    Utils.save_text_file(path_and_parameters_filename, \
                        Utils.get_pretty_json(processing_parameters), 
                        NEW_FILE)

    # print(Utils.get_pretty_json(processing_parameters))

    # returning parameters 
    return processing_parameters

def set_folder_of_output_results(processing_parameters):
    '''
    Set folder name of output results
    '''

    # getting image dataset folder according processing parameters 
    input_image_size = str(processing_parameters['input_dataset']['input_image_size'])
    running_id = processing_parameters['processing']['running_id']
    running_id_text = 'running_' + f'{running_id:03}'
    output_results_folder = os.path.join(
        processing_parameters['processing']['research_root_folder'],
        processing_parameters['results']['main_folder'],
        processing_parameters['neural_network_model']['model_name'],
        running_id_text,
        input_image_size + 'x' + input_image_size,
    )

    # creating output results folder 
    Utils.create_directory(output_results_folder)

# ###########################################
# Methods of Level 3
# ###########################################

def set_folder_of_input_images_dataset(processing_parameters):
    '''
    Set folder name of input images dataset
    '''    
    
    # getting image dataset folder according processing parameters 
    input_image_size = str(processing_parameters['input_dataset']['input_image_size'])
    image_dataset_folder = os.path.join(
        processing_parameters['processing']['research_root_folder'],
        'white-mold-dataset',
        '03-splitting_by_images',
        '03.2-output-dataset',
        processing_parameters['input_dataset']['annotation_format'],
        input_image_size + 'x' + input_image_size,
    )

    # setting image dataset folder in processing parameters 
    processing_parameters['processing']['image_dataset_folder'] = image_dataset_folder
    processing_parameters['processing']['image_dataset_folder_train'] = \
        os.path.join(image_dataset_folder, 'train')
    processing_parameters['processing']['image_dataset_folder_valid'] = \
        os.path.join(image_dataset_folder, 'valid')
    processing_parameters['processing']['image_dataset_folder_test'] = \
        os.path.join(image_dataset_folder, 'test')


# ###########################################
# Main method
# ###########################################
if __name__ == '__main__':
    main()
