"""
Project: White Mold 
Description: Utils methods and functions 
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 20/10/2023
Version: 1.0
"""
# Importing Python libraries 
import os
import shutil
import json 
# import cv2

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

class Utils:

    # Create a folder
    @staticmethod
    def create_directory(folder):
        if not os.path.isdir(folder):    
            os.makedirs(folder)

    # Remove all files from a folder
    @staticmethod
    def remove_directory(folder):
        shutil.rmtree(folder, ignore_errors=True)

    # Read list of datasets in the input folder
    @staticmethod
    def get_folders(input_path):

        # getting list of image folders 
        folders = [f for f in os.listdir(input_path) 
                              if os.path.isdir(os.path.join(input_path, f))]        
        
        # returning list of image folders
        return folders

    # Read list of supervisely annotation files of one specific folder 
    def get_files_with_extensions(folder, extension):

        # getting list of supervisely annotation files
        files = [f for f in os.listdir(folder) if f.endswith(extension)]
        
        # returning list of image folders
        return files

    # Copy one file
    @staticmethod
    def copy_file(filename, input_path, output_path):
        # source = os.path.join(input_path, filename)
        # destination = os.path.join(output_path, filename)
        # shutil.copy(source, destination)
        copy_file(input_filename=filename, input_path=input_path, output_filename=filename, output_path=output_path)

    @staticmethod
    def copy_file(input_filename, input_path, output_filename, output_path):
        source = os.path.join(input_path, input_filename)
        destination = os.path.join(output_path, output_filename)
        shutil.copy(source, destination)

    # Remove one file
    @staticmethod
    def remove_file(path_and_filename):
        if os.path.isfile(path_and_filename): 
            os.remove(path_and_filename) 

    # # Read image 
    # @staticmethod
    # def read_image(filename, path):
    #     path_and_filename = os.path.join(path, filename)
    #     image = cv2.imread(path_and_filename)
    #     return image

    # # Save image
    # @staticmethod
    # def save_image(path_and_filename, image):
    #     cv2.imwrite(path_and_filename, image)

    # Save text file 
    @staticmethod
    def save_text_file(path_and_filename, content_of_text_file, create):
        # setting annotation file
        text_file = open(path_and_filename, 'w' if create else 'a+')

        # write line
        text_file.write(content_of_text_file)

        # closing text file
        text_file.close()        

    # # Draw bounding box in the image
    # @staticmethod
    # def draw_bounding_box(image, linP1, colP1, linP2, colP2, bgrBoxColor, thickness, label):
    #     # Start coordinate represents the top left corner of rectangle
    #     startPoint = (colP1, linP1)

    #     # Ending coordinate represents the bottom right corner of rectangle
    #     endPoint = (colP2, linP2)

    #     # Draw a rectangle with blue line borders of thickness of 2 px
    #     image = cv2.rectangle(image, startPoint, endPoint, bgrBoxColor, thickness)

    #     # setting the bounding box label
    #     cv2.putText(image, label,
    #                 (colP1, linP1 - 5),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgrBoxColor, 2)


    #     # returning the image with bounding box drawn
    #     return image

    # Create zip file from a directory 
    @staticmethod
    def zip_directory(source_directory, output_filename):
        shutil.make_archive(output_filename, 'zip', source_directory)

        # Check to see if the zip file is created
        full_output_filename = output_filename + '.zip'
        if os.path.exists(full_output_filename):
            return True, full_output_filename
        else:
            return False, None
        

    # Read JSON file with execution parameters
    @staticmethod
    def read_json_parameters(filename):

        # defining  parameters dictionary
        parameters = {}

        # reading parameters file 
        with open(filename) as json_file:
            parameters = json.load(json_file)

        # returning  parameters dictionary
        return parameters

    # Create a pretty json for printing
    @staticmethod
    def get_pretty_json(json_text):

        # formatting pretty json
        json_formatted_str = json.dumps(json_text, indent=4)
        
        # returning json formatted
        return json_formatted_str
                