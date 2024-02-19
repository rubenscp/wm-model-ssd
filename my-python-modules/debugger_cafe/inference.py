import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse

# from model import create_model

# from config import (
#     NUM_CLASSES, DEVICE, CLASSES
# )

# Importing python modules
from manage_log import *
from utils import * 

# Importing entity classes
from entity.ImageAnnotation import ImageAnnotation

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

# setting seed 
np.random.seed(42)

# ###############################################
# The method below was added by Rubens
# ###############################################
def inference_neural_network_model(parameters, device, model):
    '''
    Inference test images dataset in the trained mode
    '''    

    logging_info(f'')
    logging_info(f'>> Processing the inference on the test images dataset')
    logging_info(f'')

    # creating working lists
    y_pred = []
    y_true = []

    # loading weights
    path_and_weights_filename = os.path.join(
        parameters['input']['inference']['weights_folder'],
        parameters['input']['inference']['weights_filename'],
    )
    checkpoint = torch.load(path_and_weights_filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    # getting list of test images for inference 
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    test_image_dataset_folder = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['input']['input_dataset']['input_dataset_path'],
        parameters['input']['input_dataset']['annotation_format'],
        input_image_size + 'x' + input_image_size,
        'test'
    )

    # classes 
    classes = parameters['neural_network_model']['classes']

    # COLORS 
    colors = [[0, 0, 0],        [255, 0, 0],        [0, 255, 0],    [0, 0, 255], 
              [238, 130, 238],  [106, 90, 205],     [188, 0, 239]]

    # DIR_TEST = args['input']
    test_images = glob.glob(f"{test_image_dataset_folder}/*.jpg")
    # print(f"Tests list: {test_images}")

    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.

    for i in range(len(test_images)):
        # logging image name 
        logging_info(f'Processing image #{i+1} - {test_images[i]}')
        
        # extracting parts of path and image filename 
        path, filename_with_extension, filename, extension = Utils.get_filename(test_images[i])

        # setting the annotation filename 
        path_and_filename_xml_annotation = os.path.join(path, filename + '.xml')

        # getting xml annotation of the image 
        image_annotation = ImageAnnotation()
        image_annotation.get_annotation_in_voc_pascal_format(path_and_filename_xml_annotation)
        # print(image_annotation.toString())

        # Get the image file name for saving output later on.
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        # if args['imgsz'] is not None:
        #     image = cv2.resize(image, (args['imgsz'], args['imgsz']))
        # logging_info(f'image.shape: {image.shape}')

        # BGR to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Make the pixel range between 0 and 1.
        image /= 255.0
        
        # Bring color channels to front (H, W, C) => (C, H, W).
        image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        # Convert to tensor.
        image_input = torch.tensor(image_input, dtype=torch.float).cuda()
        
        # Add batch dimension.
        image_input = torch.unsqueeze(image_input, 0)
        start_time = time.time()
        
        # Predictions
        with torch.no_grad():
            outputs = model(image_input.to(device))
        end_time = time.time()

        # NOTE: The bounding boxes selected in the "outputs" are ordered dicreasing by its scores

        # Get the current fps.
        fps = 1 / (end_time - start_time)
        
        # Total FPS till current frame.
        total_fps += fps
        frame_count += 1

        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        # logging_info(f"outputs: {outputs}")
        # logging_info(f"len(outputs[0]['boxes']): {len(outputs[0]['boxes'])}")
        # logging_info(f"outputs[0]['boxes']: {outputs[0]['boxes']}")

        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()

            # Filter out boxes according to `detection_threshold`.
            # boxes = boxes[scores >= args['threshold']].astype(np.int32)
            threshold = parameters['neural_network_model']['threshold']
            boxes = boxes[scores >= threshold].astype(np.int32)
            draw_boxes = boxes.copy()

            boxes_thresholded = []
            scores_thresholded = []
            for i in range(len(boxes)):
                if scores[i] >= threshold:
                    # logging_info(f'indice box: {i}')
                    # logging_info(f'boxes[{i}]: {boxes[i]}')
                    # logging_info(f'scores[{i}]: {scores[i]}')
                    boxes_thresholded.append(boxes[i].astype(np.int32))
                    scores_thresholded.append(scores[i])

            # logging_info(f'draw_boxes: {draw_boxes}')
            draw_boxes2 = boxes_thresholded.copy()
            # logging_info(f'draw_boxes2: {draw_boxes2}')
            # logging_info(f'-----------------------------')                        
            # logging_info(f'boxes_thresholded: {boxes_thresholded}')
            # logging_info(f'scores_thresholded: {scores_thresholded}')

            # Get all the predicited class names.
            pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]

            # logging_info(f'bboxes selected: {len(draw_boxes)}')

            # Draw the bounding boxes and write the class name on top of it.
            for j, box in enumerate(draw_boxes):
                # logging_info(f'Inside for j,box - j:{j}  box:{box}')
                # logging_info(f'pred_classes:{pred_classes}')

                class_name = pred_classes[j]
                bbox_score = scores_thresholded[j]
                bbox_label_text = class_name + ' ('+ '%.2f' % bbox_score + ')'

                # logging_info(f'bbox_score: {bbox_score}')
                # logging_info(f'Inference 1 - class_name:{class_name}')
                # logging_info(f'Inference 2 - classes:{classes}')
                # logging_info(f'Inference 3 - classes.index(class_name):{classes.index(class_name)}')

                color = colors[classes.index(class_name)]
                # Recale boxes.
                xmin = int((box[0] / image.shape[1]) * orig_image.shape[1])
                ymin = int((box[1] / image.shape[0]) * orig_image.shape[0])
                xmax = int((box[2] / image.shape[1]) * orig_image.shape[1])
                ymax = int((box[3] / image.shape[0]) * orig_image.shape[0])
                cv2.rectangle(orig_image,
                            (xmin, ymin),
                            (xmax, ymax),
                            color[::-1],
                            3)
                cv2.putText(orig_image,
                            bbox_label_text,
                            (xmin, ymin-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color[::-1],
                            2,
                            lineType=cv2.LINE_AA)

            # cv2.imshow('Prediction', orig_image)
            cv2.waitKey(1)
            inferenced_image_folder = os.path.join(
                parameters['inference_results']['inferenced_image_folder']
            )
            cv2.imwrite(f"{inferenced_image_folder}/{image_name}_predicted.jpg", orig_image)

        # logging_info(f"Image {i+1} done...")
        # logging_info(f'-'*50)

    # logging_info('TEST PREDICTIONS COMPLETE')
    # cv2.destroyAllWindows()

    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    # logging_info(f"Average FPS: {avg_fps:.3f}")


# # Construct the argument parser.
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '-i', '--input', 
#     help='path to input image directory',
# )
# parser.add_argument(
#     '--imgsz', 
#     default=None,
#     type=int,
#     help='image resize shape'
# )
# parser.add_argument(
#     '--threshold',
#     default=0.25,
#     type=float,
#     help='detection threshold'
# )
# args = vars(parser.parse_args())

# os.makedirs('inference_outputs/images', exist_ok=True)

# COLORS = [[0, 0, 0], [255, 0, 0]]

# # Load the best model and trained weights.
# model = create_model(num_classes=NUM_CLASSES, size=640)
# checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.to(DEVICE).eval()

# # Directory where all the images are present.
# DIR_TEST = args['input']
# test_images = glob.glob(f"{DIR_TEST}/*.jpg")
# print(f"Test instances: {len(test_images)}")

# frame_count = 0 # To count total frames.
# total_fps = 0 # To get the final frames per second.

# for i in range(len(test_images)):
#     # Get the image file name for saving output later on.
#     image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
#     image = cv2.imread(test_images[i])
#     orig_image = image.copy()
#     if args['imgsz'] is not None:
#         image = cv2.resize(image, (args['imgsz'], args['imgsz']))
#     print(image.shape)
#     # BGR to RGB.
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
#     # Make the pixel range between 0 and 1.
#     image /= 255.0
#     # Bring color channels to front (H, W, C) => (C, H, W).
#     image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
#     # Convert to tensor.
#     image_input = torch.tensor(image_input, dtype=torch.float).cuda()
#     # Add batch dimension.
#     image_input = torch.unsqueeze(image_input, 0)
#     start_time = time.time()
#     # Predictions
#     with torch.no_grad():
#         outputs = model(image_input.to(DEVICE))
#     end_time = time.time()

#     # Get the current fps.
#     fps = 1 / (end_time - start_time)
#     # Total FPS till current frame.
#     total_fps += fps
#     frame_count += 1

#     # Load all detection to CPU for further operations.
#     outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
#     # Carry further only if there are detected boxes.
#     if len(outputs[0]['boxes']) != 0:
#         boxes = outputs[0]['boxes'].data.numpy()
#         scores = outputs[0]['scores'].data.numpy()
#         # Filter out boxes according to `detection_threshold`.
#         boxes = boxes[scores >= args['threshold']].astype(np.int32)
#         draw_boxes = boxes.copy()
#         # Get all the predicited class names.
#         pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
#         # Draw the bounding boxes and write the class name on top of it.
#         for j, box in enumerate(draw_boxes):
#             class_name = pred_classes[j]
#             color = COLORS[CLASSES.index(class_name)]
#             # Recale boxes.
#             xmin = int((box[0] / image.shape[1]) * orig_image.shape[1])
#             ymin = int((box[1] / image.shape[0]) * orig_image.shape[0])
#             xmax = int((box[2] / image.shape[1]) * orig_image.shape[1])
#             ymax = int((box[3] / image.shape[0]) * orig_image.shape[0])
#             cv2.rectangle(orig_image,
#                         (xmin, ymin),
#                         (xmax, ymax),
#                         color[::-1], 
#                         3)
#             cv2.putText(orig_image, 
#                         class_name, 
#                         (xmin, ymin-5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 
#                         0.8, 
#                         color[::-1], 
#                         2, 
#                         lineType=cv2.LINE_AA)

#         cv2.imshow('Prediction', orig_image)
#         cv2.waitKey(1)
#         cv2.imwrite(f"inference_outputs/images/{image_name}.jpg", orig_image)
#     print(f"Image {i+1} done...")
#     print('-'*50)

# print('TEST PREDICTIONS COMPLETE')
# cv2.destroyAllWindows()
# # Calculate and print the average FPS.
# avg_fps = total_fps / frame_count
# print(f"Average FPS: {avg_fps:.3f}")

