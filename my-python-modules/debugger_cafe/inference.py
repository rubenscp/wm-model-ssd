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
from common.manage_log import *
from common.utils import * 
from common.entity.ImageAnnotation import ImageAnnotation
from common.metrics import *

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

    # creating metric object 
    inference_metric = Metrics(model=parameters['neural_network_model']['model_name'])

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
        logging_info(f'')
        logging_info(f'Processing image #{i+1} - {test_images[i]}')
        
        # extracting parts of path and image filename 
        path, filename_with_extension, filename, extension = Utils.get_filename(test_images[i])

        # setting the annotation filename 
        path_and_filename_xml_annotation = os.path.join(path, filename + '.xml')

        # getting xml annotation of the image 
        image_annotation = ImageAnnotation()
        image_annotation.get_annotation_file_in_voc_pascal_format(path_and_filename_xml_annotation)
        # print(f'inference - image_annotation: {image_annotation.to_string()} ')
        
        # getting target bounding boxes 
        target = image_annotation.get_tensor_target(classes)
        # print(f'target 2 : {target}')

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
            labels = outputs[0]['labels'].data.numpy()

            # print(f'outputs predicted: {outputs}')
            # print(f'boxes  : {len(boxes)}')
            # print(f'scores : {len(scores)}')
            # print(f'labels : {len(labels)}')

            # Filter out boxes according to `detection_threshold`.
            # boxes = boxes[scores >= args['threshold']].astype(np.int32)
            threshold = parameters['neural_network_model']['threshold']
            boxes = boxes[scores >= threshold].astype(np.int32)
            draw_boxes = boxes.copy()

            boxes_thresholded = []
            scores_thresholded = []
            labels_thresholded = []
            for i in range(len(boxes)):
                if scores[i] >= threshold:
                    # logging_info(f'indice box: {i}')
                    # logging_info(f'boxes[{i}]: {boxes[i]}')
                    # logging_info(f'scores[{i}]: {scores[i]}')
                    boxes_thresholded.append(boxes[i].astype(np.int32))
                    scores_thresholded.append(scores[i])
                    labels_thresholded.append(labels[i])

            # logging_info(f'draw_boxes: {draw_boxes}')
            draw_boxes2 = boxes_thresholded.copy()
            # logging_info(f'draw_boxes2: {draw_boxes2}')
            # logging_info(f'-----------------------------')                        
            # logging_info(f'boxes_thresholded: {boxes_thresholded}')
            # logging_info(f'scores_thresholded: {scores_thresholded}')

            # Get all the predicited class names.
            pred_classes = [classes[label] for label in outputs[0]['labels'].cpu().numpy()]
            # print(f'pred_classes : {len(pred_classes)}')
            # print(f'pred_classes : {pred_classes}')
            # print(f'')
            # print(f'outputs thresholded')
            # print(f'boxes_thresholded : {boxes_thresholded}')
            # print(f'scores_thresholded : {scores_thresholded}')
            # print(f'labels_thresholded : {labels_thresholded}')
            
            # getting predicted bounding boxes 
            predicted = inference_metric.get_predicted_bounding_boxes(
                                        boxes_thresholded,
                                        scores_thresholded,
                                        labels_thresholded)

            # print(f'--------------------------------------------------')
            # print(f'target   : {target}')
            # print(f'predicted: {predicted}')
            # print(f'--------------------------------------------------')

            # setting target and predicted bounding boxes for metrics 
            inference_metric.target.extend(target)
            inference_metric.preds.extend(predicted)
            # print(f'inference_metric.to_string: {inference_metric.to_string()}')

            # calculating IoU metric 
            # result_iou  = inference_metric.calculate_box_iou(class_metrics_indicator=False)
            # result_giou = inference_metric.calculate_generalized_box_iou()
            # result_map  = inference_metric.calculate_mean_average_precision()

            # num_labels = 4
            # resul_confusion_matrix = inference_metric.compute_confusion_matrix_111(num_labels)

            
            # result2 = inference_metric.calculate_box_iou(class_metrics_indicator=True)
            # print(f'inference result: {result}')

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

    # Computing Confusion Matrix 
    num_classes = 5
    iou_threshold = parameters['neural_network_model']['iou_threshold']
    inference_metric.compute_confusion_matrix(num_classes, iou_threshold)
    inference_metric.confusion_matrix_to_string()

    # saving confusion matrix plots 
    title = 'Full Confusion Matrix'
    path_and_filename = os.path.join(parameters['inference_results']['metrics_folder'], 
        'confusion_matrix_full_' + parameters['neural_network_model']['model_name'] + '.png'
    )
    cm_classes = classes[0:5]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    x_labels_names.append('Ghost predictions')    
    y_labels_names.append('Undetected objects')
    format='.0f'
    Utils.save_plot_confusion_matrix(inference_metric.full_confusion_matrix, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)
        
    title = 'Confusion Matrix'
    path_and_filename = os.path.join(parameters['inference_results']['metrics_folder'], 
        'confusion_matrix_' + parameters['neural_network_model']['model_name'] + '.png'
    )
    cm_classes = classes[1:5]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    format='.0f'
    Utils.save_plot_confusion_matrix(inference_metric.confusion_matrix, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)

    # get performance metrics 
    logging_info(f'')
    logging_info(f'Performance Metrics')
    logging_info(f'-------------------')
    model_accuracy = inference_metric.get_model_accuracy()
    logging_info(f'accuracy    : {model_accuracy:.4f}')
    model_precision = inference_metric.get_model_precision()
    logging_info(f'precision   : {model_precision:.4f}')
    model_recall = inference_metric.get_model_recall()
    logging_info(f'recall      : {model_recall:.4f}')
    model_f1_score = inference_metric.get_model_f1_score()
    logging_info(f'f1-score    : {model_f1_score:.4f}')
    model_specificity = inference_metric.get_model_specificity()
    logging_info(f'specificity : {model_specificity:.4f}')
    model_dice = inference_metric.get_model_dice()
    logging_info(f'dice        : {model_dice:.4f}')
    logging_info(f'')

    # path_and_filename = os.path.join(parameters['inference_results']['metrics_folder'], 
    #                         'confusion_matrix_ssd_normalized.png')
    # cm_classes = classes[1:5]
    # title = 'Confusion Matrix Normalized'
    # Utils.save_plot_confusion_matrix(inference_metric.confusion_matrix_normalized[1:,1:], 
    #                                  path_and_filename, cm_classes, title, fmt='.2f')

    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    # logging_info(f"Average FPS: {avg_fps:.3f}")


# def get_predicted_bounding_boxes(boxes, labels):
           
#     # creating predicted object 
#     predicted = []

#     # getting bounding boxes in format fot predicted object 
#     predicted_boxes = []
#     predicted_labels = []
#     for i, box in enumerate(boxes):        
#         predicted_boxes.append(box)
#         predicted_labels.append(labels[i])

#     # setting predicted dictionary 
#     item = {
#         "boxes": torch.tensor(predicted_boxes, dtype=torch.float),
#         "labels": torch.tensor(predicted_labels)
#         }
#     predicted.append(item)

#     # returning predicted object
#     return predicted 


# def inference_neural_network_model_new_version(parameters, device, model, test_dataloader):
#     '''
#     Inference test images dataset in the trained mode
#     '''    

#     # setting work objects 
#     predictions = []
#     targets = []

#     # loading weights
#     path_and_weights_filename = os.path.join(
#         parameters['input']['inference']['weights_folder'],
#         parameters['input']['inference']['weights_filename'],
#     )
#     checkpoint = torch.load(path_and_weights_filename, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device).eval()

#     # running evaluation of trained model with the test image dataset
#     with torch.no_grad():
#         for inputs, target in test_dataloader:  # assuming you have a DataLoader for your dataset

#             print(f'inputs: {inputs}')
#             print(f'target: {target}')
#             exit()

#             outputs = model(inputs)
#             predictions.append(outputs)
#             targets.append(target)

#             print(f'outputs: {outputs}')
#             print(f'targets: {targets}')
            
#             exit()

    
