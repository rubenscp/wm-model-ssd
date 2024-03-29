"""
Project: White Mold 
Description: This module implements methods and functions related to metrics used in the models
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 16/02/2024
Version: 1.0
This implementation is based on the web article "Intersection over Union (IoU) in Object Detection & Segmentation", 
from LearnOpenCV, and it can be accessed by:
- https://colab.research.google.com/drive/1wxIVwYQ6RPXRiYhGqhYoixa53CQPLKSz?authuser=1#scrollTo=a183afbb
- https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/
"""

# Importing Python libraries 
import numpy as np
import torch
import matplotlib.pyplot as plt

# from torchvision import ops
# from torchmetrics.detection import IntersectionOverUnion 
# from torchmetrics.detection import GeneralizedIntersectionOverUnion 
# from torchmetrics.detection import MeanAveragePrecision
# from torchmetrics.classification import MulticlassConfusionMatrix

from sklearn import metrics
# import seaborn as sns

# from torchvision.models.detection import box_iou
from torchvision.ops import * 

# Importing python modules
from common.manage_log import *

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

class Metrics:

    def __init__(self, model=None):
        self.model = model
        self.preds = []
        self.target = []
        self.result = None
        self.full_confusion_matrix = None
        self.confusion_matrix = None
        self.confusion_matrix_normalized = None
        self.confusion_matrix_summary = {
            'number_of_images': 0,
            'number_of_bounding_boxes_target': 0,
            'number_of_bounding_boxes_predicted': 0,
            'number_of_bounding_boxes_predicted_with_target': 0,
            'number_of_ghost_predictions': 0,
            'number_of_undetected_objects': 0,
        }
        self.counts_per_class = []
        # self.counts_of_model = []
        self.tp_model = 0
        self.fn_model = 0
        self.fp_model = 0
        self.tn_model = 0

    def to_string(self):
        text = 'Metrics' + LINE_FEED + LINE_FEED
        text += f'preds: {len(self.preds)}' + LINE_FEED
        text += str(self.preds) + LINE_FEED + LINE_FEED
        text += f'target: {len(self.target)}' + LINE_FEED
        text += str(self.target) + LINE_FEED
        return text

    def set_target(self, target):
        self.target = target 
        
    def set_preds(self, preds):
        self.preds = preds

    def get_target_size(self):
        # counting number of bounding boxes
        count = 0
        for target in self.target:
            count += len(target['labels'])

        # returning number of bounding boxes target
        return count
        

    def get_preds_size(self):
        # counting number of bounding boxes
        count = 0
        for pred in self.preds:
            count += len(pred['labels'])

        # returning number of bounding boxes predicted
        return count

    def get_predicted_bounding_boxes(self, boxes, scores, labels):
            
        # creating predicted object 
        predicted = []

        # getting bounding boxes in format fot predicted object 
        predicted_boxes = []
        predicted_scores = []
        predicted_labels = []
        for i, box in enumerate(boxes):        
            predicted_boxes.append(box)
            predicted_scores.append(scores[i])
            predicted_labels.append(labels[i])

        # setting predicted dictionary 
        item = {
            "boxes": torch.tensor(predicted_boxes, dtype=torch.float),
            "scores": torch.tensor(predicted_scores, dtype=torch.float),
            "labels": torch.tensor(predicted_labels)
            }
        predicted.append(item)

        # returning predicted object
        return predicted 

    # def calculate_box_iou(self, class_metrics_indicator):

    #     # calculating IoU of the bounding boxes
    #     result = 0.0
    #     if len(self.preds[0]['boxes']) > 0:
    #         if class_metrics_indicator:
    #             metric = IntersectionOverUnion(class_metrics=True)
    #         else:
    #             metric = IntersectionOverUnion()

    #         result = metric(self.preds, self.target)

    #     logging_info(f'box_iou             : {result}')

    #     # returning results 
    #     return result
      
    # def calculate_generalized_box_iou(self):

    #     # calculating generalized IoU of the bounding boxes
    #     result = 0.0
    #     if len(self.preds[0]['boxes']) > 0:
    #         metric = GeneralizedIntersectionOverUnion()
    #         result = metric(self.preds, self.target)

    #     logging_info(f'generalized_box_iou : {result}')

    #     # returning results 
    #     return result
      
    # def calculate_mean_average_precision(self):

    #     # calculating mean average precision of the bounding boxes
    #     result = 0.0
    #     if len(self.preds[0]['boxes']) > 0:
    #         metric = MeanAveragePrecision()
    #         result = metric(self.preds, self.target)

    #     logging_info(f'mean_average_precision : {result}')

    #     # returning results 
    #     return result    

    def compute_confusion_matrix(self, num_classes, iou_threshold):

        # Inspired from:
        # https://medium.com/@tenyks_blogger/multiclass-confusion-matrix-for-object-detection-6fc4b0135de6
        
        # Step 4 and 5: Convert bounding box coordinates and apply thresholding for multi-label classification
        # (Assuming the output format of your model is similar to the torchvision Faster R-CNN model)

        # threshold = 0.5
        self.full_confusion_matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.confusion_matrix_normalized = np.zeros((num_classes + 1, num_classes + 1))
        undetected_objects_index = ghost_predictions_index = num_classes

        # logging_info(f'\n\n')
        # logging_info(f'compute_confusion_matrix ---------------------- ')
        # logging_info(f'self.preds: {self.preds}')
        # logging_info(f'len(self.preds) : {len(self.preds)}')
        # logging_info(f'len(self.target): {len(self.target)}')
        # logging_info(f'')
        
        number_of_bounding_boxes_predicted_with_target = 0
        number_of_ghost_predictions = 0
        number_of_undetected_objects = 0     

        for pred, target in zip(self.preds, self.target):
            if len(pred['boxes']) == 0:
                # not_processed += 1
                number_of_undetected_objects += 1
                # logging_info(f'Undetected objects {not_processed}')
                # logging_info(f'pred: {pred}')
                # logging_info(f'target: {target}')
                # logging_info(f'')
                #  False Negative - counting undetected objects
                for t_label in target['labels']:
                    self.full_confusion_matrix[undetected_objects_index, t_label] += 1

            else:
                # processed += 1
                for p_box, p_label in zip(pred['boxes'], pred['labels']):
                    # number_of_bounding_boxes_predicted += 1

                    for t_box, t_label in zip(target['boxes'], target['labels']):

                        # compute IoU of two boxes 
                        iou = box_iou(p_box.unsqueeze(0), t_box.unsqueeze(0))

                        # evaluate IoU threshold and labels
                        if iou >= iou_threshold:
                            number_of_bounding_boxes_predicted_with_target += 1
                            if p_label == t_label:
                                # True Positive 
                                self.full_confusion_matrix[t_label, p_label] += 1
                            else:                        
                                # False Positive 
                                self.full_confusion_matrix[t_label, p_label] += 1
                        else:
                            # Counting ghost predictions   
                            number_of_ghost_predictions += 1                         
                            self.full_confusion_matrix[t_label, ghost_predictions_index] += 1
                            # logging_info(f'iou_less_than_iou_threshold: {iou_less_than_iou_threshold}')
                            # logging_info(f'bbox iou: {iou} < iou_threshold: {iou_threshold}    ' + \
                            #         f't_label: {t_label}    p_label: {p_label}')


        # getting just confusion matrix whithout the background, ghost predictions and undetected objects
        self.confusion_matrix = np.copy(self.full_confusion_matrix[1:-1,1:-1])
        logging_info(f'self.full_confusion_matrix: {LINE_FEED}{self.full_confusion_matrix}')
        logging_info(f'self.confusion_matrix: {LINE_FEED}{self.confusion_matrix}')

        # # normalizing values summarizing by rows
        # sum_columns = np.sum(self.confusion_matrix,axis=1)
        # row, col = self.confusion_matrix.shape
        # for i in range(row):
        #     if sum_columns[i] > 0:
        #         self.confusion_matrix_normalized[i] = self.confusion_matrix_normalized[i] / sum_columns[i]

        # logging_info(f'self.confusion_matrix_normalized: {LINE_FEED}{self.confusion_matrix_normalized}')

        # summary of confusion matrix        
        self.confusion_matrix_summary["number_of_images"] = len(self.preds)
        self.confusion_matrix_summary["number_of_bounding_boxes_target"] = self.get_target_size()
        self.confusion_matrix_summary["number_of_bounding_boxes_predicted"] = self.get_preds_size()
        self.confusion_matrix_summary["number_of_bounding_boxes_predicted_with_target"] = number_of_bounding_boxes_predicted_with_target
        self.confusion_matrix_summary["number_of_ghost_predictions"] = number_of_ghost_predictions
        self.confusion_matrix_summary["number_of_undetected_objects"] = number_of_undetected_objects

        # computing metrics from confuson matrix 
        self.compute_metrics_from_confusion_matrix()


    def confusion_matrix_to_string(self):
        logging_info(f'')
        logging_info(f'FULL CONFUSION MATRIX')
        logging_info(f'---------------------')
        logging_info(f'{LINE_FEED}{self.full_confusion_matrix}')
        # logging_info(f'Summarize confusion matrix: {torch.sum(self.confusion_matrix)}')
        logging_info(f'')
        logging_info(f'CONFUSION MATRIX')
        logging_info(f'---------------------------')
        logging_info(f'{LINE_FEED}{self.confusion_matrix}')
        logging_info(f'')
        logging_info(f'SUMMARY OF CONFUSION MATRIX')
        logging_info(f'---------------------------')
        logging_info(f'')
        logging_info(f'Total number of images               : ' + \
                     f'{self.confusion_matrix_summary["number_of_images"]}')
        logging_info(f'Bounding boxes target                : ' + \
            f'{self.confusion_matrix_summary["number_of_bounding_boxes_target"]}')
        logging_info(f'Bounding boxes predicted             : ' + \
            f'{self.confusion_matrix_summary["number_of_bounding_boxes_predicted"]}')
        logging_info(f'Bounding boxes predicted with target : ' + \
                     f'{self.confusion_matrix_summary["number_of_bounding_boxes_predicted_with_target"]}')
        logging_info(f'Number of ghost preditions           : ' + \
            f'{self.confusion_matrix_summary["number_of_ghost_predictions"]}')
        logging_info(f'Number of undetected objects         : ' + \
            f'{self.confusion_matrix_summary["number_of_undetected_objects"]}')
        logging_info(f'')
        
    # Extract from: 
    # 1) https://stackoverflow.com/questions/43697980/is-there-something-already-implemented-in-python-to-calculate-tp-tn-fp-and-fn
    # 2) https://stackoverflow.com/questions/75478099/how-to-extract-performance-metrics-from-confusion-matrix-for-multiclass-classifi?newreg=c9549e71afff4f13982ca151adedfbd5
    # 3) https://www.youtube.com/watch?v=FAr2GmWNbT0

    def compute_metrics_from_confusion_matrix(self):
        """
        Obtain TP, FN FP, and TN for each class in the confusion matrix
        """

        # getting a copy of confusion matrix 
        confusion = np.copy(self.confusion_matrix)
        logging_info(f'confusion: {LINE_FEED}{confusion}')

        # removing from confusion matrix the lines and columns of background, 
        # ghost predictions and undetected objects
        # confusion = confusion[1:-1,1:-1]
        # logging_info(f'self.confusion_matrix_normalized: {self.confusion_matrix_normalized}')
        # logging_info(f'confusion_matrix reduced: {confusion}')

        self.counts_per_class = []
                 
        # Iterate through classes and store the counts
        for i in range(confusion.shape[0]):
            tp = confusion[i, i]

            fn_mask = np.zeros(confusion.shape)
            fn_mask[i, :] = 1
            fn_mask[i, i] = 0
            fn = np.sum(np.multiply(confusion, fn_mask))

            fp_mask = np.zeros(confusion.shape)
            fp_mask[:, i] = 1
            fp_mask[i, i] = 0
            fp = np.sum(np.multiply(confusion, fp_mask))

            tn_mask = 1 - (fn_mask + fp_mask)
            tn_mask[i, i] = 0
            tn = np.sum(np.multiply(confusion, tn_mask))

            self.counts_per_class.append( {'Class': i,
                                            'TP': tp,
                                            'FN': fn,
                                            'FP': fp,
                                            'TN': tn} )

            # logging_info(f'counts_per_class: {self.counts_per_class}')
            # logging_info({'Class': i,
            #               'TP': tp,
            #               'FN': fn,
            #               'FP': fp,
            #               'TN': tn})



        # counting for model 
        self.tp_model = 0
        self.fn_model = 0
        self.fp_model = 0
        self.tn_model = 0
        for count in self.counts_per_class:
            self.tp_model += count['TP']
            self.fn_model += count['FN']
            self.fp_model += count['FP']
            self.tn_model += count['TN']

        # self.counts_of_model = {'Model': self.model,
        #                      'TP': tp_model,
        #                      'FN': fn_model,
        #                      'FP': fp_model,
        #                      'TN': tn_model}

        logging_info(f'TP / FN / FP / TN from confunsion matrix: ')
        for count in self.counts_per_class:
            logging_info(f'count {count}')
        
        logging_info(f'self.tp_model:{self.tp_model}')
        logging_info(f'self.fn_model:{self.fn_model}')
        logging_info(f'self.fp_model:{self.fp_model}')
        logging_info(f'self.tn_model:{self.tn_model}')

        # logging_info(f'counts_model: {self.counts_model}')
        # logging_info(f'counts_list: {counts_list}')             

    def get_value_metric(self, metric):
        value = 0
        for count in self.counts_per_class:
            value += count[metric]
        return value

    # https://docs.kolena.io/metrics/accuracy/
    def get_model_accuracy(self):
        accuracy = (self.tp_model + self.tn_model) /  \
                   (self.tp_model + self.tn_model + self.fp_model + self.fn_model)
        return accuracy

    # https://docs.kolena.io/metrics/precision/
    def get_model_precision(self):
        precision = (self.tp_model) /  \
                    (self.tp_model + self.fp_model)
        return precision

    # https://docs.kolena.io/metrics/recall/
    def get_model_recall(self):
        recall = (self.tp_model) /  \
                 (self.tp_model + self.fn_model)
        return recall

    # https://docs.kolena.io/metrics/f1-score/
    def get_model_f1_score(self):
        f1_score = (2.0 * self.get_model_precision() * self.get_model_recall()) /  \
                   (self.get_model_precision() + self.get_model_recall())
        return f1_score

    # https://docs.kolena.io/metrics/specificity/
    def get_model_specificity(self):
        specificity = (self.tn_model) /  \
                      (self.tn_model + self.fp_model)
        return specificity

    # https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    def get_model_dice(self):
        dice = (2 * self.tp_model) /  \
               (2 * self.tp_model + self.fp_model + self.fn_model)
        return dice
        