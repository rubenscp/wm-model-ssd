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
import torch
from torchvision import ops

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'


# iou = ops.box_iou(ground_truth_bbox, prediction_bbox)
# print(iou)


# def get_iou(ground_truth, pred):
#     # coordinates of the area of intersection.
#     ix1 = np.maximum(ground_truth[0], pred[0])
#     iy1 = np.maximum(ground_truth[1], pred[1])
#     ix2 = np.minimum(ground_truth[2], pred[2])
#     iy2 = np.minimum(ground_truth[3], pred[3])
     
#     # Intersection height and width.
#     i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
#     i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
#     area_of_intersection = i_height * i_width
     
#     # Ground Truth dimensions.
#     gt_height = ground_truth[3] - ground_truth[1] + 1
#     gt_width = ground_truth[2] - ground_truth[0] + 1
     
#     # Prediction dimensions.
#     pd_height = pred[3] - pred[1] + 1
#     pd_width = pred[2] - pred[0] + 1
     
#     area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
#     iou = area_of_intersection / area_of_union
     
#     return iou