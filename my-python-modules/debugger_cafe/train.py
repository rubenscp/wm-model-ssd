# from config import (
#     DEVICE, 
#     NUM_CLASSES, 
#     NUM_EPOCHS, 
#     OUT_DIR,
#     VISUALIZE_TRANSFORMED_IMAGES, 
#     NUM_WORKERS,
#     RESIZE_TO,
#     VALID_DIR,
#     TRAIN_DIR
# )
# from model import create_model
from debugger_cafe.custom_utils import (
    Averager, 
    SaveBestModel, 
    save_model, 
    save_loss_plot,
    save_mAP
)
from tqdm.auto import tqdm
# from datasets import (
#     create_train_dataset, 
#     create_valid_dataset, 
#     create_train_loader, 
#     create_valid_loader
# )
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import MultiStepLR

import torch
import matplotlib.pyplot as plt
import time
import os

# torch.multiprocessing.set_sharing_strategy('file_system')

# Importing python modules
from common.manage_log import *

LINE_FEED = '\n'

plt.style.use('ggplot')

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

# Function for running training iterations.
def train(train_data_loader, model, device, optimizer, train_loss_history):
    # logging_info(f'Training model')
    model.train()
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))   
    # for i, data in enumerate(prog_bar):
    # logging_info(f'train_data_loader size: {len(train_data_loader)}')

    for i, data in enumerate(train_data_loader):
        # logging_info(f'Training model loop - i:{i} ')
        # logging_info(f'Data: {data}')
        # logging_info(f'')
        # logging_info(f'Training model {i} - loop 1')
        optimizer.zero_grad()
        # logging_info(f'Training model {i} - loop 2')
        images, targets = data
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # logging_info(f'Training model {i} - loop 3')
        loss_dict = model(images, targets)
        # logging_info(f'Training model {i} - loop 4')

        losses = sum(loss for loss in loss_dict.values())
        # logging_info(f'Training model {i} - loop 5')
        loss_value = losses.item()
        # logging_info(f'Training model {i} - loop 6')

        train_loss_history.send(loss_value)
        # logging_info(f'Training model {i} - loop 7')

        losses.backward()
        # logging_info(f'Training model {i} - loop 8')
        optimizer.step()
        # logging_info(f'Training model {i} - loop 9')
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return loss_value

# Function for running validation iterations.
def validate(valid_data_loader, model, device):

    model.eval()
    
    # Initialize tqdm progress bar.
    # prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    # for i, data in enumerate(prog_bar):
    # logging_info(f'valid_data_loader size: {len(valid_data_loader)}')
    for i, data in enumerate(valid_data_loader):

        # logging_info(f'validate - 3 - i:{i}')

        images, targets = data
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images, targets)

        # logging_info(f'validate - 4 - antes calculo mAP - len(images): {len(images)}')

        # For mAP calculation using Torchmetrics.
        #####################################
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
        #####################################

    # logging_info(f'validate - 5 - antes MeanAveragePrecision')

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary

# ###############################################
# The method below was added by Rubens
# ###############################################
def train_neural_network_model(parameters, device, model, train_dataloader, valid_dataloader):
    '''
    Train model with the image dataset 
    '''    

    logging_info('3. Train model')

    # setting seeds
    plt.style.use('ggplot')        
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    logging_info(f"{total_params:,} total parameters")

    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logging_info(f'{total_trainable_params:,} training parameters.')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=parameters['neural_network_model']['learning_rate'],
        momentum=parameters['neural_network_model']['momentum'],
        nesterov=True
    )
    scheduler = MultiStepLR(
        optimizer=optimizer, 
        milestones=[45], 
        gamma=parameters['neural_network_model']['momentum'],
        verbose=True
    )

    # To monitor training loss
    train_loss_history = Averager()

    # To store training loss and mAP values.
    train_loss_list = []
    map_50_list = []
    map_list = []


    # Mame to save the trained model with.
    # MODEL_NAME = 'model'

    # Whether to show transformed images from data loader or not.
    # if VISUALIZE_TRANSFORMED_IMAGES:
    #     # from custom_utils import show_tranformed_image
    #     show_tranformed_image(train_loader)

    # To save best model.
    save_best_model_obj = SaveBestModel()

    # Training loop
    for epoch in range(parameters['neural_network_model']['number_epochs']):

        logging_info(f'')
        logging_info(f"Epoch {epoch+1} of {parameters['neural_network_model']['number_epochs']} - starting" + LINE_FEED)

        # Reset the training loss histories for the current epoch.
        train_loss_history.reset()

        # Start timer and carry out training and validation.
        start = time.time()
        train_loss = train(train_dataloader, model, device, optimizer, train_loss_history)

        # metric_summary, metric_dice_score_summary = validate(valid_dataloader, model, device)
        metric_summary = validate(valid_dataloader, model, device)

        logging_info(f"Epoch #{epoch+1} train loss: {train_loss_history.value:.3f}")
        logging_info(f"Epoch #{epoch+1} mAP@0.50:0.95: {metric_summary['map']}")
        logging_info(f"Epoch #{epoch+1} mAP@0.50: {metric_summary['map_50']}")
        # logging_info(f"Epoch #{epoch+1} Dice score: {metric_dice_score_summary}")
        # logging_info(f"Epoch #{epoch+1} f1 score: {metric_f1_score}")
        end = time.time()
        logging_info(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch+1}")

        train_loss_list.append(train_loss)
        map_50_list.append(metric_summary['map_50'])
        map_list.append(metric_summary['map'])

        # # setting output results folder 
        # output_results_folder = parameters['training_results']['output_results_folder']
        # logging_info(f'Results - output_results_folder: {output_results_folder}')

        # save the best model till now.
        save_best_model_obj(
            model,
            float(metric_summary['map']),
            epoch,
            parameters['training_results']['weights_folder'],
            parameters['training_results']['weights_base_filename']
        )
        # Save the current epoch model.
        save_model( epoch, model, optimizer, 
                    parameters['training_results']['weights_folder'],
                    parameters['training_results']['weights_base_filename']
        )

        # Save loss plot.
        save_loss_plot(parameters['training_results']['metrics_folder'], train_loss_list)

        # Save mAP plot.
        save_mAP(parameters['training_results']['metrics_folder'], map_50_list, map_list)
        scheduler.step()


# ####################
# Original main method 
# ####################
if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    train_dataset = create_train_dataset(TRAIN_DIR)
    valid_dataset = create_valid_dataset(VALID_DIR)
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    logging_info(f"Number of training samples: {len(train_dataset)}")
    logging_info(f"Number of validation samples: {len(valid_dataset)}\n")

    # Initialize the model and move to the computation device.
    model = create_model(num_classes=NUM_CLASSES, size=RESIZE_TO)
    model = model.to(DEVICE)
    logging_info(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    logging_info(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logging_info(f"{total_trainable_params:,} training parameters.")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.0005, momentum=0.9, nesterov=True
    )
    scheduler = MultiStepLR(
        optimizer=optimizer, milestones=[45], gamma=0.1, verbose=True
    )

    # To monitor training loss
    train_loss_hist = Averager()
    # To store training loss and mAP values.
    train_loss_list = []
    map_50_list = []
    map_list = []

    # Mame to save the trained model with.
    MODEL_NAME = 'model'

    # Whether to show transformed images from data loader or not.
    if VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_tranformed_image
        show_tranformed_image(train_loader)

    # To save best model.
    save_best_model_obj = SaveBestModel()

    # Training loop.
    for epoch in range(NUM_EPOCHS):
        logging_info(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        # Reset the training loss histories for the current epoch.
        train_loss_hist.reset()

        # Start timer and carry out training and validation.
        start = time.time()
        train_loss = train(train_loader, model)
        metric_summary = validate(valid_loader, model)
        logging_info(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
        logging_info(f"Epoch #{epoch+1} mAP@0.50:0.95: {metric_summary['map']}")
        logging_info(f"Epoch #{epoch+1} mAP@0.50: {metric_summary['map_50']}")   
        end = time.time()
        logging_info(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        train_loss_list.append(train_loss)
        map_50_list.append(metric_summary['map_50'])
        map_list.append(metric_summary['map'])

        # save the best model till now.
        save_best_model_obj(
            model, float(metric_summary['map']), epoch, 'outputs'
        )
        # Save the current epoch model.
        save_model(epoch, model, optimizer, None)

        # Save loss plot.
        save_loss_plot(OUT_DIR, train_loss_list)

        # Save mAP plot.
        save_mAP(OUT_DIR, map_50_list, map_list)
        scheduler.step()