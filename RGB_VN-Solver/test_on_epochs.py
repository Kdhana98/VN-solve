import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os, csv
import torch

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import random
import matplotlib.pyplot as plt


def append_to_csv(filename, data):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Epoch', 'AUC', 'ACC', 'F1', 'Recall'])  # Write header row if file is newly created
        writer.writerow(data)  # Append data as a row

visualization = 'random'
vertical = 1
coloring = 'uniform'
model_size = 'small'
test_size = 'small'
train_size = 80
val_size = 20
test_size_num = 500
# choosing a seed value from [3, 7, 11, 13, 29]
s = 3
resolution = 0.35


size = 224
resize = False
if model_size == 'all':
    folder_path = 'all_for_aws_2/data_'+model_size+'_'+str(s)+'_'+str(train_size)+"_"+str(test_size_num)+"_"+str(val_size)+'_saved_models'
else:
    if size == 224:
        if resize == True:
            folder_path = visualization+'/data_transformed' + visualization+ '_' + model_size + '_' + str(s) +'_'+str(train_size)+"_"+str(test_size_num)+"_"+str(val_size)+ '_saved_models'
        if visualization == 'ellipse':
            folder_path = str(vertical).replace('.', 'p') + '_' + coloring + '_color_' +  visualization+'/data_' + str(vertical).replace('.', 'p') + '_' + coloring + '_' + visualization+ '_' + model_size + '_' + str(s) +'_'+str(train_size)+"_"+str(test_size_num)+"_"+str(val_size)+ '_saved_models'
        if visualization == 'spiral':
            folder_path = str(resolution).replace('.', 'p') + '_sp_' + coloring + '_color_' + visualization + '/data_' + str(
                resolution).replace('.', 'p') + '_sp_' + coloring + '_' + visualization + '_' + model_size + '_' + str(
                s) + '_' + str(train_size) + "_" + str(test_size_num) + "_" + str(val_size) + '_saved_models'

        else:
            folder_path = coloring + '_color_' +  visualization+'/data_' + coloring + '_' + visualization+ '_' + model_size + '_' + str(s) +'_'+str(train_size)+"_"+str(test_size_num)+"_"+str(val_size)+ '_saved_models'
    else:
        folder_path = 'data_' + str(size)+'_' + visualization+ '_' + model_size + '_' + str(s) +'_'+str(train_size)+"_"+str(test_size_num)+"_"+str(val_size)+ '_saved_models'

# Get the list of files in the folder
file_list = os.listdir(folder_path)


for epoch in range(len(file_list)):
    file_name = [file_name for file_name in file_list if '_'+str(epoch)+'.pt' in file_name][0]
    model_name = folder_path+'/'+file_name
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_size == 'all':
        test_dir = 'all_for_aws_2/data_'+model_size+'_'+str(s)+'/test'
    else:
        if size == 224:
            if resize == True:
                test_dir = visualization+'/data_transformed' + visualization + '_' +model_size+'_'+str(s)+'_'+str(train_size)+"_"+str(test_size_num)+"_"+str(val_size)+'/test'
            elif visualization == 'ellipse':
                test_dir = str(vertical).replace('.', 'p') + '_' + coloring + '_color_' +visualization+'/data_' + str(vertical).replace('.', 'p') + '_' + coloring + '_' +  visualization + '_' +model_size+'_'+str(s)+'_'+str(train_size)+"_"+str(test_size_num)+"_"+str(val_size)+'/test'
            elif visualization == 'spiral':
                test_dir = str(resolution).replace('.', 'p') + '_sp_' + coloring + '_color_' + visualization + '/data_' + str(
                    resolution).replace('.', 'p') + '_sp_' + coloring + '_' + visualization + '_' + model_size + '_' + str(
                    s) + '_' + str(train_size) + "_" + str(test_size_num) + "_" + str(val_size) + '/test'

            else:
                test_dir = coloring + '_color_' +visualization+'/data_' + coloring + '_' + visualization + '_' +model_size+'_'+str(s)+'_'+str(train_size)+"_"+str(test_size_num)+"_"+str(val_size)+'/test'
        else:
            test_dir = visualization+'/data_'+str(size)+'_'+ visualization + '_' +model_size+'_'+str(s)+'_'+str(train_size)+"_"+str(test_size_num)+"_"+str(val_size)+'/test'


    if resize == True:
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    batch_size = 32
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)


    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=32)

    best_model = models.resnet50(pretrained=False)
    num_ftrs = best_model.fc.in_features
    best_model.fc = nn.Linear(num_ftrs, 2)
    model = nn.DataParallel(best_model)  # Utilize multiple GPUs
    model = model.to(device)

    model.load_state_dict(torch.load(model_name))

    # Evaluate the best model on the test set
    model.eval()
    test_correct = 0

    with torch.no_grad():
        all_preds = []
        all_labels = []
        test_correct = 0
        tp = 0
        fn = 0
        tt = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            st = time.time()
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)  # Obtain probabilities using softmax
            et = time.time()
            dt = et - st
            tt += dt
            _, preds = torch.max(outputs, 1)
            all_preds.extend(probs[:, 1].cpu().numpy())  # Use the probability of the positive class

            all_labels.extend(labels.cpu().numpy())

            test_correct += torch.sum(preds == labels.data)
            tp += torch.sum((preds == 1) & (labels.data == 1))
            fn += torch.sum((preds == 0) & (labels.data == 1))

        test_acc = test_correct.double() / len(test_dataset)
        auc = roc_auc_score(all_labels, all_preds)
        f1 = f1_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))
        recall = tp.double() / (tp.double() + fn.double())

        if size == 224:
            if resize == True:
                filename = 'transformed_' + visualization + '_' + model_size + '_' + test_size + '_epochs_' + str(s)+'_' + str(train_size) + "_" + str(test_size_num) + "_" + str(val_size) + '.csv'
            elif visualization == 'ellipse':
                filename = str(vertical).replace('.', 'p') + '_' + coloring + '_' + visualization + '_' + model_size + '_' + test_size + '_epochs_' + str(s) + '_' + str(train_size) + "_" + str(test_size_num) + "_" + str(val_size) + '.csv'
            elif visualization == 'spiral':
                filename = str(resolution).replace('.', 'p') + '_sp_' + coloring + '_' + visualization + '_' + model_size + '_' + test_size + '_epochs_' + str(
                    s) + '_' + str(train_size) + "_" + str(test_size_num) + "_" + str(val_size) + '.csv'

            else:
                filename = coloring + '_' + visualization + '_' + model_size + '_' + test_size + '_epochs_'+ str(s)+'_' + str(train_size)+"_"+str(test_size_num)+"_"+str(val_size)+'.csv'
        else:
            filename = str(size) + '_' + visualization + '_' + model_size + '_' + test_size + '_epochs_' + str(s)+'_' + str (train_size)+ "_" + str(test_size_num) + "_" + str(val_size) + '.csv'

        print(f"F1 Score: {f1:.4f}")
        data = [epoch, auc, test_acc.cpu().numpy(), f1, recall.cpu().numpy()]
        append_to_csv(filename, data)
