import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import random
torch.backends.cudnn.deterministic = True
# chose from ellipse, spiral, random
visualization = 'spiral'
# this file is for testing gray colored images on the best trained model on gray images
coloring = 'gray'
# chose model size from "small" or "medium"
model_size = 'small'
# chose from 1, 2, 3 referring to 80-20, 160-40, or 800-200 splits if data.
tv = 3

if tv == 1:
    tn_s = 80
    val_s = 20
if tv == 2:
    tn_s = 160
    val_s = 40
if tv == 3:
    tn_s = 800
    val_s = 200

tt_s = 500

resize = False
test_dir = None
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.DataFrame(columns=['seed', 'AUC', 'Acc', 'F1', 'Recall'])

for s in [3, 7, 11, 13, 29]:
    if resize == True:
        test_dir = visualization + '/data_transformed_' + visualization + '_' + model_size + '_' + str(s) + '_' + str(
            tn_s) + '_' + str(tt_s) + '_' + str(val_s) + '/test'

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    else:
        test_dir = coloring + '_color_' + visualization + '/data_' + coloring + '_' + visualization + '_' + model_size + '_' + str(s) + '_' + str(tn_s) + '_' + str(tt_s) + '_' + str(val_s) + '/test'
        if coloring == 'gray':
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
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
    best_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = best_model.fc.in_features
    best_model.fc = nn.Linear(num_ftrs, 2)
    model = nn.DataParallel(best_model)  # Utilize multiple GPUs
    model = model.to(device)

    if resize == True:
        model.load_state_dict(torch.load(
            'best_model_transformed_' + visualization + '_' + model_size + '_' + str(s) + '_' + str(tn_s) + '_' + str(tt_s) + '_' + str(
                val_s) + '.pt'))
    else:
        model.load_state_dict(torch.load(
            'best_model_' + coloring+ '_' + visualization + '_' + model_size + '_' + str(s) + '_' + str(
                tn_s) + '_' + str(tt_s) + '_' + str(val_s) + '.pt'))

    # Evaluate the best model on the test set
    model.eval()
    test_correct = 0

    with torch.no_grad():
        all_preds = []
        all_labels = []
        test_correct = 0
        tp = 0
        fn = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)  # Obtain probabilities using softmax
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
        test_acc = test_acc.cpu()
        recall = recall.cpu()

        new_row = pd.Series({'seed': s, 'AUC': auc, 'Acc': test_acc, 'F1': f1, 'Recall': recall})

        # Add row using loc indexer
        df.loc[len(df.index)] = [s, auc, test_acc, f1, recall]

        print(f"Test Acc: {test_acc:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")

statistics = df.agg(['mean', 'std'])

# Print the mean and standard deviation of each column
for column in df.columns:
    mean_value = round(statistics.loc['mean', column], 2)
    std_value = round(statistics.loc['std', column], 2)
    print(f"Column '{column}':")
    print(f"  Mean ± Std: {mean_value} ± {std_value}\n")
