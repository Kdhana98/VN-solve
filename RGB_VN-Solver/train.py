import random
import os
import shutil

import numpy.random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
import numpy as np
import re

torch.backends.cudnn.deterministic = True

def sort_key(string):
    # Extract numeric portion of the string using regular expression
    numeric_part = re.search(r'\d+', string).group()
    # Convert the numeric part to integer for comparison
    return int(numeric_part)

# choose among [3, 7, 11, 13, 29]
seed = 29
# set seed for reproducibility
random.seed(seed)
# select among random, ellipse, random
visualization = 'spiral'
# select between medium and small
model_size = 'medium'
# select between random or uniform
coloring = 'uniform'
# select  among 1, 2, or 3 for the train/val splits of 80/20, 160/40, or 800/200
tv = 1
node_size = 0.5
edge_width = 0.1
vertical = 1
resolution = 0.35


if tv == 1:
    train_size = 80
    val_size = 20
if tv == 2:
    train_size = 160
    val_size = 40
if tv == 3:
    train_size = 800
    val_size = 200

test_size = 500
size = 224
resize = False
params = None


if 'ellipse' in visualization:
    if params != None:
        params = str(vertical).replace('.', 'p') + '_' + params
        visualization = str(vertical).replace('.', 'p') + '_' + visualization
    else:
        params = str(vertical).replace('.', 'p') + '_' + coloring + "_" + visualization + "_" + str(
            model_size) + "_" + str(seed) + '_' + str(
            train_size) + "_" + str(test_size) + "_" + str(val_size)
        visualization = str(vertical).replace('.', 'p') + '_' + coloring + '_color_' + visualization

if node_size != 0.5:
    if params != None:
        params = 'node_size_'+str(node_size).replace('.', 'p') +'_' + params
        visualization = 'node_size_'+str(node_size).replace('.', 'p') +'_' + visualization
    else:
        params = 'node_size_'+str(node_size).replace('.', 'p') +'_' + coloring + "_" + visualization + "_" + str(model_size) + "_" + str(seed) + '_' + str(
            train_size) + "_" + str(test_size) + "_" + str(val_size)
        visualization = 'node_size_'+str(node_size).replace('.', 'p') +'_' + coloring + '_color_' + visualization

if edge_width != 0.1:
    if node_size == 0.5 and params==None:
        params = 'edge_width_'+str(edge_width).replace('.', 'p') +'_' + coloring + "_" + visualization + "_" + str(model_size) + "_" + str(
            seed) + '_' + str(
            train_size) + "_" + str(test_size) + "_" + str(val_size)
        visualization = 'edge_width_'+str(edge_width).replace('.', 'p') +'_' + coloring + '_color_' + visualization
    else:
        params = 'edge_width_'+str(edge_width).replace('.', 'p') +'_' + params
        visualization = 'edge_width_'+str(edge_width).replace('.', 'p') +'_' + visualization


if visualization == 'spiral':

    params = str(resolution).replace('.', 'p') + '_sp_' + coloring + "_" + visualization + "_" + str(model_size) + "_" + str(
        seed) + '_' + str(
        train_size) + "_" + str(test_size) + "_" + str(val_size)
    visualization = str(resolution).replace('.', 'p') + '_sp_' + coloring + '_color_' + visualization

elif params == None:
    params = coloring + "_" + visualization+"_"+str(model_size)+"_"+str(seed)+'_'+str(train_size)+"_"+str(test_size)+"_"+str(val_size)
    visualization = coloring + '_color_' + visualization
#params = visualization+"_"+str(model_size)+"_"+str(seed)+'_'+str(train_size)+"_"+str(test_size)+"_"+str(val_size)

# Set device


# Define image directories
non_hamiltonian_dir = visualization+'_non_hamiltonian_'+str(model_size)+'/'
hamiltonian_dir = visualization+'_hamiltonian_'+str(model_size)+'/'
random.seed(seed)
# Combine non_hamiltonian and hamiltonian images
all_images = os.listdir(non_hamiltonian_dir) + os.listdir(hamiltonian_dir)
#sort

all_images = sorted(all_images, key=sort_key)
#all_images = sorted(all_images, key=lambda x: (x.split('_')[0], int(x.split('_')[1])))

random.shuffle(all_images)

# Create train, validation, and test directories
os.makedirs(visualization+'/data_'+str(params)+'/train/hamiltonian', exist_ok=True)
os.makedirs(visualization+'/data_'+str(params)+'/val/hamiltonian', exist_ok=True)
os.makedirs(visualization+'/data_'+str(params)+'/test/hamiltonian', exist_ok=True)

os.makedirs(visualization+'/data_'+str(params)+'/train/non_hamiltonian', exist_ok=True)
os.makedirs(visualization+'/data_'+str(params)+'/val/non_hamiltonian', exist_ok=True)
os.makedirs(visualization+'/data_'+str(params)+'/test/non_hamiltonian', exist_ok=True)

if os.path.exists('data_' + str(params) + '_saved_models'):
    # Remove the directory and all its contents
    shutil.rmtree('data_' + str(params) + '_saved_models')

# Create the new directory
os.makedirs(visualization+'/data_' + str(params) + '_saved_models', exist_ok=True)

# Split the sampled images into train, validation, and test sets


# Move images to train, validation, and test folders
for i, image in enumerate(all_images[:train_size]):
    try:
        shutil.copy(os.path.join(non_hamiltonian_dir, image), os.path.join(visualization+'/data_'+str(params)+'/train/non_hamiltonian', image))
    except:
        shutil.copy(os.path.join(hamiltonian_dir, image), os.path.join(visualization+'/data_' + str(params) + '/train/hamiltonian', image))
for i, image in enumerate(all_images[train_size:train_size+val_size]):
    try:
        shutil.copy(os.path.join(non_hamiltonian_dir, image), os.path.join(visualization+'/data_'+str(params)+'/val/non_hamiltonian', image))
    except:
        shutil.copy(os.path.join(hamiltonian_dir, image), os.path.join(visualization+'/data_' + str(params) + '/val/hamiltonian', image))
for i, image in enumerate(all_images[train_size+val_size:train_size+val_size+test_size]):
    try:
        shutil.copy(os.path.join(non_hamiltonian_dir, image), os.path.join(visualization+'/data_'+str(params)+'/test/non_hamiltonian', image))
    except:
        shutil.copy(os.path.join(hamiltonian_dir, image), os.path.join(visualization+'/data_' + str(params) + '/test/hamiltonian', image))

seed_model = 23
torch.manual_seed(seed_model)
torch.cuda.manual_seed(seed_model)
numpy.random.seed(seed_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ResNet model
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Binary classification, 2 output classes
model = nn.DataParallel(model)  # Utilize multiple GPUs
model = model.to(device)

if resize == True:
    # Set up data loaders
    transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    ])
if coloring == 'gray':
    transform = transforms.Compose([
        transforms.Grayscale(),
        #transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
else:
    transform = transforms.Compose([
        #transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

train_dataset = datasets.ImageFolder(visualization+'/data_' + str(params) + '/train', transform=transform)
val_dataset = datasets.ImageFolder(visualization+'/data_' + str(params) + '/val', transform=transform)
test_dataset = datasets.ImageFolder(visualization+'/data_' + str(params) + '/test', transform=transform)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.09)

# Training loop
num_epochs = 200
best_val_f1 = 0.0
early_stopping_patience = 8
early_stopping_counter = 0
# Create a dataframe to store metrics at each epoch
metrics_df = pd.DataFrame(columns=['Epoch', 'AUC', 'Accuracy', 'F1', 'Recall'])
for epoch in range(num_epochs):
    # Test phase (epoch 0 only)
    if epoch == 0:
        test_correct = 0
        model.eval()

        y_true_list = []
        y_pred_list = []
        y_pred_prob_list = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                test_correct += torch.sum(preds == labels.data)

                # Convert labels and predictions to numpy arrays
                labels_np = labels.cpu().numpy()
                preds_np = preds.cpu().numpy()

                # Append true labels and predicted probabilities to the lists
                y_true_list.extend(labels_np.tolist())
                y_pred_list.extend(preds_np.tolist())
                y_pred_prob_list.extend(outputs[:, 1].cpu().numpy().tolist())

            # Convert the lists to numpy arrays
            y_true = np.array(y_true_list)
            y_pred = np.array(y_pred_list)
            y_pred_prob = np.array(y_pred_prob_list)

            test_acc = test_correct.double() / len(test_dataset)
            test_acc = test_acc.cpu().numpy()
            #y_true = test_dataset.targets
            # y_scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            auc = roc_auc_score(y_true, y_pred_prob)
            f1 = f1_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            metrics_df.loc[epoch] = [epoch, auc, test_acc, f1, recall]

            print(f"Test Acc (Epoch 0): {test_acc:.4f} | AUC: {auc:.4f}")

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print('-' * 10)
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        train_loss += loss.item() * images.size(0)
        train_correct += torch.sum(preds == labels.data)

    train_loss = train_loss / len(train_dataset)
    train_acc = train_correct.double() / len(train_dataset)
    # Save the model after each epoch
    model_path = os.path.join(visualization+'/data_' + str(params) + '_saved_models', f'model_epoch_{epoch}.pt')
    torch.save(model.state_dict(), model_path)
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0

    y_true_list = []
    y_pred_list = []
    y_pred_prob_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * images.size(0)
            val_correct += torch.sum(preds == labels.data)

            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()

            # Append true labels and predicted probabilities to the lists
            y_true_list.extend(labels_np.tolist())
            y_pred_list.extend(preds_np.tolist())
            y_pred_prob_list.extend(outputs[:, 1].cpu().numpy().tolist())

        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct.double() / len(val_dataset)
        val_acc = val_acc.cpu().numpy()
        # Calculate metrics
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        y_pred_prob = np.array(y_pred_prob_list)
        #y_scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        auc = roc_auc_score(y_true, y_pred_prob)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        # Print training and validation metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | AUC: {auc:.4f}")

        # Check for early stopping
        if f1 > best_val_f1:
            best_val_f1 = f1
            early_stopping_counter = 0

            # Save the best model
            torch.save(model.state_dict(), 'best_model_'+str(params)+'.pt')

            print("Best model saved!")

        else:
            early_stopping_counter += 1

        # Add metrics to the dataframe
        metrics_df.loc[epoch] = [epoch, auc, val_acc, f1, recall]

        # Check if early stopping criteria are met
        if early_stopping_counter >= early_stopping_patience:
            if best_val_f1 == 0:
                torch.save(model.state_dict(), 'best_model_' + str(params) + '.pt')
                #continue
            print("Early stopping!")
            break

    scheduler.step()

# Save metrics to a CSV file
metrics_df.to_csv('metrics_'+str(params)+'.csv', index=False)
print (params)