import importlib
import os

# Create a folder in Graphormer and name it customized_data_set. Place this code in the folder

try:
    importlib.import_module("graphormer.data")
except:
    importlib.import_module("Graphormer.graphormer.data")

from graphormer.data import register_dataset
from dgl.data import QM9
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from dgl.data import DGLDataset
import dgl
from numpy import savetxt
import random

model_size = "small"
seed = 13
tn_s = 160
val_s = 40
tt_s = 500
name = str(tn_s)+"_"+str(val_s)+"_hamiltonian_cycle_" + model_size + "_" + str(seed)
combined = False


class SyntheticDataset(DGLDataset):
    def __init__(self):
        super().__init__(name=name)

    def process(self):

        if model_size == 'small':
            edges = pd.read_csv("../../customized_dataset/graph_edges_le_20.csv")
            properties = pd.read_csv("../../customized_dataset/graph_properties_le_20.csv")
        if model_size == 'medium':
            edges = pd.read_csv("../../customized_dataset/graph_edges_g_20_le_50.csv")
            properties = pd.read_csv("../../customized_dataset/graph_properties_g_20_le_50.csv")
        # edges = pd.read_csv("graph_edges_3.csv")
        # properties = pd.read_csv("graph_properties_3.csv")
        edges = edges.drop(columns='Unnamed: 0')
        properties = properties.drop(columns='Unnamed: 0')
        self.graphs = []
        self.labels = []

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row["graph_id"]] = row["label"]
            num_nodes_dict[row["graph_id"]] = row["num_nodes"]

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby("graph_id")

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id["src"].to_numpy()
            dst = edges_of_id["dst"].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            self.graphs.append(g)
            self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


@register_dataset(name)
def create_customized_dataset():
    # dataset = QM9(label_keys=["mu"])
    dataset = SyntheticDataset()
    num_graphs = len(dataset)

    cwd = os.getcwd()
    print(cwd)

    dataset = SyntheticDataset()
    num_graphs = len(dataset)
    random.seed(seed)
    graph_ids = [i for i in range(num_graphs)]
    random.shuffle(graph_ids)
    sampled_objects = np.array(graph_ids)
    dataset_folder = name
    os.makedirs(dataset_folder, exist_ok=True)

    train_objects = sampled_objects[:tn_s]
    validation_objects = sampled_objects[tn_s:tn_s + val_s]
    test_objects = sampled_objects[tn_s + val_s:tn_s + val_s + tt_s]

    #train_objects = np.load(name + '/train_objects.npy')
    #test_objects = np.load(name + '/test_objects.npy')
    #validation_objects = np.load(name + '/val_objects.npy')
    if combined:
        train_valid_idx, test_objects = train_test_split(
            np.arange(num_graphs), test_size=num_graphs // 10, random_state=0
        )
        train_objects, validation_objects = train_test_split(
            train_valid_idx, test_size=(num_graphs * 18) // 100, random_state=0
        )
        test_objects = np.concatenate((test_objects, validation_objects, train_objects))
        #if model_size == 'small':
        #    test_objects = test_objects[:3710]
        #if model_size == 'medium':
        #    test_objects = test_objects[:13193]
    return {
        "dataset": dataset,
        # "train_idx": train_idx_2,
        "train_idx": train_objects,
        "valid_idx": validation_objects,
        "test_idx": test_objects,
        "source": "dgl"
    }

