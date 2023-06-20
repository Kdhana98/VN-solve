import networkx as nx
import matplotlib.pyplot as plt
#import pandas as pd
import os
import numpy as np
import random


horizontal = 1
# Vertical indicates b parameter in ellipse layout.
# Circular layout is a special case of ellipse layout when vertical=1.
vertical = 1
# resolution indicates resolution parameter in spiral layout.
resolution = 0.35
# coloring = ['gray', 'random', 'uniform']
coloring = 'random'
# visualization = ['ellipse', 'random', 'spiral']
visualization = 'random'
# model_size = ['small', 'medium']
model_size = 'medium'
# by default node_size = 0.5 and edge_size=0.1
node_size = 0.5
edge_width = 0.1


random.seed(23)
np.random.seed(23)

if model_size == 'medium':
    min_node = 20
    max_node = 50
if model_size == 'small':
    min_node = 4
    max_node = 20


visualization = coloring + '_color_' + visualization

if 'ellipse' in visualization:
    visualization = str(vertical).replace('.', 'p') + '_' + visualization

if 'spiral' in visualization:
    visualization = str(resolution).replace('.', 'p') + '_sp_' + visualization

if node_size != 0.5:
    visualization = 'node_size_' + str(node_size).replace('.', 'p') + '_' + visualization


if edge_width != 0.1:
    visualization = 'edge_width_'+ str(edge_width).replace('.', 'p') +'_' + visualization

print(visualization)
i = -1
cycles = ['non_hamiltonian', 'hamiltonian']
for each in cycles:

    graph_files = [file for file in os.listdir(each) if file.endswith('.mat')]
    for file in graph_files:
        flag = True
        with open(str(each)+'/'+str(file), 'r') as f:
            matrices_text = f.read().strip().split('\n\n')

        graphs = []
        matrixs = []
        for matrix_text in matrices_text:
            if '1' in matrix_text:
                count = matrix_text.count('\n')
                if count >= min_node and count < max_node:
                    lines = matrix_text.strip().split('\n')
                    matrix = np.loadtxt(lines, dtype=int)
                    matrixs.append(matrix)
                    graph = nx.from_numpy_array(matrix)
                    graphs.append(graph)
        for j in range(len(graphs)):
            i += 1
            # pos = nx.circular_layout(graphs[j])  # Layout algorithm
            if visualization.endswith('random'):
                pos = nx.random_layout(graphs[j])  # Layout algorithm
            if 'circular' in visualization:
                pos = nx.circular_layout(graphs[j])  # Layout algorithm
            if 'spiral' in visualization:
                pos = nx.spiral_layout(graphs[j], resolution=resolution)  # Layout algorithm
            if 'ellipse' in visualization:
                pos = nx.ellipse_layout(graphs[j], horizontal=horizontal, vertical=vertical)

            fig, ax = plt.subplots(figsize=(4, 4))
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            if coloring == 'gray':
                node_color = 'gray'
                edge_color = 'gray'
                plt.gray()
            if coloring == 'random':
                node_color = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for i in
                              range(len(pos))]
                edge_color = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for i in
                              range(len(pos))]
            if coloring == 'uniform':
                node_color = 'red'
                edge_color = 'black'
            nx.draw_networkx(graphs[j], pos, with_labels=False, node_color=node_color, node_size=node_size, edge_color=edge_color, width=edge_width)
            ax.set_xlim([min(pos[node][0] for node in graphs[j].nodes)-0.05, max(pos[node][0] for node in graphs[j].nodes)+0.05])
            ax.set_ylim([min(pos[node][1] for node in graphs[j].nodes)-0.05, max(pos[node][1] for node in graphs[j].nodes)+0.05])
            if coloring == 'gray':
                plt.gray()

            ax = plt.gca()
            ax.collections[0].set_edgecolor("none")
            #plt.show()
            os.makedirs(visualization + '_' +str(each) + '_'+ model_size, exist_ok=True)
            os.makedirs(visualization + '_' + str(each) + '_' + model_size+ '_mat', exist_ok=True)
            plt.savefig( visualization + '_' +str(each) + '_'+ model_size +'/' + str(each) + '_' + str(i) + '.png', dpi=56)
            plt.close()
            mat_file_name = visualization + '_' + str(each) + '_' + model_size+ '_mat/' + str(each) + '_' + str(i)
            np.save(mat_file_name, matrixs[j])
print (i)