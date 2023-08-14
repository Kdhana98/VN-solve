# VN-Solver: Vision-based Neural Solver for Combinatorial Optimization over Graphs
This repository contains the code for the experiment in the paper entitled ["VN-Solver: Vision-based Neural Solver for Combinatorial Optimization over Graphs"](https://arxiv.org/abs/2308.03185)  accepted in Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM '23).

Our experiments consist of 4 models, namely, Naive Bayesian, Graph Transform [(Graphormer)](https://github.com/Microsoft/Graphormer), Gray-scale VN-Solver, Uniform-color VN-Solver.


## RGB VN-Solver
In the VN-Solver RGB folder, we have the code for generating figures from adjacency matrixes in generate_figure.py where the uniform configuration is used (nodes are red, edges are black). We introduced ellipse layout in layout.py which should be replaced by the layout.py in networkx package. After generating figures, we train 5 ResNet models using 3, 7, 11, 13, 29 as seeds to split the data. The random seed of the model is 23 and the models are trained on 2 GPUs. As we have datasets with 3 different sizes, we defined a tv variable to indicate the size of the training set and validation set. The test_on_best.py evaluate the model with the highest F1 score on the test set while the test_on_epochs.py evaluate models saved in each epoch against the test set. Evaluation of each epoch is further used to draw Figure 3 in the manuscript. The code to generate this figure is available in draw_figures_in_manuscript.py. After evaluating the best model on the test set, we aggregate the results of the seeds to have the final result. This is done in aggregate_seed.py. 


## Gray-scale VN-Solver
The experiment is the same as RGB VN-Solver, except the images are generated in grayscale. Having images with only one channel instead of RGB changes the architecture of the ResNet. As a result, the training (grayscale_train.py) and testing (test_on_best_grayscale.py) for grayscale images are placed in this folder. 


## Graph Transformer
To run the Graphormer experiment, we adopted the graphormer-slim architecture from [(Graphormer)](https://github.com/Microsoft/Graphormer) Github page. Then, to generate the input data, we used the data from [House of Graphs](https://houseofgraphs.org/) website in edge list, i.e., ".lst" format and run the generate_dataset.py code to have graph properties data and edge_data. In the customized dataset folder in Graphormer, we place the hamiltonian_cycle_eval.py code to register the dataset and be able to test and train the Graphormer models. To do a fair comparison, we used 3, 7, 11, 13, and 29 as seeds to split the data which results in having the same graphs as VN-Solver as data to the Graphormer model

## Naive Bayesian
As we run the experiments, for each seed we generate a folder containing test, train, and validation. In each of these folders, some Hamiltonian and some non-Hamiltonian instances exist. In bayesian.py code, we count the proportion of the Hamiltonian instances to all instances in the training set and set that as a threshold to randomly guess true and false instances in the test set.
