import os
import random
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import pandas as pd


def generate_random_list(a, b, length):
    random_list = random.choices([0, 1], [a, b], k=length)
    return random_list


visualization = 'circular'
coloring = 'uniform'

model_size = "medium"
seeds = [3, 7, 11, 13, 29]
tn_s = 800
tt_s = 500
val_s = 200

df = pd.DataFrame(columns=['seed', 'AUC', 'Acc', 'F1'])
for s in seeds:
    random.seed(s)
    # Specify the directory path
    directory_train = coloring + '_color_' + visualization+ '/data_' + coloring + '_' + visualization + '_' + model_size + '_' + str(s) + '_' + str(tn_s) + '_' + str(tt_s) + '_' + str(val_s) +'/train'
    directory_test = coloring + '_color_' + visualization+ '/data_' + coloring + '_' + visualization + '_' + model_size + '_' + str(s) + '_' + str(tn_s) + '_' + str(tt_s) + '_' + str(val_s) +'/test'

    directory_train_ham = directory_train + '/hamiltonian'
    directory_train_non_ham = directory_train + '/non_hamiltonian'

    directory_test_ham = directory_test + '/hamiltonian'
    directory_test_non_ham = directory_test + '/non_hamiltonian'

    train_ham_list = os.listdir(directory_train_ham)
    train_non_ham_list = os.listdir(directory_train_non_ham)
    test_ham_list = os.listdir(directory_test_ham)
    test_non_ham_list = os.listdir(directory_test_non_ham)

    number_train_ham_list = len(train_ham_list)
    number_train_non_ham_list = len(train_non_ham_list)
    number_test_ham_list = len(test_ham_list)
    number_test_non_ham_list = len(test_non_ham_list)

    test_label = generate_random_list(number_test_non_ham_list, number_test_ham_list, tt_s)
    #count_1 = test_label.count(1)
    threshold = float(number_train_non_ham_list)/float(tn_s)
    #test_pred = [1 if random.random() > threshold else 0 for _ in range(tt_s)]
    test_pred = [1 if random.random() > 0.5 else 0 for _ in range(tt_s)]

    auc = roc_auc_score(test_label, test_pred)

    # Calculate the accuracy
    accuracy = accuracy_score(test_label, test_pred)

    # Calculate the F1 score
    f1 = f1_score(test_label, test_pred)

    df2 = {'seed': s, 'AUC': auc, 'Acc': accuracy, 'F1': f1}
    df = df._append(df2, ignore_index=True)


statistics = df.agg(['mean', 'std'])

# Print the mean and standard deviation of each column
for column in df.columns:
    mean_value = round(statistics.loc['mean', column], 2)
    std_value = round(statistics.loc['std', column], 2)
    print(f"Column '{column}':")
    print(f"  Mean ± Std: {mean_value} ± {std_value}\n")



