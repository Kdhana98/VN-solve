import pandas as pd
import glob

''' The file list order for non_hamiltonian in our system is:
['non_hamiltonian/list_127_graphs.lst', 'non_hamiltonian/list_2_graphs (2).lst', 'non_hamiltonian/list_487_graphs.lst', 
'non_hamiltonian/list_164_graphs.lst', 'non_hamiltonian/list_456_graphs.lst', 'non_hamiltonian/list_1_graphs (9).lst', 
'non_hamiltonian/list_1_graphs (11).lst', 'non_hamiltonian/list_1_graphs (8).lst', 
'non_hamiltonian/list_46_graphs (1).lst', 'non_hamiltonian/list_38_graphs.lst', 'non_hamiltonian/list_78_graphs.lst', 
'non_hamiltonian/list_18_graphs.lst', 'non_hamiltonian/list_2_graphs.lst', 'non_hamiltonian/list_8_graphs.lst', 
'non_hamiltonian/list_1_graphs (1).lst', 'non_hamiltonian/list_66_graphs.lst', 'non_hamiltonian/list_9_graphs.lst', 
'non_hamiltonian/list_2_graphs (3).lst', 'non_hamiltonian/list_324_graphs.lst', 'non_hamiltonian/list_148_graphs.lst', 
'non_hamiltonian/list_1_graphs (10).lst', 'non_hamiltonian/list_1_graphs (4).lst', 
'non_hamiltonian/list_1_graphs (12).lst', 'non_hamiltonian/list_1_graphs (5).lst', 
'non_hamiltonian/list_3_graphs (3).lst', 'non_hamiltonian/list_409_graphs.lst', 'non_hamiltonian/list_43_graphs.lst', 
'non_hamiltonian/list_3_graphs.lst', 'non_hamiltonian/list_117_graphs.lst', 'non_hamiltonian/list_165_graphs.lst', 
'non_hamiltonian/list_338_graphs.lst', 'non_hamiltonian/list_3_graphs (1).lst', 'non_hamiltonian/list_6_graphs.lst', 
'non_hamiltonian/list_46_graphs.lst', 'non_hamiltonian/list_1_graphs.lst', 'non_hamiltonian/list_306_graphs.lst', 
'non_hamiltonian/list_1_graphs (2).lst', 'non_hamiltonian/list_557_graphs.lst', 'non_hamiltonian/list_782_graphs.lst', 
'non_hamiltonian/list_17_graphs.lst', 'non_hamiltonian/list_185_graphs.lst', 'non_hamiltonian/list_1_graphs (7).lst', 
'non_hamiltonian/list_62_graphs.lst', 'non_hamiltonian/list_1_graphs (3).lst', 'non_hamiltonian/list_121_graphs.lst', 
'non_hamiltonian/list_9_graphs (1).lst', 'non_hamiltonian/list_29_graphs.lst', 'non_hamiltonian/list_26_graphs.lst', 
'non_hamiltonian/list_25_graphs.lst', 'non_hamiltonian/list_510_graphs.lst', 'non_hamiltonian/list_2_graphs (1).lst', 
'non_hamiltonian/list_5_graphs.lst', 'non_hamiltonian/list_149_graphs.lst', 'non_hamiltonian/list_14_graphs.lst', 
'non_hamiltonian/list_3_graphs (2).lst', 'non_hamiltonian/list_1_graphs (6).lst', 'non_hamiltonian/list_58_graphs.lst', 
'non_hamiltonian/list_373_graphs.lst', 'non_hamiltonian/list_809_graphs.lst', 'non_hamiltonian/list_497_graphs.lst']


The file list in our system for hamiltonian is:

['hamiltonian/list_2_graphs (2).lst', 'hamiltonian/list_60_graphs.lst', 'hamiltonian/list_24_graphs.lst', 
'hamiltonian/list_37_graphs.lst', 'hamiltonian/list_436_graphs.lst', 'hamiltonian/list_8_graphs (2).lst', 
'hamiltonian/list_13_graphs.lst', 'hamiltonian/list_4_graphs.lst', 'hamiltonian/list_21_graphs.lst', 
'hamiltonian/list_2_graphs (6).lst', 'hamiltonian/list_1_graphs (9).lst', 'hamiltonian/list_47_graphs.lst', 
'hamiltonian/list_1_graphs (8).lst', 'hamiltonian/list_2_graphs (5).lst', 'hamiltonian/list_126_graphs.lst', 
'hamiltonian/list_42_graphs.lst', 'hamiltonian/list_1842_graphs.lst', 'hamiltonian/list_98_graphs.lst', 
'hamiltonian/list_2_graphs.lst', 'hamiltonian/list_8_graphs.lst', 'hamiltonian/list_2_graphs (4).lst', 
'hamiltonian/list_4_graphs (1).lst', 'hamiltonian/list_353_graphs.lst', 'hamiltonian/list_22_graphs (1).lst', 
'hamiltonian/list_1_graphs (1).lst', 'hamiltonian/list_20_graphs.lst', 'hamiltonian/list_7_graphs (1).lst', 
'hamiltonian/list_60_graphs (1).lst', 'hamiltonian/list_617_graphs.lst', 'hamiltonian/list_9_graphs.lst', 
'hamiltonian/list_2_graphs (3).lst', 'hamiltonian/list_34_graphs.lst', 'hamiltonian/list_166_graphs.lst', 
'hamiltonian/list_21_graphs (1).lst', 'hamiltonian/list_33_graphs.lst', 'hamiltonian/list_21_graphs (3).lst', 
'hamiltonian/list_21_graphs (4).lst', 'hamiltonian/list_42_graphs (1).lst', 'hamiltonian/list_779_graphs.lst', 
'hamiltonian/list_1_graphs (4).lst', 'hamiltonian/list_741_graphs.lst', 'hamiltonian/list_1_graphs (5).lst', 
'hamiltonian/list_8_graphs (1).lst', 'hamiltonian/list_43_graphs.lst', 'hamiltonian/list_21_graphs (2).lst', 
'hamiltonian/list_3_graphs.lst', 'hamiltonian/list_20_graphs (1).lst', 'hamiltonian/list_1362_graphs.lst', 
'hamiltonian/list_189_graphs.lst', 'hamiltonian/list_12_graphs.lst', 'hamiltonian/list_6_graphs.lst', 
'hamiltonian/list_1_graphs.lst', 'hamiltonian/list_72_graphs.lst', 'hamiltonian/list_14_graphs (1).lst', 
'hamiltonian/list_222_graphs.lst', 'hamiltonian/list_7_graphs (2).lst', 'hamiltonian/list_1_graphs (2).lst', 
'hamiltonian/list_119_graphs.lst', 'hamiltonian/list_156_graphs.lst', 'hamiltonian/list_17_graphs.lst', 
'hamiltonian/list_290_graphs.lst', 'hamiltonian/list_1_graphs (7).lst', 'hamiltonian/list_428_graphs.lst',
'hamiltonian/list_7_graphs.lst', 'hamiltonian/list_1_graphs (3).lst', 'hamiltonian/list_26_graphs.lst', 
'hamiltonian/list_2_graphs (1).lst', 'hamiltonian/list_94_graphs.lst', 'hamiltonian/list_11_graphs.lst', 
'hamiltonian/list_5_graphs.lst', 'hamiltonian/list_14_graphs.lst', 'hamiltonian/list_1_graphs (6).lst', 
'hamiltonian/list_565_graphs.lst', 'hamiltonian/list_308_graphs.lst', 'hamiltonian/list_22_graphs.lst', 
'hamiltonian/list_100_graphs.lst', 'hamiltonian/list_28_graphs.lst', 'hamiltonian/list_32_graphs.lst', 
'hamiltonian/list_155_graphs.lst']
'''
folder_path = ["non_hamiltonian", "hamiltonian"]  # Replace with the path to your folder
suffix = "*.lst"  # Replace with the desired suffix


edge_data = pd.DataFrame(columns=['graph_id', 'src', 'dst'])
graph_properties = pd.DataFrame(columns=['graph_id', 'label', 'num_nodes'])
i = -1
for j in folder_path:
    label = 1 if j == "hamiltonian" else 0
    # Get all files with the specified suffix
    file_list = glob.glob(j + "/" + suffix)

    # Iterate over the dictionary
    for f in file_list:
        with open(f, "r") as file:
            # Read the entire file content
            file_content = file.read()
            if file_content[-1] == file_content[-2] and file_content[-1] == '\n':
                file_content = file_content[:-2]
        lists = []
        lists = file_content.split('\n\n')
        for l in lists:
            data = {}
            # Split the data string into lines
            lines = l.split('\n')
            num_nodes = len(lines)
            if num_nodes >= 20:
                continue
            count = sum([len(item.split(":")[1].split()) for item in lines[1:]])
            #if count == 0:
            #    continue
            # Iterate over each line
            i += 1
            for line in lines:
                # Split the line into key and values
                key, values = line.split(':')
                values = values.strip().split(' ')
                try:
                    for value in values:
                        edge_data.loc[len(edge_data.index)] = [i, int(key)-1, int(value)-1]
                except:
                    continue

            graph_properties.loc[len(graph_properties.index)] = [i, label, num_nodes]

edge_data.to_csv('edge_data_small.csv')
graph_properties.to_csv('graph_properties_small.csv')




