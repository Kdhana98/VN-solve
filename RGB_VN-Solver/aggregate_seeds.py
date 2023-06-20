import pandas as pd
import os
import re

# After calling test_on_epochs.py, we aggregate seeds. The visualization parameter should be set based on the experiment

visualization = '0p35_sp_uniform_spiral'
visualization = 'uniform_random'

tn_s = 160
tt_s = 500
val_s = 40
model_size = 'small'
test_size = 'small'

matching_files = [
    file for file in os.listdir()
    if file.startswith(visualization + '_' + model_size + '_' + test_size + '_' ) and file.endswith( str(val_s) + ".csv")
]


df = pd.DataFrame(index=range(50))
for f in matching_files:
    df1 = pd.read_csv(f)
    seed = re.search(r'epochs_(\d+)_', f)
    seed = str(seed.group(1))
    df['F1_'+seed] = df1['F1']
    print ('hi')

df['Mean'] = df.mean(axis=1)
df['Std'] = df.std(axis=1)
df = df.dropna(how='all')
new_string = f.replace("epochs_"+seed+"_", "")
new_string = 'result_' + new_string
df.to_csv(new_string)

print('hi')


