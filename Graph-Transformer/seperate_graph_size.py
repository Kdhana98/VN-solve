import pandas as pd

df = pd.read_csv('graph_properties_3.csv', index_col=[0])
df2 = pd.read_csv('graph_edges_3.csv', index_col=[0])

df_less_than_20 = df[df['num_nodes'] <= 20]
df2_less_than_20 = df2[df2['graph_id'].isin(df_less_than_20['graph_id'])]
#df_less_than_20.drop('Unnamed: 0', axis=0)
#df2_less_than_20.drop('Unnamed: 0', axis=0)

# Filter rows with greater than 20 and less than or equal to 50 nodes
df_20_to_50 = df[(df['num_nodes'] > 20) & (df['num_nodes'] <= 50)]
df2_20_to_50 = df2[df2['graph_id'].isin(df_20_to_50['graph_id'])]
#df_20_to_50.drop('Unnamed: 0', axis=0)
#df2_20_to_50.drop('Unnamed: 0', axis=0)

# Filter rows with greater than 50 and less than or equal to 100 nodes
df_50_to_100 = df[(df['num_nodes'] > 50) & (df['num_nodes'] <= 100)]
df2_50_to_100 = df2[df2['graph_id'].isin(df_50_to_100['graph_id'])]


df_less_than_20.to_csv('graph_properties_le_20.csv')
df2_less_than_20.to_csv('graph_edges_le_20.csv')

df_20_to_50.to_csv('graph_properties_g_20_le_50.csv')
df2_20_to_50.to_csv('graph_edges_g_20_le_50.csv')

df_50_to_100.to_csv('graph_properties_g_50_le_100.csv')
df2_50_to_100.to_csv('graph_edges_g_50_le_100.csv')

