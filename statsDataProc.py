#%%
import pandas as pd
# %%
data = pd.read_excel("./TournamentReportMaster2020.xlsx", sheet_name=2)
# %%
data
# %%
data_primary = data[data.Primary == 1.0]
data_primary.reset_index(inplace=True)
data_primary
# %%
data_primary.groupby(['Faction'])['Win % By Primary'].mean()
# %%
import numpy as np
detach_winrate = data.groupby(['Player', 'Date'])[['Primary', 'Win % By Primary']].agg(
    Num_detachments=pd.NamedAgg(column='Primary', aggfunc='count'),
    winrate=pd.NamedAgg(column='Win % By Primary', aggfunc='mean')
    )
# %%
import seaborn as sns
sns.boxplot(data=detach_winrate, x='Num_detachments', y='winrate')
# %%
secondary_factions =[]
data_secondary = data[data.Primary != 1.0]
for player in zip(data_primary.Player,data_primary.Date):
    temp = data_secondary[data_secondary.Player==player[0]]
    temp = temp[temp.Date==player[1]]
    secondary_factions.append([faction for faction in temp.Faction])
data_primary['SecondaryFactions'] = secondary_factions

# %%
data_primary
# %%
Allies = data_primary[['Faction', 'SecondaryFactions', 'Win % By Primary']].explode('SecondaryFactions')
Edgelist = Allies.groupby(['Faction', 'SecondaryFactions']).mean().reset_index()
Edgelist['Win % By Primary'] = Edgelist['Win % By Primary'] - Edgelist.groupby('Faction')['Win % By Primary'].transform(np.mean)
# %%
import networkx as nx
import csv
Edgelist.to_csv('test.edgelist', sep=',', header=False, index=False, quoting=csv.QUOTE_NONE)
Data = open('test.edgelist', "r")
Graphtype = nx.DiGraph()

G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                      nodetype=str, data=(('weight', float),))

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
fig, ax =plt.subplots(1,1,figsize=[15,15])
pos = nx.layout.spring_layout(G, k = 1)
nodes = nx.draw_networkx_nodes(G, pos, node_color='grey', alpha=0.3)
edges = nx.draw_networkx_edges(G, pos, arrowstyle='->',
                               arrowsize=15, edge_color=Edgelist['Win % By Primary'],
                               edge_cmap=plt.cm.PiYG, width=3, with_labels=True)
#labels ={i:label for i,label in enumerate(list(G.nodes))}
nx.draw_networkx_labels(G,pos,font_size=16)
pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.PiYG)
pc.set_array(Edgelist['Win % By Primary'])
cbar = plt.colorbar(pc)
cbar.set_label('Win % diff. to average', fontsize = '14')
ax.set_axis_off()
plt.savefig('AlliesNetwork.png')
# %%
fig, ax =plt.subplots(1,1,figsize=[20,20])
A = nx.adjacency_matrix(G).todense()
AlliesDF = pd.DataFrame({node:np.array(vector).flatten() for node, vector in zip(list(G.nodes), A)},
columns=list(G.nodes))
AlliesDF.index = list(G.nodes)
sns.set(font_scale=2)
sns.heatmap(AlliesDF, cmap = 'PiYG' ,ax=ax, 
cbar_kws={'label': 'Win % diff. to average'}, annot=True,
annot_kws={"fontsize":9}, fmt='0.1f')
ax.xaxis.tick_top()
plt.xticks(rotation=90)
plt.savefig('AlliesHeatmap.png', transparent=True, bbox_inches='tight')
# %%
Edgelist[Edgelist['Faction'] == 'Chaos Space Marines']
# %%
data_primary[data_primary['SecondaryFactions'] == '['Ultramarines', 'Chaos Space Marines', 'Renegade Knights']]
# %%
data.to_csv('TournamentReportMaster2020.csv')
# %%
csm = data_primary[data_primary['Faction'] == 'Chaos Space Marines']
for i, secondaries in enumerate(list(csm['SecondaryFactions'])):
    if 'Ultramarines' in secondaries:
        print(csm.iloc[i])
# %%
AlliesDF
# %%
{node:vector.flatten() for node, vector in zip(list(G.nodes), A)}
# %%
[A]
# %%
