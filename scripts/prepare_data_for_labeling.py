import pandas as pd
import json

## Step 1 Duplicate Detection
## Task 1.4 (part 1) validation of duplicate detection

# specify maximum number of data for labeling
number_of_similarity_clusters = 20
max_number_of_data_for_labeling_per_similarity_cluster = 45

# load data
pairs_file_name='pairs_with_labels_and_ids.pkl'
pairs_with_labels = pd.read_pickle(pairs_file_name)

# calculate overall similarity of pairs (more precisely: sum up all similarity values for all key attributes)
sim_score_cols = [col for col in list(pairs_with_labels.columns) if not 'jurek' in col and not '_id']

pairs_with_labels['sum']=pairs_with_labels[sim_score_cols].sum(axis=1)

# get overview numbers of pair similarities
num_exact_matches = pairs_with_labels[pairs_with_labels['sum']==19].shape[0]
num_almost_matches = pairs_with_labels[(pairs_with_labels['sum']>18) & (pairs_with_labels['sum']<19)].shape[0]
num_no_matches = pairs_with_labels[pairs_with_labels['sum']<10].shape[0]

# in the following, for each similarity cluster (based on the similarity sum
# per pair) at most "max_number_of_data_for_labeling_per_similarity_cluster"
# are selected for labeling
ids_for_labeling = pd.DataFrame()
for i in range(number_of_similarity_clusters):
    if bool(list(pairs_with_labels[(pairs_with_labels['sum']>=i) & (pairs_with_labels['sum']<i+1)][['id_p2','id_p1']].index)):
        shuffled_partition = pairs_with_labels[(pairs_with_labels['sum']>=i) & (pairs_with_labels['sum']<i+1)][['id_p2','id_p1']].sample(frac=1)
        ids_for_labeling=pd.concat([ids_for_labeling,shuffled_partition.iloc[:max_number_of_data_for_labeling_per_similarity_cluster]])

with open('../input_data/data_key_attributes_p1.json','r') as infile:
    instances_p1 = json.load(infile)
with open('../input_data/data_key_attributes_p2.json','r') as infile:
    instances_p2 = json.load(infile)


temp_dict = dict()
for index,row in ids_for_labeling.iterrows():
    instance_p2 = instances_p2[row[0]]
    instance_p2['id']=row[0]
    instance_p1 = instances_p1[row[1]]
    instance_p1['id']=row[1]
    for k,v in instance_p2.items():
        curr_list = temp_dict.setdefault('p2_'+k,list())
        curr_list.append(v)
        temp_dict['p2_'+k] = curr_list
    for k,v in instance_p1.items():
        curr_list = temp_dict.setdefault('p1_'+k,list())
        curr_list.append(v)
        temp_dict['p1_'+k] = curr_list
    curr_list = temp_dict.setdefault('true_label',list())
    curr_list.append(None)
    temp_dict['true_label'] = curr_list

data_for_labeling = pd.DataFrame(temp_dict)

# save data for labeling
# Remark: the labeling has to be '1' for duplicate pairs and '0' for non-duplicate pairs in the last column "true label"
data_for_labeling.to_excel("unlabeled_data.xlsx")

