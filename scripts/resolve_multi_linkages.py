import pandas as pd
import numpy as np
import pickle


## Step 1 Duplicate Detection
## Task 1.3 resolution of transitive duplicates

# load result of duplicate detection
pairs_file_name='pairs_with_labels.pkl'
pairs_with_labels = pd.read_pickle("./"+pairs_file_name)

# load ids of pairs
with open("id_pairs.pkl",'rb') as infile:
    id_pairs=pickle.load(infile)

# convert data to dataframe
id_pairs_after_blocking = {'id_p2':list(),'id_p1':list()}
matching_pairs  = {'id_p2':list(),'id_p1':list(),'probs_jurek':list()}
for index,row in pairs_with_labels.iterrows():
    label = row['label_jurek']
    id_p2 = id_pairs[index][0]
    id_p1 = id_pairs[index][1]
    id_pairs_after_blocking['id_p2'].append(id_p2)
    id_pairs_after_blocking['id_p1'].append(id_p1)
    if label == 1:
        matching_pairs['id_p2'].append(id_p2)
        matching_pairs['id_p1'].append(id_p1)
        matching_pairs['probs_jurek'].append(row['probs_jurek'])

id_pairs_after_blocking_df = pd.DataFrame.from_dict(id_pairs_after_blocking)
matching_pairs_df = pd.DataFrame.from_dict(matching_pairs)

## resolve n-to-m linkages from p1 to p2
# first, reduce n-to-m linkages to 1-to-m linkages
p2_matching_pairs = dict()
for index,row in matching_pairs_df.iterrows():
    id_p2 = row['id_p2']
    curr_instance = p2_matching_pairs.setdefault(id_p2,dict())
    ids_p1 = curr_instance.setdefault('id_p1',list())
    probs_jurek = curr_instance.setdefault('probs_jurek',list())
    ids_p1.append(row['id_p1'])
    probs_jurek.append(row['probs_jurek'])

p2_matching_pairs_unique = {'id_p2':list(),'id_p1':list(),'probs_jurek':list()}
for id_p2, row in p2_matching_pairs.items():
    index_max = np.argmax(np.array(row['id_p1']))
    p2_matching_pairs_unique['id_p2'].append(id_p2)
    p2_matching_pairs_unique['id_p1'].append(row['id_p1'][index_max])
    p2_matching_pairs_unique['probs_jurek'].append(row['probs_jurek'][index_max])

p2_matching_pairs_unique = pd.DataFrame(p2_matching_pairs_unique)

# second, reduce 1-to-m linkages to 1-to-1 linkages
p1_matching_pairs = dict()
for index,row in p2_matching_pairs_unique.iterrows():
    id_p1 = row['id_p1']
    curr_instance = p1_matching_pairs.setdefault(id_p1,dict())
    ids_p2 = curr_instance.setdefault('id_p2',list())
    probs_jurek = curr_instance.setdefault('probs_jurek',list())
    ids_p2.append(row['id_p2'])
    probs_jurek.append(row['probs_jurek'])

p1_matching_pairs_unique = {'id_p1':list(),'id_p2':list(),'probs_jurek':list()}
for id_p1, row in p1_matching_pairs.items():
    index_max = np.argmax(np.array(row['id_p2']))
    p1_matching_pairs_unique['id_p1'].append(id_p1)
    p1_matching_pairs_unique['id_p2'].append(row['id_p2'][index_max])
    p1_matching_pairs_unique['probs_jurek'].append(row['probs_jurek'][index_max])

matching_pairs_unique = pd.DataFrame(p1_matching_pairs_unique)

matching_pairs_unique.to_pickle(str("matching_")+pairs_file_name.replace("_with_labels.pkl","_idsp1p2.pkl"))

# save data
pairs_with_labels=pairs_with_labels.reset_index()
pairs_with_labels_and_ids = pd.concat([pairs_with_labels,id_pairs_after_blocking_df],axis=1)
pairs_with_labels_and_ids.to_pickle(pairs_file_name.replace('.pkl','_and_ids.pkl'))