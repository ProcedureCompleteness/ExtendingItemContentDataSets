# -*- coding: utf-8 -*-

## Step 1 Duplicate Detection
## Task 1.4 (part 2) validation of duplicate detection

import pandas as pd

pairs_file_name='matching_pairs_idsp1p2.pkl'
print("reading matching_pairs")
matching_pairs = pd.read_pickle(pairs_file_name)


matching_pairs_new = pd.DataFrame(columns=matching_pairs.columns)
for index in matching_pairs.index:
    row = matching_pairs.iloc[[index]]
    if row['id_p2'].values[0] not in matching_pairs_new['id_p2'].values \
    and row['id_p1'].values[0] not in matching_pairs_new['id_p1'].values:
        matching_pairs_new = matching_pairs_new.append(row, ignore_index=True)
print("items with more than one duplicate referenced: "+str(matching_pairs.shape[0]-matching_pairs_new.shape[0]))


## create "labeled_data.xlsx" with prepare_data_for_labeling.py and label the data!
temp_df = pd.read_excel("labeled_data.xlsx", sheet_name='Sheet1')
labeled_df = temp_df[(temp_df['true_label']==1) |(temp_df['true_label']==0)]

tp = 0; fp = 0; tn = 0; fn = 0

errors = list()
for index,row in labeled_df.iterrows():
    id_p2 = row['p2_id']
    id_p1 = row['p1_id']
    true_label = row['true_label']
    pred_label = matching_pairs_new[(matching_pairs_new['id_p2']==id_p2) & (matching_pairs_new['id_p1']==id_p1)].shape[0]
#    pred_label_altern = 0
#    if all()
    if true_label == 1 and pred_label == 1: tp +=1
    if true_label == 0 and pred_label == 1:
        fp +=1
        l = {**dict(row),**{'pred_label':pred_label}}
        errors.append(l)
    if true_label == 0 and pred_label == 0: tn +=1
    if true_label == 1 and pred_label == 0:
        fn +=1
        l = {**dict(row),**{'pred_label':pred_label}}
        errors.append(l)

eval_precision = tp/(tp+fp)
eval_recall = tp/(tp+fn)
eval_f1 = 2*(eval_precision*eval_recall)/(eval_precision+eval_recall)
print('precision: '+str(eval_precision))
print('recall: '+str(eval_recall))
print('f1-measure: '+str(eval_f1))