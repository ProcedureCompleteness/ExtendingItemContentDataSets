import pandas as pd
import tqdm
import numpy as np
from sklearn.impute import SimpleImputer 

## Step 2 Data Integration

# read data
matching_pairs = pd.read_pickle("./matching_pairs_idsp1p2.pkl")
ratings_p1= pd.read_json('../input_data/data_ratings_p1.json')
ratings_p2 = pd.read_json('../input_data/data_ratings_p2.json')
nominal_features_p1 = pd.read_json('../input_data/data_nominal_attributes_p1.json')
nominal_features_p2 = pd.read_json('../input_data/data_nominal_attributes_p2.json')
binary_features_p1 = pd.read_json('../input_data/data_binary_attributes_p1.json')
binary_features_p2 = pd.read_json('../input_data/data_binary_attributes_p2.json')


# Task 2.1
# specify matching attributes of content_p2
matching_attributes_p2 = []


# Task 2.2
# merge content (and ratings)
def get_nominal_feature_to_values(nominal_features):
    # get list of features and its expressions
    nominal_feature_to_values = dict()
    for feat in nominal_features.columns.values:
        for val in nominal_features[feat].unique():
            row = nominal_feature_to_values.setdefault(feat,list())
            if val != '':
                row.append(val)
    return nominal_feature_to_values

def get_nominal_features_binary(nominal_features):
    # create nominal_features_binary
    nominal_features_binary = dict()
    nominal_feature_to_values = get_nominal_feature_to_values(nominal_features)
    for row in nominal_features.iterrows():
        row_new = nominal_features_binary.setdefault(row[0],dict())
        for feat,values in nominal_feature_to_values.items():
            for val in values:
                col = feat+'##'+val
                if row[1][feat] == val:
                    row_new[col] = 1
                else:
                    row_new[col] = 0
            if sum([row_new[feat+'##'+val] for val in values]) == 0:
                for val in values:
                    col = feat+'##'+val
                    row_new[col] = np.nan
    nominal_features_binary= pd.DataFrame.from_dict(nominal_features_binary,orient='index')
    return nominal_features_binary

nominal_features_binary_p1 = get_nominal_features_binary(nominal_features_p1)
nominal_features_binary_p2 = get_nominal_features_binary(nominal_features_p2)

# join binary and nominal features
content_p1 = binary_features_p1.join(nominal_features_binary_p1)
content_p2 = binary_features_p2.join(nominal_features_binary_p2)
del nominal_features_binary_p1,nominal_features_binary_p2,binary_features_p1,binary_features_p2
del nominal_features_p1,nominal_features_p2

# filter content; discard items without ratings
content_p1 = content_p1[content_p1.index.isin(ratings_p1['id_items_p1'].values)]
content_p2 = content_p2[content_p2.index.isin(ratings_p2['id_items_p2'].values)]

# filter matching_pairs; discard items without ratings
matching_pairs = matching_pairs[matching_pairs['id_p1'].isin(ratings_p1['id_items_p1'].values)]
matching_pairs = matching_pairs[matching_pairs['id_p2'].isin(ratings_p2['id_items_p2'].values)]


# filter/remove rating data of items which are not contained in content data
ratings_p2 = ratings_p2[ratings_p2['id_items_p2'].isin(list(content_p2.index))]
ratings_p1 = ratings_p1[ratings_p1['id_items_p1'].isin(list(content_p1.index))]

# prepare merge of ratings; duplicate items get new id
def rename_duplicate_items_ratings(ratings,matching_pairs):
    # rename item_ids with its duplicate id from matching_pairs it is a duplicate
    ratings_copy = ratings.copy()
    key_item_ids = ratings.columns[1]
    key_item_ids_short = ''.join(key_item_ids.split('_items'))
    id_duplicates = list(set(matching_pairs[key_item_ids_short].values))
    for index,row in tqdm.tqdm(ratings.iterrows()):
        id_item = row[key_item_ids]
        if id_item in id_duplicates:
            id_match_item = 'match_i'+str(matching_pairs[matching_pairs[key_item_ids_short]==id_item].index.values[0])
            ratings_copy.at[index,key_item_ids]=id_match_item
    return ratings_copy

# rename columns for merging
ratings_p1_renamed_matches = rename_duplicate_items_ratings(ratings_p1,matching_pairs)
ratings_p2_renamed_matches = rename_duplicate_items_ratings(ratings_p2,matching_pairs)
ratings_p1_renamed_matches = ratings_p1_renamed_matches.rename(columns={"id_items_p1":"idItem","id_users_p1":"idUser"})
ratings_p2_renamed_matches = ratings_p2_renamed_matches.rename(columns={"id_items_p2":"idItem","id_users_p2":"idUser"})
ratings_p1 = ratings_p1.rename(columns={"id_items_p1":"idItem","id_users_p1":"idUser"})
ratings_p2 = ratings_p2.rename(columns={"id_items_p2":"idItem","id_users_p2":"idUser"})

# merge ratings
ratings_with_duplicate_detection = ratings_p2_renamed_matches.append(ratings_p1_renamed_matches,ignore_index=True, sort=False)
ratings_without_duplicate_detection = ratings_p2.append(ratings_p1,ignore_index=True, sort=False)

del ratings_p1,ratings_p2,ratings_p2_renamed_matches,ratings_p1_renamed_matches


# remove matching attributes from content_p2
for matching_attribute in matching_attributes_p2:
    content_p2.drop(matching_attribute,axis=1,inplace=True)

# merge content
def rename_duplicate_items_content(content,matching_pairs,portal):
    # rename item_ids with its duplicate id from matching_pairs it is a duplicate
    content_copy = content.copy()
    key_item_ids_short = 'id_'+portal
    id_duplicates = list(set(matching_pairs[key_item_ids_short].values))
    for index,row in tqdm.tqdm(content.iterrows()):
        id_item = index
        if id_item in id_duplicates:
            id_match_item = 'match_i'+str(matching_pairs[matching_pairs[key_item_ids_short]==id_item].index.values[0])
            content_copy.rename(index={id_item:id_match_item},inplace=True)
    return content_copy

content_p1_renamed_matches = rename_duplicate_items_content(content_p1,matching_pairs,'p1')
content_p2_renamed_matches = rename_duplicate_items_content(content_p2,matching_pairs,'p2')

content_with_duplicate_detection = content_p2_renamed_matches.join(content_p1_renamed_matches, how = 'outer', lsuffix='##p2',rsuffix='##p1')
content_without_duplicate_detection = content_p2.join(content_p1, how = 'outer', lsuffix='##p2',rsuffix='##p1')
del content_p1_renamed_matches,content_p2_renamed_matches,content_p1,content_p2

# Task 2.3
# imputation of missing values
imp = SimpleImputer(missing_values=np.nan, strategy='mean') #actual "mean" imputation, for more sophisticated imputation methods, one have to import another imputer from sklearn.impute
imp.fit(content_with_duplicate_detection)
content_with_duplicate_detection_and_imputation = pd.DataFrame(imp.transform(content_with_duplicate_detection),index=content_with_duplicate_detection.index,columns=content_with_duplicate_detection.columns)
content_with_duplicate_detection_and_imputation = content_with_duplicate_detection_and_imputation.round()

# fill nan values of other content tables with 0
content_with_duplicate_detection = content_with_duplicate_detection.fillna(0)
content_without_duplicate_detection = content_without_duplicate_detection.fillna(0)

# save data
content_with_duplicate_detection_and_imputation.to_json('../output_data/merged_content_with_duplicate_detection_and_imputation.json')
content_with_duplicate_detection.to_json('../output_data/merged_content_with_duplicate_detection.json')
ratings_with_duplicate_detection.to_json('../output_data/merged_ratings_with_duplicate_detection.json')

content_without_duplicate_detection.to_json('../output_data/merged_content_without_duplicate_detection.json')
ratings_without_duplicate_detection.to_json('../output_data/merged_ratings_without_duplicate_detection.json')