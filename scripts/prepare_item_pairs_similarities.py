import usaddress
import phonenumbers
import sys
import pandas as pd
import itertools
import pickle
import json

import textdistance # jaro_winkler, damerau_levenshtein, smith_waterman, hamming, levenshtein, strcmp95, mlipns etc.
from fuzzywuzzy import fuzz # ratio, partial_ratio, token_sort_ratio, token_set_ratio
import haversine

## Step 1 Duplicate Detection
## Task 1.1 data standardization and preparation

# Parameters
theshold_name_blocking = 0.9


with open('../input_data/data_key_attributes_p1.json','r') as infile:
    instances_p1 = json.load(infile)
with open('../input_data/data_key_attributes_p2.json','r') as infile:
    instances_p2 = json.load(infile)

key_attributes_p1=list(instances_p1[list(instances_p1.keys())[0]].keys())
key_attributes_p2=list(instances_p2[list(instances_p2.keys())[0]].keys())

key_attributes_to_unique_values_p1 = dict()
for attr_dict in instances_p1.values():
    for key_attr,value_attr in attr_dict.items():
        unique_values = key_attributes_to_unique_values_p1.setdefault(key_attr,set())
        unique_values.add(value_attr)
for key_attr,value_attr in key_attributes_to_unique_values_p1.items():
    key_attributes_to_unique_values_p1[key_attr]=list(value_attr)

key_attributes_to_unique_values_p2 = dict()
for attr_dict in instances_p2.values():
    for key_attr,value_attr in attr_dict.items():
        unique_values = key_attributes_to_unique_values_p2.setdefault(key_attr,set())
        unique_values.add(value_attr)
for key_attr,value_attr in key_attributes_to_unique_values_p2.items():
    key_attributes_to_unique_values_p2[key_attr]=list(value_attr)

# prepare key_attributes for similarity measures
# standardize zip
for id,instance in instances_p1.items():
    if "-" in instance['zip']:
        instance['zip'] = instance['zip'].split("-")[0]
    instances_p1[id] = instance

# standardize gps, phone, address
for id,instance in instances_p1.items():
    # convert latitude longitude to tuple
    try:
        instance['gps']=(float(instance['latitude']),float(instance['longitude']))
    except Exception:
        instance['gps'] = None
    # get phone_national_number
    try:
        instance['phone_national_number'] = str(phonenumbers.parse(instance['phone'],"US").national_number)
    except Exception:
        exc_type, value, traceback = sys.exc_info()
        assert exc_type == phonenumbers.phonenumberutil.NumberParseException
        instance['phone_national_number'] = None
    # get street_AdressNumber
    try:
        parsed_address = usaddress.tag(instance['street'])[0]
        instance['street_AddressNumber'] = parsed_address.setdefault('AddressNumber',None)
    except Exception:
        exc_type, value, traceback = sys.exc_info()
        assert exc_type == usaddress.RepeatedLabelError
        instance['street_AddressNumber']=None
    instances_p1[id] = instance

for id,instance in instances_p2.items():
    # convert latitude longitude to tuple
    try:
        instance['gps']=(float(instance['latitude']),float(instance['longitude']))
    except Exception:
        instance['gps'] = None
    # get phone_national_number
    try:
        instance['phone_national_number'] = str(phonenumbers.parse(instance['phone'],"US").national_number)
    except Exception:
        exc_type, value, traceback = sys.exc_info()
        assert exc_type == phonenumbers.phonenumberutil.NumberParseException
        instance['phone_national_number'] = None
    # get street_AdressNumber
    try:
        parsed_address = usaddress.tag(instance['street'])[0]
        instance['street_AddressNumber'] = parsed_address.setdefault('AddressNumber',None)
    except Exception:
        exc_type, value, traceback = sys.exc_info()
        assert exc_type == usaddress.RepeatedLabelError
        instance['street_AddressNumber']=None
    instances_p2[id] = instance

key_attributes_for_sim=["name","zip","gps","street",'street_AddressNumber',"phone_national_number","neighborhood"]

# create dict with all string similarity measures
edit_sim_dict = {'jaro_winkler':textdistance.jaro_winkler.normalized_similarity, 
                   'damerau_levenshtein':textdistance.damerau_levenshtein.normalized_similarity,
                   'smith_waterman':textdistance.smith_waterman.normalized_similarity,
                   'hamming':textdistance.hamming.normalized_similarity,
                   'strcmp95':textdistance.strcmp95.normalized_similarity,
                   'mlipns':textdistance.mlipns.normalized_similarity,
                   'needleman_wunsch':textdistance.needleman_wunsch.normalized_similarity,
                   'gotoh':textdistance.gotoh.normalized_similarity
                   }
token_sim_dict = {'jaccard':textdistance.jaccard.normalized_similarity,
                  'sorensen':textdistance.sorensen.normalized_similarity,
                  'tversky':textdistance.tversky.normalized_similarity,
                  'tanimoto':textdistance.tanimoto.normalized_similarity,
                  'monge_elkan':textdistance.monge_elkan.normalized_similarity,
                  'bag':textdistance.bag.normalized_similarity
                  }
sequence_sim_dict = {'lcsseq':textdistance.lcsseq.normalized_similarity,
                     'lcsstr':textdistance.lcsstr.normalized_similarity,
                     'ratcliff_obershelp':textdistance.ratcliff_obershelp.normalized_similarity
                     }
fuzzy_sim_dict = {'partial_ratio': lambda n1,n2 : fuzz.partial_ratio(n1,n2)/100,
                  'partial_token_set_ratio': lambda n1,n2 : fuzz.partial_token_set_ratio(n1,n2)/100,
                  'partial_token_sort_ratio': lambda n1,n2 : fuzz.partial_token_sort_ratio(n1,n2)/100,
                  'ratio': lambda n1,n2 : fuzz.ratio(n1,n2)/100,
                  'token_set_ratio': lambda n1,n2 : fuzz.token_set_ratio(n1,n2)/100,
                  'token_sort_ratio': lambda n1,n2 : fuzz.token_sort_ratio(n1,n2)/100
                  }
sim_measures = {}
for k,v in edit_sim_dict.items(): sim_measures[k]=v
for k,v in token_sim_dict.items(): sim_measures[k]=v
for k,v in sequence_sim_dict.items(): sim_measures[k]=v
for k,v in fuzzy_sim_dict.items(): sim_measures[k]=v

sim_measures['haversine'] = lambda x,y : 1-haversine.haversine(x,y)

#create list of id pairs
id_pairs = list(itertools.product(list(instances_p2.keys()),list(instances_p1.keys())))

# create similarity scores for each pair
key_attribute_to_sim_measure_names = {'name':['jaro_winkler','damerau_levenshtein',
                                         'lcsstr',
                                         'partial_ratio','partial_token_sort_ratio'],
                                    "zip":['jaro_winkler','hamming','lcsstr'],
                                    "gps":['haversine'],
                                    "street":['jaro_winkler','damerau_levenshtein',
                                         'lcsstr',
                                         'partial_ratio','partial_token_sort_ratio'],
                                    "street_AddressNumber":['jaro_winkler','hamming','lcsstr'],
                                    "phone_national_number":['hamming'],
                                    "neighborhood":['partial_ratio']}
pairs_dict = dict()
count = 0
# compute similarities for key attributes
for i,(id_p2,id_p1) in enumerate(id_pairs):
    instance_p2 = instances_p2[id_p2]
    instance_p1 = instances_p1[id_p1]
    if bool(instance_p2['gps']) and bool(instance_p1['gps']) and \
    sim_measures['haversine'](instance_p2['gps'],instance_p1['gps']) > 0: # blocking on gps
        if any(sim_measures[sim_measure_name](instance_p2['name'],instance_p1['name'])>theshold_name_blocking \
               for sim_measure_name in key_attribute_to_sim_measure_names['name']): # blocking on name
            count += 1
            id_pair = pairs_dict.setdefault("id_pair",list())
            id_pair.append(i)
            for key_attribute,sim_measure_names in key_attribute_to_sim_measure_names.items():
                value_p2 = instance_p2[key_attribute]
                value_p1 = instance_p1[key_attribute]
                for sim_measure_name in sim_measure_names:
                    sim_measure = sim_measures[sim_measure_name]
                    try:
                        sim_value = sim_measure(value_p2,value_p1)
                    except:
                        sim_value = None
                    name_column = key_attribute+'_'+sim_measure_name
                    sim_values = pairs_dict.setdefault(name_column,list())
                    sim_values.append(sim_value)

pairs = pd.DataFrame.from_dict(pairs_dict)
pairs = pairs.set_index('id_pair')

pairs.to_pickle("./pairs.pkl")
pickle.dump(id_pairs,open("id_pairs.pkl",'wb'))