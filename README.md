# Extending Item Content Data

## Dependencies
Python 3.6 or higher  
Package `usaddress`  
Package `phonenumbers`  
Package `textdistance`  
Package `fuzzywuzzy`  
Package `haversine`  

Installation via `pip install <package>`

## Input Data

Example data can be found in the folder `input_data`. The procedure is self-contained, i.e. it should be run with this data.
In order to enable the application of the procedure to a different data set, following files have to be provided:
- data_binary_attributes_p1.json
- data_binary_attributes_p2.json
- data_key_attributes_p1.json
- data_key_attributes_p2.json
- data_nominal_attributes_p1.json
- data_nominal_attributes_p2.json
- data_ratings_p1.json
- data_ratings_p2.json

## Instructions

To run the procedure, the following tasks have to be perfomed:
1. Execute script `prepare_item_pairs_similarities.py`
2. Execute script `unsupervised_duplicate_detection.py`
3. Execute script `resolve_multi_linkages.py`
4. Execute script `prepare_data_for_labeling.py`
5. Label data in file `unlabeled_data.xlsx` and rename file to `labeled_data.xlsx`
6. Execute script `evaluate_duplicate_detection.py`
7. Execute script `prepare_data_for_RS.py`

## Output Data

After running the procedure, the following files are created in the `output_data` folder:
- `merged_ratings_with_duplicate_detection.json`
- `merged_content_with_duplicate_detection.json`
- `merged_ratings_without_duplicate_detection.json`
- `merged_content_without_duplicate_detection.json`
