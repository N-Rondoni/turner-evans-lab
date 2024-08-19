import csv
from collections import defaultdict

def parse_corr_coef(file_path, algorithm_name):
    corr_data = defaultdict(list)
    
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            if row['algo_algorithm'] == algorithm_name and row['label_label'] == 'corr':
                dataset_id = row['dset__id']
                correlation_value = float(row['label_value'])
                corr_data[dataset_id].append(correlation_value)
    
    return dict(corr_data)

# Example usage:
# file_path = 'your_file_path.csv'
# algorithm_name = 'oopsi'
# corr_coefficients = parse_corr_coefficients(file_path, algorithm_name)
# print(corr_coefficients)

