import numpy as np

def isFloat(n):
    try:
        float(n)
        return true
    except:
        return false

def lineCounter(filename):
    with open(filename, 'r') as fp:
        lines = len(fp.readlines())
        return lines


def parse_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
    
    # Split the content into individual records based on new lines
    records = content.strip().split('\n')
    
    # Initialize a dictionary to hold the values of kf_, dset, etc.
    data = {
        'first_numbers': [],
        'node': [],
        'dset': [],
        'alpha': [],
        'gamma': [],
        'kf': [],
        'kr': [],
        'bl': []
    }
    
    for record in records:
        parts = record.split()
        try:
            # Extract the first value and add to the list
            first_value = float(parts[0])
            data['first_numbers'].append(first_value)
            
            # Extract other values by their labels
            for part in parts[1:]:
                if part.startswith('node_'):
                    data['node'].append(part.split('_')[1])
                elif part.startswith('dset_'):
                    data['dset'].append(part.split('_')[1])
                elif part.startswith('alpha_'):
                    data['alpha'].append(float(part.split('_')[1]))
                elif part.startswith('gamma_'):
                    data['gamma'].append(float(part.split('_')[1]))
                elif part.startswith('kf_'):
                    data['kf'].append(float(part.split('_')[1]))
                elif part.startswith('kr_'):
                    data['kr'].append(float(part.split('_')[1]))
                elif part.startswith('bl_'):
                    data['bl'].append(float(part.split('_')[1]))
        except (ValueError, IndexError) as e:
            print(f"Error processing record '{record}': {e}")
    
    return data

# Usage
filename = 'corrScores.dat'

data = parse_file(filename)
corrsList = data['first_numbers']

corrs = np.zeros(len(corrsList))
for i in range(len(corrsList)):
    corrs[i] = float(corrsList[i])

print(np.max(corrs))

maxIndex = np.argmax(corrs)
print(corrs[maxIndex])

node_values = data['node']
dset_values = data['dset']
alpha_values = data['alpha']
gamma_values = data['gamma']
kf_values = data['kf']
kr_values = data['kr']
bl_values = data['bl']

i = maxIndex
print(corrs[i], "node:", node_values[i], "dset:", dset_values[i], "alpha:", alpha_values[i], "gamma:", gamma_values[i], "kf:", kf_values[i], "kr:", kr_values[i], "bl", bl_values[i])

counter = 0
for i in range(len(corrs)):
    if corrs[i] >= .25:
        print(corrs[i], "node:", node_values[i], "dset:", dset_values[i], "alpha:", alpha_values[i], "gamma:", gamma_values[i], "kf:", kf_values[i], "kr:", kr_values[i], "bl", bl_values[i])
        counter = counter + 1

print("# with cor score greater than 0.3", counter)

lineNumber = lineCounter(filename)
print("Number of lines:", lineNumber)
   
