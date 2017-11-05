import csv
import numpy as np

data = np.genfromtxt('data/questionnaire.csv', delimiter=',', filling_values=0)

with open('data/questionnaire.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		first_row = row
		break

smoking_fieldnames = ['SMQ020', 'SMD030', 'SMQ040', 'SMQ050Q', 'SMQ050U', 'SMD055', 'SMD057', 'SMQ078', 'SMD641', 'SMD650', 'SMD093', 'SMDUPCA', 'SMD100BR', 'SMD100FL', 'SMD100MN', 'SMD100LN', 'SMD100TR', 'SMD100NI', 'SMD100CO', 'SMQ621', 'SMD630', 'SMQ661', 'SMQ665A', 'SMQ665B', 'SMQ665C', 'SMQ665D', 'SMQ670', 'SMQ848', 'SMQ852Q', 'SMQ852U', 'SMAQUEX2', 'SMD460', 'SMD470', 'SMD480', 'SMQ856', 'SMQ858', 'SMQ860', 'SMQ862', 'SMQ866', 'SMQ868', 'SMQ870', 'SMQ872', 'SMQ874', 'SMQ876', 'SMQ878', 'SMQ880', 'SMAQUEX.x', 'SMQ681', 'SMQ690A', 'SMQ710', 'SMQ720', 'SMQ725', 'SMQ690B', 'SMQ740', 'SMQ690C', 'SMQ770', 'SMQ690G', 'SMQ845', 'SMQ690H', 'SMQ849', 'SMQ851', 'SMQ690D', 'SMQ800', 'SMQ690E', 'SMQ817', 'SMQ690I', 'SMQ857', 'SMQ690J', 'SMQ861', 'SMQ863', 'SMQ690F', 'SMQ830', 'SMQ840']

first_smoking_index = first_row.index('SMQ020')
last_smoking_index = first_row.index('SMQ840')

data_size, num_features = data.shape

validation_set_size = round(data_size*.8)
training_set_size = round(validation_set_size*.8)

training_data = data[0:training_set_size]
validation_data = data[training_set_size:validation_set_size]
test_data = data[validation_set_size:data_size]


'''
smoking_data is just the columns associated with smoking tobacco
nonsmoking_data is the remainder of the data with smoking columns removed
'''

nonsmoking_1, smoking_data, nonsmoking_2 = np.split(training_data, [first_smoking_index, last_smoking_index], axis=1)

nonsmoking_data = np.concatenate((nonsmoking_1, nonsmoking_2), axis=1)
