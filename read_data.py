import csv
import numpy as np

DATASET = 'questionnaire'

data = np.genfromtxt('data/%s.csv' %DATASET, delimiter=',', filling_values=0)

with open('data/%s.csv' %DATASET) as f:
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

#SMQ040 = 'Do you now smoke cigarettes?'
Y_index = first_row.index('SMQ040')

Y_train = training_data[:, Y_index]
Y_val = validation_data[:, Y_index]
Y_test = test_data[:, Y_index]


train_1, smoking_data, train_2 = np.split(training_data, [first_smoking_index, last_smoking_index], axis=1)
val_1, smoking_data, val_2 = np.split(validation_data, [first_smoking_index, last_smoking_index], axis=1)
test_1, smoking_data, test_2 = np.split(test_data, [first_smoking_index, last_smoking_index], axis=1)

X_train = np.concatenate((train_1, train_2), axis=1)
X_val = np.concatenate((val_1, val_2), axis=1)
X_test = np.concatenate((test_1, test_2), axis=1)
