import csv
import numpy as np
import copy
import matplotlib.pyplot as plt

# np.set_printoptions(threshold=10000)


def get_dataset():
	DATASET = 'questionnaire'

	data = np.genfromtxt('data/%s.csv' %DATASET, delimiter=',', filling_values=99.99)

	with open('data/%s.csv' %DATASET) as f:
		reader = csv.reader(f)
		for row in reader:
			first_row = row
			break

	smoking_fieldnames = ['SMQ020', 'SMD030', 'SMQ040', 'SMQ050Q', 'SMQ050U', 'SMD055', 'SMD057', 'SMQ078', 'SMD641', 'SMD650', 'SMD093', 'SMDUPCA', 'SMD100BR', 'SMD100FL', 'SMD100MN', 'SMD100LN', 'SMD100TR', 'SMD100NI', 'SMD100CO', 'SMQ621', 'SMD630', 'SMQ661', 'SMQ665A', 'SMQ665B', 'SMQ665C', 'SMQ665D', 'SMQ670', 'SMQ848', 'SMQ852Q', 'SMQ852U', 'SMAQUEX2', 'SMD460', 'SMD470', 'SMD480', 'SMQ856', 'SMQ858', 'SMQ860', 'SMQ862', 'SMQ866', 'SMQ868', 'SMQ870', 'SMQ872', 'SMQ874', 'SMQ876', 'SMQ878', 'SMQ880', 'SMAQUEX.x', 'SMQ681', 'SMQ690A', 'SMQ710', 'SMQ720', 'SMQ725', 'SMQ690B', 'SMQ740', 'SMQ690C', 'SMQ770', 'SMQ690G', 'SMQ845', 'SMQ690H', 'SMQ849', 'SMQ851', 'SMQ690D', 'SMQ800', 'SMQ690E', 'SMQ817', 'SMQ690I', 'SMQ857', 'SMQ690J', 'SMQ861', 'SMQ863', 'SMQ690F', 'SMQ830', 'SMQ840']

	first_smoking_index = first_row.index('SMQ020')
	last_smoking_index = first_row.index('SMQ840')

	data_size, num_features = data.shape

	validation_set_size = int(round(data_size*.8))
	training_set_size = int(round(validation_set_size*.8))

	training_data = data[0:training_set_size]

	validation_data = data[training_set_size:validation_set_size]
	test_data = data[validation_set_size:data_size]


	#SMQ040 = 'Do you now smoke cigarettes?'
	Y_index = first_row.index('SMQ040')

	Y_train = _norm_Y_data(training_data[:, Y_index])
	Y_val = _norm_Y_data(validation_data[:, Y_index])
	Y_test = _norm_Y_data(test_data[:, Y_index])


	train_1, smoking_data, train_2 = np.split(training_data, [first_smoking_index, last_smoking_index], axis=1)
	val_1, smoking_data, val_2 = np.split(validation_data, [first_smoking_index, last_smoking_index], axis=1)
	test_1, smoking_data, test_2 = np.split(test_data, [first_smoking_index, last_smoking_index], axis=1)

	X_train = np.concatenate((train_1, train_2), axis=1)
	X_val = np.concatenate((val_1, val_2), axis=1)
	X_test = np.concatenate((test_1, test_2), axis=1)

	X_train_norm, Y_train_norm = _get_nonzero_rows(X_train, Y_train)
	X_val_norm, Y_val_norm = _get_nonzero_rows(X_val, Y_val)
	X_test_norm, Y_test_norm = _get_nonzero_rows(X_test, Y_test)

	return X_train_norm, Y_train_norm, X_val_norm, Y_val_norm, X_test_norm, Y_test_norm


def _norm_Y_data(Y):
	new_Y = copy.deepcopy(Y)

	for i in range(len(Y)):
		if (Y[i] == 1) or (Y[i] == 2):
			new_Y[i] = 1
		elif Y[i] == 0:
			new_Y[i] = 0
		else:
			new_Y[i] = -1

	return new_Y


def _get_nonzero_rows(X, Y):
	new_X = copy.deepcopy(X)
	new_Y = copy.deepcopy(Y)

	indices = []
	for i in range(len(Y)):
		if Y[i] == 0:
			indices.append(i)

	new_Y = np.delete(new_Y, indices)
	new_X = np.delete(new_X, indices, 0)

	return new_X, new_Y


def get_traindata():
	return get_dataset()[0], get_dataset()[1]


def get_valdata():
	return get_dataset()[2], get_dataset()[3]


def get_testdata(): 
	return get_dataset()[4], get_dataset()[5]



def clean_column(spar,X,Y): 
    #X, Y = get_traindata()
    #print ("X size :",X.shape)
    #print ("Y size :",Y.shape)
    count = 0
    i= 0
    L= len(X[0])
    while i < L:
        #print (i)
        #print (X.shape)
        n=1.0 - np.count_nonzero(X[:,i] == 99.99) *1.0 / len(X)
        #print("hello")
        if n <= spar:
            
            X= np.delete(X,i,1)
            L -= 1
        else:
            count+=1
            i += 1
            
    return count , X, Y

def clean_row(spar,X,Y):
    count = 0
    i= 0
    L= len(X)
    while i < L:
        n=1.0 - np.count_nonzero(X[i] == 99.99) *1.0 / len(X[i])
        
        if n <= spar:
            
            X= np.delete(X,i,0)
            Y= np.delete(Y,i)
            L -= 1
        else:
            count+=1
            i += 1
            
    return count , X, Y

def clean_data(spar,X,Y):
    print("input data: ",X.shape)
    _,X1,Y1 = clean_column(spar,X,Y)
    print("column cleaned: ",X1.shape)
    _,X2,Y2 = clean_row(spar,X1,Y1)
    print("final cleaned: ",X2.shape)
    return  X2 , Y2
    
 
X, Y = get_traindata() 
clean_data(.99,X,Y)


def spars_plot():        
    X, Y = get_traindata()        
    xaxis=np.arange(0.0, 1.0, 0.01)
    yaxis=[clean_column(i,X,Y)[0] for i in np.arange(0.0, 1.0, 0.01)]
    
    plt.plot(xaxis , yaxis)
    plt.ylabel('# of columns')
    plt.xlabel('sparsity')
    plt.show()

#spars_plot()






