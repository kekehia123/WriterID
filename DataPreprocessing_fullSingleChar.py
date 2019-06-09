import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Here, I
parser = argparse.ArgumentParser(description='Arguments for preprocessing.')
parser.add_argument('--NClass', type=int, default=107, help='Number of class (10/107)')
parser.add_argument('--BadThreshold', type=int, default=5, help=' ')
parser.add_argument('--FileName', type=str, default='SampleRHS_fullSingleChar_107', help=' ')

args = parser.parse_args()
NClass = args.NClass
BadCharThr = args.BadThreshold
FileName = args.FileName

# parameters setting
data_root = '/data/data_nTGG0ILaS1qu/WriterID'

assert NClass in [10, 107]
if NClass == 10:
    path = os.path.join(data_root, 'Data10')
else:
    path = os.path.join(data_root, 'Data')
result_file = os.path.join('prepared_data', FileName+'.txt')

train_path = os.path.join(path, 'Train')
val_path = os.path.join(path, 'Validation')

train_files = os.listdir(train_path)
val_files = os.listdir(val_path)

def SwitchAxes_singleChar(Word):
    Scatter_x, Scatter_y = [], []
    for stroke in range(len(Word)):
        # print(stroke)
        for point in range(len(Word[stroke])):
            # Width and Height of the input window are 600
            Scatter_x.append(Word[stroke][point][0])
            Scatter_y.append(600 - Word[stroke][point][1])
    data_newAxis = [Scatter_x, Scatter_y]
    return data_newAxis

Train_RHS_Sample, Train_RHS_Label_Sample, Validation_RHS_Sample, Validation_RHS_Label_Sample = [], [], [], []
n_samples_train, n_samples_val = [], []
charLen_train = []
#num_lessLen = 0
for i in range(len(train_files)):
    data_sub = np.load(os.path.join(train_path, train_files[i]))
    file_sub = train_files[i].split('.')[0]
    for j in range(len(data_sub)):
        Word = data_sub[j]
        data_newAxis = SwitchAxes_singleChar(Word)
        # If the char length < 5, it is probably problematic.
        if len(data_newAxis[0]) < BadCharThr:
            continue
        charLen_train.append(len(data_newAxis[0]))
        data_sampled = []
        for k in range(len(data_newAxis[0])):
            data_sampled.append([data_newAxis[0][k], data_newAxis[1][k]])
        Train_RHS_Sample.append(data_sampled)
        Train_RHS_Label_Sample.append(file_sub)
        n_samples_train.append(1)
#print('Number of training samples smaller that LenOfSample:', num_lessLen)

charLen_val = []
#num_lessLen = 0
for i in range(len(val_files)):
    data_sub = np.load(os.path.join(val_path, val_files[i]))
    file_sub = val_files[i].split('.')[0]
    for j in range(len(data_sub)):
        Word = data_sub[j]
        data_newAxis = SwitchAxes_singleChar(Word)
        # If the char length < 5, it is probably problematic.
        if len(data_newAxis[0]) < BadCharThr:
            continue
        charLen_val.append(len(data_newAxis[0]))
        data_sampled = []
        for k in range(len(data_newAxis[0])):
            data_sampled.append([data_newAxis[0][k], data_newAxis[1][k]])
        Validation_RHS_Sample.append(data_sampled)
        Validation_RHS_Label_Sample.append(file_sub)
        n_samples_val.append(1)
#print('Number of validation samples smaller that LenOfSample:', num_lessLen)

SampleRHS = dict(Train_RHS_Sample=Train_RHS_Sample, Train_RHS_Label_Sample=Train_RHS_Label_Sample,
                 Validation_RHS_Sample=Validation_RHS_Sample, Validation_RHS_Label_Sample=Validation_RHS_Label_Sample,
                 Train_NSamples=n_samples_train, Validation_NSamples=n_samples_val)


f = open(result_file, 'w')
f.write(str(SampleRHS))
f.close()
