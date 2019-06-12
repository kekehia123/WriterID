import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Here, I
parser = argparse.ArgumentParser(description='Arguments for preprocessing.')
parser.add_argument('--NClass', type=int, default=10, help='Number of class (10/107)')
parser.add_argument('--LenSample', type=int, default=10, help='Length of samples')
parser.add_argument('--NSample', type=int, default=5, help='Number of samples')
parser.add_argument('--FlexibleNSample', type=bool, default=True, help='Whether to decide N samples according to the char length')
parser.add_argument('--BadThreshold', type=int, default=5, help=' ')
parser.add_argument('--FileName', type=str, default='SampleRHS_singleChar_len10_10', help=' ')

args = parser.parse_args()
NClass = args.NClass
NumOfSample = args.NSample
LenOfSample = args.LenSample
BadCharThr = args.BadThreshold
FlexibleNSample = args.FlexibleNSample
FileName = args.FileName

for arg in vars(args):
    print(arg, getattr(args, arg))

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

def CreateRHS_singleChar(data_newAxis, LenOfSample, NumOfSample):
    data_sampled = []

    if len(data_newAxis[0]) <= LenOfSample + NumOfSample:
        n_sample = 3
        for i in range(n_sample):
            start_point = i
            one_sample = []
            for point in range(len(data_newAxis[0])-n_sample-1):
                one_sample.append([data_newAxis[0][start_point + point + 1] - data_newAxis[0][start_point + point],
                                   data_newAxis[1][start_point + point + 1] - data_newAxis[1][start_point + point]])
            data_sampled.append(one_sample)
    else:
        if FlexibleNSample:
            n_sample = max(NumOfSample, len(data_newAxis[0]) // NumOfSample)
        else:
            n_sample = NumOfSample
        for i in range(n_sample):
            start_point = np.random.randint(len(data_newAxis[0]) - LenOfSample)
            one_sample = []
            for point in range(LenOfSample):
                one_sample.append([data_newAxis[0][start_point + point + 1] - data_newAxis[0][start_point + point],
                                   data_newAxis[1][start_point + point + 1] - data_newAxis[1][start_point + point]])
            data_sampled.append(one_sample)
    return data_sampled, n_sample


def interpChar(data_newAxis, threshold):
    num_points = len(data_newAxis[0])
    # If threshold is 60, then those chars with points number less than 60 will be
    # interped by 2 times.
    times_interp = int(np.ceil(threshold / num_points))
    Scatter_interped_x, Scatter_interped_y = [], []
    for i in range(num_points):
        new_point_x = data_newAxis[0][i] / times_interp
        new_point_y = data_newAxis[1][i] / times_interp
        for j in range(times_interp):
            Scatter_interped_x.append(new_point_x)
            Scatter_interped_y.append(new_point_y)
    data_newAxis_interped = [Scatter_interped_x, Scatter_interped_y]
    return data_newAxis_interped


def ShowRHS(SampleRHS):
    Train_RHS_Sample = SampleRHS['Train_RHS_Sample']

    plt.interactive(False)
    plt.figure()
    for i in range(20):
        sample_sel = np.random.randint(len(Train_RHS_Sample))
        print(sample_sel)
        plt.subplot(4, 5, i+1)
        Scatter_x, Scatter_y = [], []
        crt_position_x, crt_position_y = 100, 100
        for j in range(len(Train_RHS_Sample[sample_sel])):
            Scatter_x.append(Train_RHS_Sample[sample_sel][j][0] + crt_position_x)
            Scatter_y.append(Train_RHS_Sample[sample_sel][j][1] + crt_position_y)
            crt_position_x += Train_RHS_Sample[sample_sel][j][0]
            crt_position_y += Train_RHS_Sample[sample_sel][j][1]
        plt.scatter(Scatter_x, Scatter_y, marker='o', color='r', s=1)
    plt.show()

Train_RHS_Sample, Train_RHS_Label_Sample, Validation_RHS_Sample, Validation_RHS_Label_Sample = [], [], [], []
n_samples_train, n_samples_val = [], []
charLen_train = []
num_lessLen = 0
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
        if len(data_newAxis[0]) < LenOfSample:
            num_lessLen += 1
        data_sampled, n_sample = CreateRHS_singleChar(
            data_newAxis, LenOfSample, NumOfSample)
        for k in range(len(data_sampled)):
            Train_RHS_Sample.append(data_sampled[k])
            Train_RHS_Label_Sample.append(file_sub)
        n_samples_train.append(n_sample)
print('Number of training samples smaller that LenOfSample:', num_lessLen)

charLen_val = []
num_lessLen = 0
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
        if len(data_newAxis[0]) < LenOfSample:
            num_lessLen += 1
        data_sampled, n_sample = CreateRHS_singleChar(data_newAxis, LenOfSample, NumOfSample)
        for k in range(len(data_sampled)):
            Validation_RHS_Sample.append(data_sampled[k])
            Validation_RHS_Label_Sample.append(file_sub)
        n_samples_val.append(n_sample)
print('Number of validation samples smaller that LenOfSample:', num_lessLen)

SampleRHS = dict(Train_RHS_Sample=Train_RHS_Sample, Train_RHS_Label_Sample=Train_RHS_Label_Sample,
                 Validation_RHS_Sample=Validation_RHS_Sample, Validation_RHS_Label_Sample=Validation_RHS_Label_Sample,
                 Train_NSamples=n_samples_train, Validation_NSamples=n_samples_val)


f = open(result_file, 'w')
f.write(str(SampleRHS))
f.close()
