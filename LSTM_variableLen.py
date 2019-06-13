import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import time
import argparse
import os
from io_utils import load_data, LoadinRHS, convertLabel2Num

parser = argparse.ArgumentParser(description='Arguments for preprocessing.')
parser.add_argument('--NClass', type=int, default=10, help='Number of class (10/107)')
parser.add_argument('--Epoch', type=int, default=200, help='Number of epochs')
parser.add_argument('--MaxTol', type=int, default=30, help='Tolerance of epochs with no test loss decrease')
parser.add_argument('--LR', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--Gamma', type=float, default=0.6, help='Decay rate in learning')
parser.add_argument('--ResumeModel', type=bool, default=False, help='Whether resume existing model')
#parser.add_argument('--LenSample', type=int, default=20, help='Length of samples')
#parser.add_argument('--NSample', type=int, default=5, help='Number of samples')
parser.add_argument('--BatchSize', type=int, default=200, help='Number of samples')
parser.add_argument('--Vote', type=bool, default=True, help='Number of samples')
parser.add_argument('--ModelName', type=str, default='standard_10', help=' ')
parser.add_argument('--InputFile', type=str, default='SampleRHS_singleChar_len20_10', help=' ')
parser.add_argument('--ModelType', type=str, default='standard', help='single direction or bidirectional (whether use mean pooling)')
parser.add_argument('--LSTMLayer', type=int, default=1, help='layer of LSTM')
parser.add_argument('--LSTMHiddenDim', type=int, default=128, help='single direction or bidirection')

args = parser.parse_args()
NumOfCategory = args.NClass
EPOCH = args.Epoch
maxTol = args.MaxTol
learning_rate = args.LR
gamma = args.Gamma
resume_model = args.ResumeModel
#NumofSamples = args.NSample
#LenOfSample = args.LenSample
BATCH_SIZE = args.BatchSize
vote = args.Vote
model_name = args.ModelName
input_file_name = args.InputFile
model_type = args.ModelType
LSTM_layer = args.LSTMLayer
LSTM_hiddenDim = args.LSTMHiddenDim

assert model_type in ['standard', 'bidirectional', 'mean_pooling']
save_dir = os.path.join('models', model_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

f = open(os.path.join(save_dir, 'configs.txt'), 'w')
for arg in vars(args):
    f.writelines(arg + ' ' + str(getattr(args, arg)) + '\n')
    print(arg, getattr(args, arg))
f.close()

assert NumOfCategory in [10, 107]
input_file = os.path.join('prepared_data', input_file_name+'.txt')

use_gpu = torch.cuda.is_available()

# 定义了一个单向LSTM网络，隐藏层为100个节点，通过线性分类器分为10类
class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTM, self).__init__()
        self.n_layer = n_layer
        # dimensions of the input feature
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        if model_type == 'standard':
            self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer,
                                batch_first=True)
            # self.out = nn.Linear(hidden_dim, n_class)
            self.classifier = nn.Linear(hidden_dim, n_class)
        elif model_type == 'bidirectional':
            self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer,
                                batch_first=True, bidirectional=True)
            # self.out = nn.Linear(hidden_dim, n_class)
            self.classifier = nn.Linear(hidden_dim*2, n_class)
        elif model_type == 'mean_pooling':
            self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer,
                                batch_first=True, bidirectional=True)
            self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x, x_lengths):
        self.lstm.flatten_parameters()
        x = pack_padded_sequence(x, x_lengths.cpu().numpy(), batch_first = True)
        out, (ht, ct) = self.lstm(x)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = out[:, -1, :]
        if model_type == 'mean_pooling':
            out = 0.5 * (out[:, :self.hidden_dim] + out[:, self.hidden_dim:])
        #ht = ht.view(-1, ht.size()[-1])
        out = self.classifier(out)
        return out

# LSTM会返回每一个RHS sample的分类结果，由于每个学生有NumofSamples个sample，因此可以进行投票，返回最终判断
def Vote(pred, n_samples):
    # print(len(pred))
    # print(type(pred))
    pred_return = []
    current_ind = 0
    for i in range(len(n_samples)):
        # print(i)
        temp = pred[current_ind: current_ind + n_samples[i]]
        current_ind = current_ind + n_samples[i]
        counts = np.bincount(temp)
        # print(temp)
        # print(counts)
        # 返回众数
        ind = np.argmax(counts)
        # print(ind)
        list_ind = ind * np.ones((1, n_samples[i]))
        #pred_return.extend(list_ind.tolist()[0])
        pred_return.append(ind)
    return pred_return

def train(model, loader, criterion, optimizer, scheduler, epoch):
    print(' ')
    #print('EPOCH: ' + str(epoch))
    # Each epoch has a training and validation phase
    scheduler.step()
    model.train(True)  # Set model to training mode

    running_corrects = 0
    AllTrain = 0
    roundnum = 0

    for data in loader:
        roundnum += 1
        # print(roundnum)
        inputs, seq_lengths, labels = data

        # Sort the samples in descending order
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        inputs = inputs[perm_idx]
        labels = labels[perm_idx]

        #inputs = inputs.view(-1, LenOfSample, 2)
        if use_gpu:
            inputs = Variable(inputs.cuda().type(torch.cuda.FloatTensor))
            seq_lengths = Variable(seq_lengths.cuda().type(torch.cuda.LongTensor))
            labels = Variable(labels.cuda().long())
        else:
            inputs, labels = Variable(inputs).type(torch.FloatTensor), Variable(labels.long())
            seq_lengths = Variable(seq_lengths).type(torch.LongTensor)

        optimizer.zero_grad()

        # forward
        outputs = model(inputs, seq_lengths)
        _, preds = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Analyze the accuracy
        for i in range(len(preds)):
            AllTrain += 1
            if preds[i] == labels.data[i]:
                running_corrects += 1

    epoch_acc = running_corrects / LenTrain
    print('Epoch %d, training accuracy: %f' % (epoch, epoch_acc))
    loss = loss.data.cpu().tolist()
    return model, epoch_acc, loss

def test(model, loader, criterion, epoch, vote, n_samples, Validation_Label_List):
    model.train(False)  # Set model to evaluate mode

    running_corrects_single = 0
    AllTrain = 0
    roundnum = 0

    accum = []
    for data in loader:
        roundnum += 1
        # print(roundnum)
        inputs, seq_lengths, labels = data

        # Sort the samples in descending order
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        inputs = inputs[perm_idx]
        labels = labels[perm_idx]

        #inputs = inputs.view(-1, LenOfSample, 2)
        if use_gpu:
            inputs = Variable(inputs.cuda().type(torch.cuda.FloatTensor))
            seq_lengths = Variable(seq_lengths.cuda().type(torch.cuda.LongTensor))
            labels = Variable(labels.cuda().long())
        else:
            inputs, labels = Variable(inputs).type(torch.FloatTensor), Variable(labels.long())
            seq_lengths = Variable(seq_lengths).type(torch.LongTensor)

        # forward
        outputs = model(inputs, seq_lengths)
        _, preds = torch.max(outputs.data, 1)
        accum.extend(preds.cpu().numpy())
        loss = criterion(outputs, labels)

        # Analyze the accuracy
        for i in range(len(preds)):
            AllTrain += 1
            if preds[i] == labels.data[i]:
                running_corrects_single += 1
    epoch_acc_single = running_corrects_single / len(Validation_Label_List)
    print('Epoch %d, Single test accuracy: %f' % (epoch, epoch_acc_single))
    loss = loss.data.cpu().tolist()
    epoch_acc_vote = epoch_acc_single

    if vote == True:
        running_corrects_vote = 0
        accum_vote = Vote(accum, n_samples)

        current_ind = 0
        for i in range(len(accum_vote)):
            if accum_vote[i] == Validation_Label_List[current_ind]:
                running_corrects_vote += 1
            current_ind += n_samples[i]
        epoch_acc_vote = running_corrects_vote / len(n_samples)

        print('Epoch %d, Vote accuracy: %f' % (epoch, epoch_acc_vote))
    return epoch_acc_single, epoch_acc_vote, loss

# Define Input data：  (batch_size, seq_len, dims)
print('Loading. Please wait... It may take 2-3 minutes')
since = time.time()
SampleRHS = LoadinRHS(input_file)

Train_loader = load_data(SampleRHS['Train_RHS_Sample'],
                         SampleRHS['Train_RHS_Label_Sample'], BATCH_SIZE, True)
Validation_loader = load_data(SampleRHS['Validation_RHS_Sample'],
                              SampleRHS['Validation_RHS_Label_Sample'], BATCH_SIZE, False)
Validation_Label_List, _ = convertLabel2Num(SampleRHS['Validation_RHS_Label_Sample'])

n_samples_train, n_samples_val = SampleRHS['Train_NSamples'], SampleRHS['Validation_NSamples']
LenTrain = len(SampleRHS['Train_RHS_Label_Sample'])
LenValidation = len(SampleRHS['Validation_RHS_Label_Sample'])
print('Number of Training data: ', LenTrain)
print('Number of Validation data: ', LenValidation)

# Hidden_dim is determined by the needs
model = LSTM(in_dim=2, hidden_dim=LSTM_hiddenDim, n_layer=LSTM_layer, n_class=NumOfCategory)
# print(model)
if use_gpu:
    model = model.cuda()

para_num = sum([p.data.nelement() for p in model.parameters()])
print('Total number of parameters:', para_num)

# 一下损失函数、优化器和学习率调整都可以修改
criterion = torch.nn.CrossEntropyLoss()
# This weight_decay parameter was set to prevent overfitting
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)# weight_decay=1e-8)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)

print('Training process started')
train_acc_history = []
test_single_acc_history = []
test_vote_acc_history = []
train_loss_history, test_loss_history = [], []
bad_counter = 0
best_acc = 0
best_epoch = 0
if resume_model:
    checkpoint = torch.load(os.path.join(save_dir, model_name+'.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

for epoch in range(EPOCH):
    start_time = time.time()
    model, train_acc, train_loss = train(model, Train_loader, criterion, optimizer, scheduler, epoch)
    train_endTime = time.time()
    test_single_acc, test_vote_acc, test_loss = test(model, Validation_loader, criterion, epoch,
                                          vote, n_samples_val, Validation_Label_List)
    test_endTime = time.time()
    train_acc_history.append(train_acc)
    test_single_acc_history.append(test_single_acc)
    test_vote_acc_history.append(test_vote_acc)
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    print('Training loss: %f, Test loss: %f' % (train_loss, test_loss))
    print('Training time: %.2f s, Test time: %.2f s' % (train_endTime - start_time, test_endTime - train_endTime))
    if test_single_acc > best_acc:
        best_acc = test_single_acc
        #best_loss = test_loss
        best_model = model
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1
    if bad_counter > maxTol:
        break

acc_history = np.array([train_acc_history, test_single_acc_history, test_vote_acc_history,
                        train_loss_history, test_loss_history])
LSTM_model = best_model
save_file = os.path.join(save_dir, model_name+'.pth')
torch.save({
    'epoch': best_epoch,
    'train_acc': train_acc_history[best_epoch],
    'test_single_acc': test_single_acc_history[best_epoch],
    'test_vote_acc': test_vote_acc_history[best_epoch],
    'train_loss': train_loss_history[best_epoch],
    'test_loss': test_loss_history[best_epoch],
    'model_state_dict': LSTM_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, save_file)

np.save(os.path.join(save_dir, model_name+'_acc_history.npy'), acc_history)



