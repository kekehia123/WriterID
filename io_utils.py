import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os

# 加载经过数据预处理所得到的RHS文件
def LoadinRHS(path):
    f = open(path, 'r')
    a = f.read()
    SampleRHS = eval(a)
    f.close()
    return SampleRHS

# 文件名所提供的Label为字符串格式，该函数通过一个字典实现字符串形式标签和one-hot标签的对应
def convertLabel2Num(Train_RHS_Label_Sample):
    Label_dict = {}
    num = -1
    Label_return = []
    for ind in Train_RHS_Label_Sample:
        if ind in Label_dict:
            continue
        else:
            num += 1
            Label_dict[ind] = num
    for ind in Train_RHS_Label_Sample:
        Label_return.append(Label_dict[ind])
    return Label_return, Label_dict

class SingleCharDataset(Dataset):
    def __init__(self, data, label):
        seq_lengths = torch.LongTensor([len(seq) for seq in data])
        seq_tensor = torch.zeros((len(data), seq_lengths.max(), 2)).float()
        for idx, (seq, seqlen) in enumerate(zip(data, seq_lengths)):
            seq_tensor[idx, :seqlen, :] = torch.FloatTensor(seq)

        label_list, _ = convertLabel2Num(label)
        label = np.transpose(np.array(label_list))

        self.label = torch.from_numpy(label)
        self.seq_tensor = seq_tensor
        self.seq_lengths = seq_lengths

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        one_seq = self.seq_tensor[idx]
        one_len = self.seq_lengths[idx]
        one_label = self.label[idx]
        #sample = {'seq': one_seq, 'len': one_len, 'label': one_label}
        return one_seq, one_len, one_label

def load_data(data, label, batch_size, shuffle):
    # 把数据放在数据集中并以DataLoader送入网络训练
    dataset = SingleCharDataset(data, label)
    data_loader = DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return data_loader