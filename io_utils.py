import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

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

# class SingleCharDataset(Dataset):
#     def __init__(self, data, label):
#         self.data = data
#         self.label = label
#
#     def __len__(self):
#         return len(self.landmarks_frame)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample

def load_data(data, label, batch_size, shuffle):
    print('pass0')
    seq_lengths = torch.LongTensor([len(seq) for seq in data])
    print('pass1')
    seq_tensor = torch.zeros((len(data), seq_lengths.max(), 2)).float()
    print('pass2')
    for idx, (seq, seqlen) in enumerate(zip(data, seq_lengths)):
        seq_tensor[idx, :seqlen, :] = torch.FloatTensor(seq)
    print('pass3')
    #seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    #seq_tensor = seq_tensor[perm_idx]

    label_list, _ = convertLabel2Num(label)
    label = np.transpose(np.array(label_list))

    # 把数据放在数据集中并以DataLoader送入网络训练
    dataset = TensorDataset(seq_tensor, seq_lengths, torch.from_numpy(label))
    print('pass4')
    data_loader = DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    print('pass5')
    return data_loader, label_list