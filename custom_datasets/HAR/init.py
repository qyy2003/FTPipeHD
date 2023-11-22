import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from global_variables.config import cfg
from tensordict import TensorDict


def load_data_from_csv(path):
    # this functions load data of each subject in list_of_subjects
    # and return the list of subjects data
    data = []
    label = []
    
    df = pd.read_csv(path)
    
    # turn label into 0 for the first label 
    df.loc[:, "label"] = df["label"].apply(lambda x: x - 1)
    # acc
    df['body_acc_x'] = df['body_acc_x'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['body_acc_y'] = df['body_acc_y'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['body_acc_z'] = df['body_acc_z'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    # gyro
    df['body_gyro_x'] = df['body_gyro_x'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['body_gyro_y'] = df['body_gyro_y'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['body_gyro_z'] = df['body_gyro_z'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    # total acc
    df['total_acc_x'] = df['total_acc_x'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['total_acc_y'] = df['total_acc_y'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['total_acc_z'] = df['total_acc_z'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    
    for x in range(1, 26):
        client_pd = df.loc[df['subject'] == x]
        client_label = client_pd["label"].to_numpy() 
        client_data = np.transpose(np.apply_along_axis(np.stack, 1, client_pd.drop(["label","subject"], axis=1).to_numpy()),(0,1,2))
        data.append(client_data)
        label.append(client_label)
    
    data = np.concatenate([np.array(i) for i in data])
    label = np.concatenate([np.array(i) for i in label])

    return data,label


class HARDataset(Dataset):
    def __init__(self, data_, label_):
        self.data = data_
        self.labels = label_

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx])
        label = self.labels[idx]
        
        return data,label

def custom_collate_fn(datas):
    data0=[]
    label0=[]
    for data,label in datas:
        data0.append(data)
        label0.append(label)
    # return {x: torch.tensor(unit_x), y: torch.tensor(unit_y)}
    # print(data0)
    # print("----")
    data0=torch.stack(data0,0)
    label0=torch.tensor(label0)
    # print(len(label0))
    # return {"inputs": data0, "labels": label0}
    return TensorDict({"inputs": data0, "labels": label0}, batch_size=[len(data0)])
def init_har():
    data, label = load_data_from_csv(cfg.data.train.dataset_path)
    dataset = HARDataset(data, label)

    # split into training and validation
    train_idx, valid_idx= train_test_split(
        np.arange(len(label)), test_size=0.3, random_state=42, shuffle=True, stratify=label)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # trainloader = torch.utils.data.DataLoader(dataset,
    #                                           batch_size=1,collate_fn=custom_collate_fn,
    #                                           sampler=train_sampler, num_workers=1)
    trainloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=cfg.data.batch_size,collate_fn=custom_collate_fn,
                                              sampler=train_sampler, num_workers=1)

    validloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=cfg.data.batch_size,collate_fn=custom_collate_fn,
                                              sampler=valid_sampler, num_workers=1)
    
    return trainloader, validloader