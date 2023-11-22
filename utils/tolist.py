import torch
def state_dict_list(dic):
    for key,value in dic.items():
        dic[key]=value.tolist()
    return dic

def state_dict_torch(dic):
    for key,value in dic.items():
        dic[key]=torch.tensor(value,requires_grad=True)
    return dic