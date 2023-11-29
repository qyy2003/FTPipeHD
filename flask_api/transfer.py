import torch
def single_MNN_to_pytorch(data,is_long):
    # print(len(data))
    # data[0]=np.array(data[0]).reshape(tuple(data[1]))
    if(data[3]==0):
        data0=torch.tensor(data[0],dtype=torch.float32).reshape(data[1]).detach()
        data0.requires_grad=True
        return data0
    else:
        if is_long:
            return torch.tensor(data[0], dtype=torch.long).reshape(data[1])
        else:
            return torch.tensor(data[0], dtype=torch.int32).reshape(data[1])

def MNN_to_pytorch(data,lens=4,is_long=0):
    data0=[]
    for i in range(0,len(data),lens):
        data0.append(single_MNN_to_pytorch(data[i:i+lens],is_long))
    return data0

def single_pytorch_to_MNN(data,order=2):
    data0=[]
    data0.append(data.flatten().tolist())
    data0.append(list(data.shape))
    if data.dtype==torch.int32 or data.dtype==torch.int64 or data.dtype==torch.long:
        data0.extend([order,1])
    else:
        data0.extend([order, 0])
    return data0
def pytorch_to_MNN(data):
    data_final=[]
    for item in data:
        data_final.extend(single_pytorch_to_MNN(item))
    return data_final