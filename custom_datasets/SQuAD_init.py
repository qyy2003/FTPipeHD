import torch
from global_variables.config import cfg
from tensordict import TensorDict

from datasets import load_dataset
import os
def data_collator(data):

    input_ids = [x["input_ids"].type(torch.int64) for x in data]
    start_pos = [x["start_pos"].type(torch.int64) for x in data]
    end_pos = [x["end_pos"].type(torch.int64) for x in data]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    # [torch.tensor(start_pos),torch.tensor(end_pos)]
    return {"input_ids":x,"labels":[torch.tensor(start_pos),torch.tensor(end_pos)]}
    # return TensorDict({"inputs": x, "labels": y}, batch_size=[len(data)])
    # return x, y
def init_SQuAD():
    """
        Initialize the CIFAR10 dataset and return the training set and test set
    """
    dataset_config=cfg.data
    train_data = torch.load(os.path.join(dataset_config.path, "train.pt"))
    val_data = torch.load(os.path.join(dataset_config.path, "test.pt"))

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=dataset_config.batch_size, shuffle=True, collate_fn=data_collator
    )

    eval_dataloader = torch.utils.data.DataLoader(
        val_data , collate_fn=data_collator, batch_size=dataset_config.batch_size
    )
    return train_dataloader, eval_dataloader
