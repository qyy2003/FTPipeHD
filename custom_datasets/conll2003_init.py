import torchvision.transforms as transforms
import torch
from global_variables.config import cfg
from torchvision.datasets import CIFAR10
from datasets import load_dataset
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("../models/bert-base-uncased")
label_all_tokens = True
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def init_conll2003():
    """
        Initialize the CIFAR10 dataset and return the training set and test set
    """
    dataset_config=cfg.data
    datasets = load_dataset(dataset_config.path)
    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True,remove_columns=datasets["train"].column_names)
    from transformers import DataCollatorForTokenClassification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # train_dataloader=tokenized_datasets["train"]
    # import torch
    # train_size = 100
    # test_size = len(train_dataloader) - train_size
    # train_dataloader, test_dataset = torch.utils.data.random_split(train_dataloader, [train_size, test_size])
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataloader, batch_size=dataset_config.batch_size, shuffle=True, collate_fn=data_collator
    # )
    def my_data_collator(data):
        data=data_collator(data)
        data["labels"]=[data["labels"]]
        return data
    train_dataloader = torch.utils.data.DataLoader(
        tokenized_datasets["train"], batch_size=dataset_config.batch_size, shuffle=True,collate_fn=my_data_collator
    )

    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_datasets["validation"], collate_fn=data_collator, batch_size=dataset_config.batch_size
    )
    return train_dataloader ,eval_dataloader