import torch.nn as nn
import torch
import time
import numpy as np
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import copy
import logging
import os
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter

from models.GPT2.config import GPT2Config
from models.GPT2.component import Block, GPT2Embedding


class SubGPT2ForClassification(nn.Module):
    def __init__(self, start, end, args):
        super(SubGPT2ForClassification, self).__init__()
        self.vocab_size = int(args["vocab_size"]) if args.get("vocab_size") is not None else 50257
        self.n_ctx = int(args["n_ctx"]) if args.get("n_ctx") is not None else 1024
        self.n_positions = int(args["n_positions"]) if args.get("n_positions") is not None else 1024
        self.hidden = int(args["hidden_size"]) if args.get("hidden_size") is not None else 768
        self.total_layer = int(args["total_layer"]) if args.get("total_layer") is not None else 12

        self.numAttentionHeads = int(args["num_attention_heads"]) if args.get("num_attention_heads") is not None else 12
        self.norm_epsilon = float(args["norm_epsilon"]) if args.get("norm_epsilon") is not None else 1e-5
        self.initializer_range = float(args["initializer_range"]) if args.get("initializer_range") is not None else 0.02
        self.numClasses = int(args["num_classes"]) if args.get("num_classes") is not None else 2

        config = GPT2Config(
            vocab_size_or_config_json_file=self.vocab_size,
            n_positions=self.n_positions,
            n_ctx=self.n_ctx,
            n_embd=self.hidden,
            n_layer=self.total_layer,
            n_head=self.numAttentionHeads,
            layer_norm_epsilon=self.norm_epsilon,
            initializer_range=self.initializer_range, )

        if (end == -1):
            end = self.total_layer - 1

        self.embeddings = GPT2Embedding(config) if start == 0 else None
        onelayer = Block(config.n_ctx, config, scale=True)
        self.encoder = nn.ModuleList([copy.deepcopy(onelayer) for _ in range(
            min(end, self.total_layer) - max(start, 1) + 1)]) if end > 0 and start <= self.total_layer else None
        self.lnf = nn.LayerNorm(config.n_embd,
                                eps=config.layer_norm_epsilon) if start <= self.total_layer + 1 <= end else None

        self.classifier = nn.Linear(self.hidden, self.numClasses) if start <= self.total_layer + 1 <= end else None
        # self.apply(self.init_gpt2_weights)
        # self.from_origin_pretrained("../models/bert-base-uncased/",start)


    def forward(self, data, labels=None):
        start_positions = None
        end_positions = None

        if self.embeddings is None:
            output = data[0]
            past = data[1]

        else:
            input_ids = data.get("input_ids")
            position_ids = data.get("position_ids")
            token_type_ids = data.get("token_type_ids")
            past = data.get("past")

            # embedding
            embed = self.embeddings(input_ids, position_ids, token_type_ids, past)  # hidden_states
            output = embed[0]
            past = None

        # enconder
        if self.encoder is not None:
            for layer_module in self.encoder:
                [output, past] = layer_module(x=output, layer_past=past)

        if self.classifier is None:
            return [output, past]
        else:
            logits = self.classifier(self.lnf(output))

        if labels is not None:
            start_logits, end_logits = logits.split(1, dim=2)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            start_positions = torch.tensor(labels[0])
            end_positions = torch.tensor(labels[1])

            if start_positions is not None and end_positions is not None:
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_function = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_function(start_logits, start_positions)
                end_loss = loss_function(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                return [total_loss]
            else:
                return [start_logits, end_logits]
        else:
            return [logits]


if __name__ == "__main__":
    print("gpt2_test")
    args = {"vocab_size": 50257, "n_ctx": 1024, "n_positions": 1024, "hidden_size": 768, "total_layer": 12,
            "num_attention_heads": 12, "norm_epsilon": 1e-5, "initializer_range": 0.02, "num_classes": 2}
    data = {"input_ids": torch.tensor([[8,784, 1024, 1, 1, 1, 1, 1, 34, 65]]), "position_ids": None,
            "token_type_ids": None, "past": None}
    from custom_datasets.SQuAD_init import init_SQuAD
    from global_variables.config import cfg, load_config
    load_config(cfg, args.config)
    train,test=init_SQuAD()
    for batch in train:
        data=batch
        model_1 = SubGPT2ForClassification(0, 4, args)
        model1_output = model_1(data)
        model_2 = SubGPT2ForClassification(5, 8, args)
        model2_output = model_2(model1_output, labels=batch['labels'])
    print(model2_output)
