import torch.nn as nn
import torch
import time
import numpy as np
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import copy
import logging
import os
from models.BERT.component import BertPooler,BertLayer,BertConfig,BertEmbeddings,BertLayerNorm,BertIntermediate,BertAttention,BertPooler,BertOutput,BertSelfAttention,BertSelfOutput

class BERTForClassification(nn.Module):
    def __init__(self, args):
        super(BERTForClassification, self).__init__()

        self.total_layer = int(args["total_layer"]) if args.get("total_layer") is not None else 15
        self.numClasses = int(args["n_class"]) if args.get("n_class") is not None else 9
        self.hidden = int(args["hidden_size"]) if args.get("hidden_size") is not None else 768
        self.vocabSize = int(args["vocab_size"]) if args.get("vocab_size") is not None else 30522
        self.numHiddenLayers = int(args["num_hidden_layers"]) if args.get("num_hidden_layers") is not None else 12

        self.attentionDropout = float(args["attention_dropout_prob"]) if args.get("attention_dropout_prob") is not None else 0.0
        self.hiddenDropout = float(args["hidden_dropout_prob"]) if args.get("hidden_dropout_prob") is not None else 0.0

        self.numAttentionHeads = int(args["num_attention_heads"]) if args.get("num_attention_heads") is not None else 12
        self.intermediateSize = int(args["intermediate_size"]) if args.get("intermediate_size") is not None else 3072
        config=BertConfig(
            vocab_size_or_config_json_file=self.vocabSize,
            hidden_size=self.hidden,
            num_hidden_layers=self.numHiddenLayers ,
            num_attention_heads=self.numAttentionHeads,
            intermediate_size=self.intermediateSize,
            hidden_dropout_prob=self.hiddenDropout,
            attention_probs_dropout_prob=self.attentionDropout)

        self.embeddings = BertEmbeddings(self.vocabSize,self.hidden,self.hiddenDropout)
        onelayer = BertLayer(config)
        self.encoder = nn.ModuleList([copy.deepcopy(onelayer) for _ in range(self.numHiddenLayers)])
        # self.pooler = BertPooler(config)
        # self.apply(self.init_bert_weights)
        self.dropout = nn.Dropout(self.hiddenDropout )
        self.classifier = nn.Linear(self.hidden, self.numClasses)
        self.apply(self.init_bert_weights)
        self.from_origin_pretrained("../models/bert-base-uncased/")

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def from_origin_pretrained(self, pretrained_model_path):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_path: either:

                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """

        weights_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location='cpu')

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)


        # print(state_dict.keys())

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata


        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            # print(local_metadata)
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    # print(prefix,";",name)
                    load(child, prefix + name + '.')

        load(self.embeddings, prefix='bert.embeddings.')
        load(self.encoder,prefix='bert.encoder.layer.')
        if(len(missing_keys)+len(unexpected_keys)+len(error_msgs)==0):
            print("Load pretrained weight successfully!")
        else:
            print("missing_keys",missing_keys)
            print("unexpected_keys",unexpected_keys)
            print("error_msgs",error_msgs)

    def forward(self, data):
        input_ids = data.get("input_ids")
        batch_size=len(input_ids)
        token_type_ids = data.get("token_type_ids")
        attention_mask = data.get("attention_mask")
        labels = data.get("labels")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        #embedding
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # enconder
        all_encoder_layers = []
        hidden_states=embedding_output
        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        sequence_output = self.dropout(hidden_states)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, self.numClasses)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return logits,loss
        else:
            return logits

    def calculate_acc(self,logits,labels):
        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, self.numClasses)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        _, predicted = torch.max(active_logits, -1)

        last = 0
        correct = 0
        cor = predicted == active_labels
        for x in (labels.view(labels.size(0), -1) != -100):
            num = x.sum().item()
            correct += cor[last:last + num].sum().item() / num
            last = last + num
        return correct
    # def profile_helper(self, x, rounds):
    #     return 10,10
    #     forward_time = np.zeros(self.total_layer + 1)
    #     backward_time = np.zeros(self.total_layer + 1)
    #
    #     batch_size = x.size(0)
    #     for i in range(rounds):
    #         outputs = []
    #         inputs = []
    #         print("Execution round {} start ...".format(i))
    #         # forward
    #         # feature
    #         for idx, module in enumerate(self.features):
    #             # detach from previous
    #             x = Variable(x.data, requires_grad=True)
    #             inputs.append(x)
    #
    #             # compute output
    #             start_time = time.time()
    #             x = module(x)
    #             forward_time[idx] += (time.time() - start_time)
    #
    #             outputs.append(x)
    #
    #         x = Variable(x.data, requires_grad=True)
    #         inputs.append(x)
    #         start_time = time.time()
    #         x = x.view((batch_size, -1))
    #         forward_time[len(self.features)] += (time.time() - start_time)
    #         outputs.append(x)
    #
    #         # classifier
    #         for idx, module in enumerate(self.classifier):
    #             # detach from previous
    #             x = Variable(x.data, requires_grad=True)
    #             inputs.append(x)
    #
    #             # compute output
    #             start_time = time.time()
    #             x = module(x)
    #             forward_time[idx + len(self.features) + 1] += (time.time() - start_time)
    #
    #             outputs.append(x)
    #
    #         # backward
    #         g = x
    #         for i, output in reversed(list(enumerate(outputs))):
    #             if i == (len(outputs) - 1):
    #                 start_time = time.time()
    #                 output.backward(g)
    #             else:
    #                 start_time = time.time()
    #                 output.backward(inputs[i + 1].grad.data)
    #
    #             backward_time[i] += (time.time() - start_time)
    #
    #     forward_time /= rounds
    #     backward_time /= rounds
    #     return forward_time, backward_time


class SubBERTForClassification(nn.Module):
    def __init__(self, start,end,args):
        super(SubBERTForClassification, self).__init__()
        self.total_layer = int(args["total_layer"]) if args.get("total_layer") is not None else 15
        self.numClasses = int(args["n_class"]) if args.get("n_class") is not None else 9
        self.hidden = int(args["hidden_size"]) if args.get("hidden_size") is not None else 768
        self.vocabSize = int(args["vocab_size"]) if args.get("vocab_size") is not None else 30522
        self.numHiddenLayers = int(args["num_hidden_layers"]) if args.get("num_hidden_layers") is not None else 12

        self.attentionDropout = float(args["attention_dropout_prob"]) if args.get("attention_dropout_prob") is not None else 0.0
        self.hiddenDropout = float(args["hidden_dropout_prob"]) if args.get("hidden_dropout_prob") is not None else 0.0

        self.numAttentionHeads = int(args["num_attention_heads"]) if args.get("num_attention_heads") is not None else 12
        self.intermediateSize = int(args["intermediate_size"]) if args.get("intermediate_size") is not None else 3072
        config=BertConfig(
            vocab_size_or_config_json_file=self.vocabSize,
            hidden_size=self.hidden,
            num_hidden_layers=self.numHiddenLayers ,
            num_attention_heads=self.numAttentionHeads,
            intermediate_size=self.intermediateSize,
            hidden_dropout_prob=self.hiddenDropout,
            attention_probs_dropout_prob=self.attentionDropout)


        if(end==-1):
            end=self.total_layer-1

        self.embeddings = BertEmbeddings(self.vocabSize,self.hidden,self.hiddenDropout) if start==0 else None
        onelayer = BertLayer(config)
        self.encoder = nn.ModuleList([copy.deepcopy(onelayer) for _ in range(min(end,self.numHiddenLayers)-max(start,1)+1)]) if end>0 and start<=self.numHiddenLayers else None
        # self.pooler = BertPooler(config)
        # self.apply(self.init_bert_weights)
        self.dropout = nn.Dropout(self.hiddenDropout) if start <= self.numHiddenLayers + 1 <= end else None
        self.classifier = nn.Linear(self.hidden, self.numClasses) if start <= self.numHiddenLayers + 2 <= end else None
        self.apply(self.init_bert_weights)
        self.from_origin_pretrained("../models/bert-base-uncased/",start)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def from_origin_pretrained(self, pretrained_model_path,start):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_path: either:

                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """

        weights_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location='cpu')

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)


        # print(state_dict.keys())

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata


        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            # print(prefix)
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        if self.embeddings is not None:
            load(self.embeddings, prefix='bert.embeddings.')
        start =start-1 if start>0 else 0
        if self.encoder is not None:
            for name, child in self.encoder._modules.items():
                if child is not None:
                    load(child, 'bert.encoder.layer.' + str(start) + '.')
                    start=start+1
        if(len(missing_keys)+len(unexpected_keys)+len(error_msgs)==0):
            print("Load pretrained weight successfully!")
        else:
            print("missing_keys",missing_keys)
            print("unexpected_keys",unexpected_keys)
            print("error_msgs",error_msgs)

    def forward(self, data,labels=None):
        output=extended_attention_mask=attention_mask=None

        if self.embeddings is None:
            output=data[0]
            attention_mask=data[1]

            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            # extended_attention_mask=torch.tensor(extended_attention_mask)

        if self.embeddings is not None:
            input_ids = data.get("input_ids")
            token_type_ids = data.get("token_type_ids")
            attention_mask = data.get("attention_mask")

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)

            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            # extended_attention_mask = torch.tensor(extended_attention_mask)

            #embedding
            output = self.embeddings(input_ids, token_type_ids) #hidden_states

        # enconder
        if self.encoder is not None:
            for layer_module in self.encoder:
                output = layer_module(output, extended_attention_mask)


        if self.dropout is not None:
            output = self.dropout(output)

        if self.classifier is None:
            return  [output,attention_mask]

        # sequence_output=data.pop("sequence_output")
        # labels=data.get("labels")
        # torch.save(output, "data1.pt")
        logits = self.classifier(output)

        if labels is not None:
            labels=labels[0]
            loss_fct = CrossEntropyLoss()
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, self.numClasses)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return logits,loss
        else:
            return logits

    def calculate_acc(self,logits,labels):
        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, self.numClasses)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        _, predicted = torch.max(active_logits, -1)

        last = 0
        correct = 0
        cor = predicted == active_labels
        for x in (labels.view(labels.size(0), -1) != -100):
            num = x.sum().item()
            correct += cor[last:last + num].sum().item() / num
            last = last + num
        return correct

class TryBert(nn.Module):
    def __init__(self, args):
        super(TryBert, self).__init__()
        self.total_layer = int(args["total_layer"]) if args.get("total_layer") is not None else 15
        self.numClasses = int(args["n_class"]) if args.get("n_class") is not None else 9
        self.hidden = int(args["hidden_size"]) if args.get("hidden_size") is not None else 768
        self.vocabSize = int(args["vocab_size"]) if args.get("vocab_size") is not None else 30522
        self.numHiddenLayers = int(args["num_hidden_layers"]) if args.get("num_hidden_layers") is not None else 12

        self.attentionDropout = float(args["attention_dropout_prob"]) if args.get(
            "attention_dropout_prob") is not None else 0.0
        self.hiddenDropout = float(args["hidden_dropout_prob"]) if args.get("hidden_dropout_prob") is not None else 0.0

        self.numAttentionHeads = int(args["num_attention_heads"]) if args.get("num_attention_heads") is not None else 12
        self.intermediateSize = int(args["intermediate_size"]) if args.get("intermediate_size") is not None else 3072
        config = BertConfig(
            vocab_size_or_config_json_file=self.vocabSize,
            hidden_size=self.hidden,
            num_hidden_layers=self.numHiddenLayers,
            num_attention_heads=self.numAttentionHeads,
            intermediate_size=self.intermediateSize,
            hidden_dropout_prob=self.hiddenDropout,
            attention_probs_dropout_prob=self.attentionDropout)
        self.bert1=SubBERTForClassification(0,3,args)
        self.bert2=SubBERTForClassification(4,14,args)
        # self.bert2=SubBERTForClassification(4,10,args)
        # self.bert3=SubBERTForClassification(11,14,args)

    def forward(self,data1):
        data2=self.bert1(data1)
        data20={}
        for key,value in data2.items():
            data20[key]=value
        data20['default']=data2['default'].detach().clone()
        data20['default'].requires_grad=True
        data3=self.bert2(data20)
        data3[1].backward()
        # print(data20['default']retain_grad())
        print(data20['default'].retain_grad())
        data2['default'].backward(data20['default'].retain_grad())
        # print(data)
        return data3

    def calculate_acc(self,logits,labels):
        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, self.numClasses)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        _, predicted = torch.max(active_logits, -1)

        last = 0
        correct = 0
        cor = predicted == active_labels
        for x in (labels.view(labels.size(0), -1) != -100):
            num = x.sum().item()
            correct += cor[last:last + num].sum().item() / num
            last = last + num
        return correct


