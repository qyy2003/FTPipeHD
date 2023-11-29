'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import copy
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import CrossEntropyLoss

# input：x
# output：gelu
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))# 
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        ans = self.weight * x + self.bias
        return ans

# input nx ——> nf
# 卷积机制
class Conv1D(nn.Module):
    # nf output, nx input
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()

        self.nf = nf
        w = torch.empty(nx, nf)# initial a tensor 
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

# attention 机制
# Attention机制的作用是，对于每一个词，都会计算它与其他所有词的attention score，然后对其他所有词的embedding加权求和，作为该词的context embedding。
class old_Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=True):
        super(old_Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        # torch.tril()函数代表取下三角，另上三角全为0
        self.n_head = config.n_head
        self.split_size = n_state # 768
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        # print("self.c_attan",self.c_attn,"~~~~~~~~~~~~~~\n")
        self.c_proj = Conv1D(n_state, nx)

# nd, head =12
# ns, 768/12 = 64
# q,v  (batch, n_head, seq_length, head_features/ n_head)
# k . (batch, n_head, head_feature / n_head , seq_length, )
    def _attn(self, q, k, v):
        w = torch.matmul(q, k)#(batch, n_head, seq_length, seq_length)。
        w = w / math.sqrt(v.size(-1))
        # print(w.shape,"~~~~~~~~~~~~~~")
        nd, ns = w.size(-2), w.size(-1)
        # b = self.bias[:, :, 0:1024, :1024]
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)# 在最后一个维度上进行softmax
        return torch.matmul(w, v) 

# x.permute(0, 2, 1, 3) 用于交换张量的维度，将 num_heads 和 sequence_length 交换位置，以便在合并时按照正确的顺序组织注意力头。
# x.contiguous() 用于确保张量的内存布局是连续的，这在某些情况下是必要的，以确保后续的操作可以正确执行。
# new_x_shape 计算新的形状，将多头注意力的结果展平为一个单一的特征维度，即 (batch_size, sequence_length, num_heads * head_features)。
# x.view(*new_x_shape) 将张量重新整形为新的形状。*new_x_shape 表示解包 new_x_shape 元组，以传递给 view 函数。
    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states


    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        #(batch_size, seq_length, dim) => (batch_size, seq_length, n_head, dim // n_head)
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, n_head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, n_head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)# 3个线性层,将最后一个维度变为3倍
        query, key, value = x.split(self.split_size, dim=2)# 将最后一个维度分为3份,分别为query,key,value

        query = self.split_heads(query)# 将query分为多个头 (batch, head, seq_length, head_features)
        key = self.split_heads(key, k=True)#              (batch, head, head_features, seq_length) 不要转置了
        value = self.split_heads(value)#                 (batch, head, seq_length, head_features)

        if layer_past is not None:# 如果有过去的层
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below # 将过去的key和value转置
            key = torch.cat((past_key, key), dim=-1)# 将过去的key和现在的key拼接
            value = torch.cat((past_value, value), dim=-2)# 将过去的value和现在的value拼接

        #[2, batch_size, num_heads, head_features, sequence_length]。第一个维度 2 表示两个堆叠的张量，分别是转置后的键和值    
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value)
        a = self.merge_heads(a) # [batch_size, sequence_length, hidden_size]]
        a = self.c_proj(a) # 
        return a, present
        
# new attan
class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=True):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        # torch.tril()函数代表取下三角，另上三角全为0
        self.n_head = config.n_head
        self.split_size = n_state # 768
        self.scale = scale
        self.c_attn_query = Conv1D(n_state , nx)
        self.c_attn_key = Conv1D(n_state , nx)
        self.c_attn_value = Conv1D(n_state , nx)
        # print("self.c_attan",self.c_attn,"~~~~~~~~~~~~~~\n")
        self.c_proj = Conv1D(n_state, nx)

# nd, head =12
# ns, 768/12 = 64
# q,v  (batch, n_head, seq_length, head_features/ n_head)
# k . (batch, n_head, head_feature / n_head , seq_length, )
    def _attn(self, q, k, v):
        k_shape = k.shape
        q_shape = q.shape
        w = torch.matmul(q, k)#(batch, n_head, seq_length, seq_length)。
        w = w / math.sqrt(v.size(-1))
        shape= w.shape
        nd, ns = w.size(-2), w.size(-1)
        # b = self.bias[:, :, 0:1024, :1024]
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)# 在最后一个维度上进行softmax
        return torch.matmul(w, v) 

# x.permute(0, 2, 1, 3) 用于交换张量的维度，将 num_heads 和 sequence_length 交换位置，以便在合并时按照正确的顺序组织注意力头。
# x.contiguous() 用于确保张量的内存布局是连续的，这在某些情况下是必要的，以确保后续的操作可以正确执行。
# new_x_shape 计算新的形状，将多头注意力的结果展平为一个单一的特征维度，即 (batch_size, sequence_length, num_heads * head_features)。
# x.view(*new_x_shape) 将张量重新整形为新的形状。*new_x_shape 表示解包 new_x_shape 元组，以传递给 view 函数。
    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states


    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        #(batch_size, seq_length, dim) => (batch_size, seq_length, n_head, dim // n_head)
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, n_head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, n_head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        # x = self.c_attn(x)# 3个线性层,将最后一个维度变为3倍
        # query, key, value = x.split(self.split_size, dim=2)# 将最后一个维度分为3份,分别为query,key,value
        query = self.c_attn_query(x)
        key = self.c_attn_key(x)
        value = self.c_attn_value(x)

        k_size, v_size , q_size= key.shape, value.shape, query.shape
        query = self.split_heads(query)# 将query分为多个头 (batch, head, seq_length, head_features)
        key = self.split_heads(key, k=True)#              (batch, head, head_features, seq_length) 不要转置了
        value = self.split_heads(value)#                 (batch, head, seq_length, head_features)
        k_size, v_size , q_size= key.shape, value.shape, query.shape

        if layer_past is not None:# 如果有过去的层
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below # 将过去的key和value转置
            key = torch.cat((past_key, key), dim=-1)# 将过去的key和现在的key拼接
            value = torch.cat((past_value, value), dim=-2)# 将过去的value和现在的value拼接
        
        k_size, v_size , q_size= key.shape, value.shape, query.shape

        #[2, batch_size, num_heads, head_features, sequence_length]。第一个维度 2 表示两个堆叠的张量，分别是转置后的键和值    
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value)# 1,12,2,64
        a = self.merge_heads(a) # 1,2,768 [batch_size, sequence_length, hidden_size]]
        a = self.c_proj(a) # 
        return [a, present]

# 多层感知机
class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()

        nx = config.n_embd# 768
        self.c_fc = Conv1D(n_state, nx)# n_state=3072 (4 * n_embd) nx=768
        self.c_proj = Conv1D(nx, n_state)# nx=768 n_state=3072 (4 * n_embd)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2 # 

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=True):
        super(Block, self).__init__()
        nx = config.n_embd # 768
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        # print(type(x),"~~~~~~~~~~~~~~~" )
        [a, present] = self.attn(self.ln_1(x), layer_past=layer_past)# [batch_size , sequence_length, hidden_size]
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m # 
        return [x, present]

# 修改后的仅仅12层model
# main part ,for blocks 
class GPT2Decoder(nn.Module):
    def __init__(self, config):
        super(GPT2Decoder, self).__init__()
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])       
 
    def forward(self, x):
        presents = []
        hidden_states = x
        for block in self.h:
            hidden_states, present = block(hidden_states, None)
            presents.append(present)

        return hidden_states,presents
    
    
class GPT2Embedding(nn.Module):
    def __init__(self, config):
        super(GPT2Embedding, self).__init__()
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size
        self.h = config.n_layer

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)# Wte 词嵌入层
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)# position embeeding 

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)# 输入的神经元个数  输出神经元个数  是否包含偏置
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * self.h# 12
        else:
            past_length = past[0][0].size(-2)# 就是倒数最后第二个维度的长度

        if position_ids is None:
 #  这行代码使用 PyTorch 创建一个从 past_length 到 past_length + input_ids.size(-1) 的整数序列。这个序列用于表示输入序列中每个位置的位置编码。past_length 表示之前历史信息的长度，          
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)# 0表示,在第一个位置增加维度，然后在第一个维度上扩展
        position_ids = position_ids.view(-1, position_ids.size(-1))# 类似[3,4,10,5] => [3*4*10,5] 

        input_shape = input_ids.size()# [batch_size, sequence_length]
        input_ids = input_ids.view(-1, input_ids.size(-1))#  [3,4,10,5] => [3*4*10,5]
       
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))# [3,4,10,5] => [3*4*10,5]
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds

        return [hidden_states]


# main part ,for blocks 
class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)# Wte 词嵌入层
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)# position embeeding 

        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        #copy.deepcopy(block)：这是使用 copy 模块中的 deepcopy 函数来创建 block 的深层拷贝。深层拷贝意味着创建了一个 block 的独立副本，
        # 它与原始的 block 完全独立，对一个副本的修改不会影响其他副本。结果，列表推导生成了 config.n_layer 个深层拷贝的 block 模块，这些模块存储在一个列表中。
        
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)# n_embd=768

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)# 输入的神经元个数  输出神经元个数  是否包含偏置
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)# 12
        else:
            past_length = past[0][0].size(-2)# 就是倒数最后第二个维度的长度
        # 没有救生舱个一个position_ids
        if position_ids is None:
 #  这行代码使用 PyTorch 创建一个从 past_length 到 past_length + input_ids.size(-1) 的整数序列。这个序列用于表示输入序列中每个位置的位置编码。past_length 表示之前历史信息的长度，          
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)# 0表示,在第一个位置增加维度，然后在第一个维度上扩展
        position_ids = position_ids.view(-1, position_ids.size(-1))# 类似[3,4,10,5] => [3*4*10,5] 

        input_shape = input_ids.size()# [batch_size, sequence_length]
        input_ids = input_ids.view(-1, input_ids.size(-1))#  [3,4,10,5] => [3*4*10,5]
       
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))# [3,4,10,5] => [3*4*10,5]
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds

        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)# [batch_size  ,sequence_length, hidden_size]
        output_shape = input_shape + (hidden_states.size(-1),)# [batch_size, sequence_length, hidden_size]
        return hidden_states.view(*output_shape), presents

# the 概率分布，用于下一个词的预测
class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd #768
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits

# 合成的最后整体模型
class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(config) #  将其子模型定义为transformer
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)# 表示在计算损失时会忽略标签中值为 -1 的项,  计算交叉熵损失（cross-entropy loss）
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents

class GPT2Train(nn.Module):
    def __init__(self, config):
        super(GPT2Train, self).__init__()
        self.transformer = GPT2Model(config)
        self.qa_output = nn.Linear(config.n_embd, 2)
    
    def forward(self, input_ids):
        hidden_states, presents = self.transformer(input_ids, position_ids=None, token_type_ids=None, past=None)
        logits =  self.qa_output(hidden_states)
        start_logits, end_logits = logits.split(1, dim = 2)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

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
            return total_loss
        else:
            return start_logits, end_logits
        return logits
    

if __name__ == '__main__':
    print("model test")
