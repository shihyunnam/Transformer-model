import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

sequence_length = 4#my name is j
batch_size = 1
input_dim = 512
d_model = 512 #output of attention unit for individual words
x = torch.randn((batch_size, sequence_length, d_model))
print(x.size())


#
qkv_layer = nn.Linear(input_dim , 3 * d_model)
qkv = qkv_layer(x)
# print(qkv.shape)
num_heads = 8
head_dim = d_model // 8
# print(head_dim)
qkv = qkv.reshape(batch_size, sequence_length, num_heads, 3 * head_dim)
# print(qkv.shape)
qkv = qkv.permute(0,2,1,3) # just changing dimension to parallelize it easier



q,k,v = qkv.chunk(3, dim=-1)
d_k = q.size()[-1]
print(q.shape,k.shape,v.shape)
scaled = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(d_k)
print(scaled.size())

mask = torch.full(scaled.size(), float('-inf'))
mask = torch.triu(mask, diagonal=1)
print(mask[0][1])