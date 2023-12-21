import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import math 





######################SINGLE ATTENTION HEAD##########################
L, dk, dv = 4, 8, 8 #L = length of input sequence -> ex) My name is Sean
q = np.random.rand(L, dk)
k = np.random.rand(L, dk)
v = np.random.rand(L, dv)
# print(q,k,v)
#SELF ATTENTION :  Every word to look at every other words to see if there is a higher affinity
qKT = np.matmul(q, k.T)
#minimize the variance
scaled = qKT / np.sqrt(dk) #scaled qKTsqrtdk


#MASKING
#to ensure the words to not get context from words generated in the future
mask = np.tril(np.ones((L,L)))#4 * 4
mask[mask == 0] = -np.inf
mask[mask == 1] = 0
# print(mask)
# print(scaled + mask)

#making into probability distribution, values add up to 1, interpretable 
def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x) , axis = -1)).T
attention = softmax(scaled + mask)
# print(attention)
new_v = np.matmul(attention, v)




##################Multi Head Attention#######################
sequenceLength = 4
batchsize = 1
inputDim = 512 #input of attention unit
d_model = 512#output of attention unit
x = torch.randn((batchsize, sequenceLength, inputDim)) #x is the one that is put in just before attention step:1,4,512
qkv_layer = nn.Linear(inputDim, 3 * d_model)
qkv = qkv_layer(x)

numHeads= 8 # head number for attention to make it in parallel
headDim = d_model  // numHeads
qkv = qkv.reshape(batchsize,sequenceLength,numHeads, 3 * headDim)
qkv = qkv.permute(0,2,1,3)
q,k,v = qkv.chunk(3, dim=-1)
# print(q.shape, k.shape, v.shape)

d_k = q.size()[-1]
scaled = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)
# print(scaled.shape)
mask = torch.full(scaled.size(), float('-inf'))
mask = torch.triu(mask, diagonal=1)
# print(mask)
scaled += mask
attention = F.softmax(scaled, dim=-1)
# print(attention.shape)
values = torch.matmul(attention, scaled)
# print(values.shape)

################ positional encoding  ###################
max_sequence_length = 10
d_model = 6
even_i = torch.arange(0,d_model , 2).float()
odd_i = torch.arange(1, d_model, 2).float()
even_denominator = torch.pow(10000, even_i / d_model)
odd_denominator = torch.pow(10000, odd_i / d_model)
position = torch.arange(max_sequence_length, dtype= torch.float).reshape(max_sequence_length, 1)
# print(position)
even_pe = torch.sin(position/ even_denominator)
odd_pe = torch.cos(position/ odd_denominator)
# print(even_pe)
# print(odd_pe)
stacked = torch.stack([even_pe, odd_pe], dim = 2)
# print(stacked)
PE = torch.flatten(stacked, start_dim=1, end_dim=2)
# print(PE)


################4. layer normalization ################
inputs = torch.Tensor([[[0.2,0.1,0.3], [0.5,0.1,0.1]]])
B , S , E = inputs.size()# S = num of words, B = batch size , E = embedding
inputs = inputs.reshape(S,B,E)
parameter_shape = inputs.size()[-2:]
# print(parameter_shape)
gamma = nn.Parameter(torch.ones(parameter_shape))
beta = nn.Parameter(torch.zeros(parameter_shape))

dims = [-(i + 1) for i in range(len(parameter_shape))] # dimension for layer normalization
# print(dims)
# print(gamma.size())
# print(beta.size())
mean = inputs.mean(dim = dims, keepdim=True)
# print(mean.size())
var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
epsilon = 1e-5
std = (var + epsilon).sqrt()
y = (inputs - mean) / std
print(y)
out = gamma * y + beta
print(out)


