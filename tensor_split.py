'''
File created by Yukuan Yang
9 Aug, 2020
'''
import torch

def split_4d_tensor(input_tensor,split_group=[0,1,1,1]):
    #input_c_split = input.chunk(split_group[1],dim=1)
    input_h_split = input_tensor.chunk(split_group[2],dim=2)

    input_hw_split = []
    for input_h in input_h_split:
        input_hw_ = input_h.chunk(split_group[3],dim=3)
        input_hw_split.append(input_hw_)

    return input_hw_split


def cat_4d_tensor(input_hw_split):
    input_h_list = []
    for input_hw_ in input_hw_split:
        input_h = torch.cat(list(input_hw_),axis = 3)
        input_h_list.append(input_h)
    input_tensor_cat = torch.cat(input_h_list,axis = 2)
    return input_tensor_cat


def channel_shuffle(input_tensor,random_id):
    channel = input_tensor.shape[1]
    input_shuffle = input_tensor[:,random_id,:,:]
    return input_shuffle


'''
#split test
input_tensor = torch.rand([32,3,224,224])
input_hw_split = split_4d_tensor(input_tensor)
input_tensor_cat=cat_4d_tensor(input_hw_split)

print(torch.equal(input_tensor,input_tensor_cat))

print('tensor split test')

'''


