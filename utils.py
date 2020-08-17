import torch
import torch.nn as nn
from torch.nn import functional as F#for Conv2d
from torch.autograd import Function
import numpy as np
from torch.nn.parameter import Parameter


bitsW = 8
bitsA = 8


def delta(bits):
    result = (2.**(1-bits))
    return result

def clip(x, bits):
    if bits >= 32:
        step = 0
    else:
        step = delta(bits)
    ceil  = 1 - step
    floor = step - 1
    result = torch.clamp(x, floor, ceil)
    return result

def quant(x, bits):
    if bits >= 32:
        result = x
    else:
        result = torch.round(x/delta(bits))*delta(bits)
    return result

def qw(x):
    bits = bitsW
    if bits >= 32:
        result = x
    else:
        result = clip(quant(x,bits),bits)
    return result

def qa(x):
    bits = bitsA
    if bits >= 32:
        result = x
    else:
        result = quant(x,bits)
    return result

class QW(Function):
    @staticmethod
    def forward(self, x):
        result = qw(x)
        return result

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output
        return grad_input

class QA(Function):
    @staticmethod
    def forward(self, x):
        result = qa(x)
        return result

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output
        return grad_input

quantizeW = QW().apply
quantizeA = QA().apply





#quantization conv
class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

        #stride
        #s1 for line; s2 for column
        if isinstance(stride,int) or len(stride) == 1:
            self.s1 = stride
            self.s2 = stride
        elif len(stride) == 2:
            self.s1 = stride[0]
            self.s2 = stride[1]
        #padding
        self.p = padding

        #kernel size
        if isinstance(kernel_size,int) or len(kernel_size) == 1:
            self.k1 = kernel_size
            self.k2 = kernel_size
        elif len(kernel_size) == 2:
            self.k1 = kernel_size[0]
            self.k2 = kernel_size[1]

        #output feature size
        self.output_H = None
        self.output_W = None

        self.input_shape = None


    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        self.input_shape = input.shape
        return self.conv2d_forward(quantizeA(input),quantizeW(self.weight))


    def get_feat_ID(self,weight_position):

        '''
        :param input: conv input
        :param weight_position: the weight relative position in the kernel, the first point in the top left is (0,0)
        :return: the corresponding feature position index of weight in the position of given weight_position
        '''


        #input_shape = input.shape
        input_shape = self.input_shape
        assert len(input_shape) == 4
        assert len(weight_position) == 2

        input_C = input_shape[1]
        input_H = input_shape[2]
        input_W = input_shape[3]

        import numpy as np
        self.output_H = np.int(np.floor((input_H + 2*self.p-self.k1)/self.s1) + 1)
        self.output_W = np.int(np.floor((input_W + 2*self.p-self.k2)/self.s2) + 1)

        #first weight (0,0)
        H_ID_base = []
        W_ID_base = []

        for i in range(self.output_H):
            H_ID_base.append(self.s1*i)

        for i in range(self.output_W):
            W_ID_base.append(self.s2*i)

        H_ID = [h + weight_position[0] for h in H_ID_base]
        W_ID = [w + weight_position[1] for w in W_ID_base]

        HW_position = np.zeros([len(H_ID)*len(W_ID),2]).astype(np.int)

        for i, h in enumerate(H_ID):
            for j, w in enumerate(W_ID):
                HW_position[i*len(W_ID)+j][0] = np.int(h)
                HW_position[i*len(W_ID)+j][1] = np.int(w)
        return HW_position


#control the feature structure
class Mask(nn.Module):
    def __init__(self,bits=1):
        super(Mask, self).__init__()
        self.weight_shape = None
        self.channel = None
        self.height = None
        self.width = None
        self.weight = None
        self.bits = bits

    def quan(self):
        if self.bits == None:
            return self.weight
        else:
           max_quan = 2**(self.bits-1)
           quan_weight = torch.round(max_quan*self.weight)/max_quan
        return quan_weight

    def forward(self,input):

        if type(self.weight) == type(None):
           # normal random initialization
           # self.weight = Parameter(torch.nn.init.normal_(torch.Tensor(input.shape[1:]),mean=0.5,std=0.1).cuda())

           # uniform initialization
           # self.weight = Parameter(torch.nn.init.uniform(torch.Tensor(input.shape[1:]), a=0.0, b=1.0).cuda())

           # constant initialization all 1
           self.weight = Parameter(torch.nn.init.constant_(torch.Tensor(input.shape[1:]), val=1.0).cuda(),requires_grad=False)
        if type(self.weight_shape) == type(None):
           self.weight_shape = input.shape[1:]
           self.channel = input.shape[1]
           self.height = input.shape[2]
           self.width = input.shape[3]

        return input*self.quan()


#get ready for regularization loss
#divide the feat_ID into groups according to structure paramteter
def array_split(feat_ID, stru_param):

    array_shape = feat_ID.shape
    length = array_shape[0]
    import numpy as np
    group_num = np.int(np.floor(length/np.float(stru_param)))
    all_group_list = []
    for i in range(group_num):
        small_group_list = []
        for j in range(stru_param):
            small_group_list.append(feat_ID[i*stru_param+j])
        all_group_list.append(small_group_list)

    return all_group_list


#computing single layer regularization loss
def structure_loss_singlelayer(mask, conv_layer, stru_param=None):
    if stru_param == None:
        return 0.0
    else:
        k1 = conv_layer.k1
        k2 = conv_layer.k2

        all_positions = np.zeros([k1*k2,2]).astype(np.int)

        for i in range(k1):
            for j in range(k2):
                all_positions[i*k2+j,0] = i
                all_positions[i*k2+j,1] = j

        singlelayer_loss = 0
        for i,position in enumerate(all_positions):
            feat_ID = conv_layer.get_feat_ID((position[0],position[1]))
            all_group = array_split(feat_ID,stru_param)
            single_weight_loss = 0
            for small_group in all_group:
                all_channel_smallgroup_loss = 0
                for channel in range(mask.channel):
                    small_group_loss = 0
                    for single in small_group:
                        structure_regularization_weight = 0.001
                        small_group_loss = small_group_loss + structure_regularization_weight*torch.abs(mask.weight[channel,single[0],single[1]])
                    all_channel_smallgroup_loss = all_channel_smallgroup_loss + small_group_loss
                single_weight_loss = single_weight_loss + all_channel_smallgroup_loss
            singlelayer_loss = singlelayer_loss + single_weight_loss

        return singlelayer_loss

