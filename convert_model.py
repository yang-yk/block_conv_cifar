import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.nn import functional as F
import collections
from torchsummary import summary
from tensor_split import split_4d_tensor,cat_4d_tensor,channel_shuffle


#for transfering layer from conv to fc
class Reshape(nn.Module):
    def __init__(self, name='reshape layer'):
        super(Reshape, self).__init__()
        self.name = name
    def forward(self, x):
        return torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3]))

class Block_Conv2d(nn.Conv2d):
    def __init__(self, input_shape, output_shape, layer_class, split_group):
        self.input_shape = input_shape
        self.output_shape = output_shape
        #self.layer_class = layer_class.__init__()
        self.split_group = split_group
        print('split group',self.split_group)

        super(Block_Conv2d, self).__init__(
            in_channels=layer_class.in_channels, out_channels = layer_class.out_channels, kernel_size = layer_class.kernel_size, stride = layer_class.stride, \
            padding = layer_class.padding, dilation = layer_class.dilation, groups = self.split_group[1], bias = True, padding_mode = layer_class.padding_mode)

        #self.weight = layer_class.weight
        #self.bias = layer_class.bias

        self.weight = torch.nn.init.kaiming_uniform_(self.weight)
        print(self.weight.shape)
        self.chanel_shuffle_id = torch.randperm(output_shape[1])
        #print(self.chanel_shuffle_id)

    def conv2d_forward(self, input):
        return F.conv2d(input, self.weight, bias=self.bias, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)


    def forward(self,input):
        assert len(input.shape) == 4
        split_input_hw = split_4d_tensor(input,self.split_group)

        output_split = []
        for input_h in split_input_hw:
            output_h = []
            for input_hw in input_h:
                output_hw =  self.conv2d_forward(input_hw)
                #output_hw = input_hw
                #print('output hw shape',output_hw.shape)
                output_h.append(output_hw)
            output_split.append(output_h)
        output_cat = cat_4d_tensor(output_split)
        output_cat = channel_shuffle(output_cat,self.chanel_shuffle_id)
        #output_test = self.conv2d_forward(input)
        #test_result = torch.equal(output_cat,output_cat)
        #print('<<<<<weight shape:>>>>>',self.weight.shape)
        #print('<<<<<output_shape:>>>>>',output_cat.shape)
        return output_cat

def get_split_group(layer_input_shape,layer_output_shape,num_neuron_per_core = 256):
    input_batch_size = layer_input_shape[0]
    input_c = layer_input_shape[1]
    input_h = layer_input_shape[2]
    input_w = layer_input_shape[3]

    output_batch_size = layer_output_shape[0]
    output_c = layer_output_shape[0]
    output_h = layer_output_shape[1]
    output_w = layer_output_shape[2]

    assert input_batch_size == output_batch_size

    #8*8*8 block
    split_input_c = int(np.ceil(input_c/8.))
    split_input_h = int(np.ceil(input_h/8.))
    split_input_w = int(np.ceil(input_w/8.))

    final_split_group = [0,split_input_c,split_input_h,split_input_w]
    #final_split_group = [0,1,1,1]

    return final_split_group

def convert_conv_layer(layer,model_input,model_output,model_layer_id):
    if type(layer).__name__ == 'Conv2d':
       layer_id = model_layer_id[layer]
       layer_input_shape = model_input[layer_id]
       layer_output_shape = model_output[layer_id]

       split_group = get_split_group(layer_input_shape, layer_input_shape)
       layer = Block_Conv2d(layer_input_shape,layer_output_shape,layer,split_group)
       return layer
    else:
        if type(layer).__name__ == 'AdaptiveAvgPool2d':
           print('AdaptiveAvgPool2d')
        return layer

def convert_model(model,model_input,model_output,model_layer_id):
    new_model_layer_list = []
    print('start converting conv layer')
    for layer in model.features:
        new_model_layer_list.append(convert_conv_layer(layer,model_input,model_output,model_layer_id))

    #new_model_layer_list.append(model.avgpool)
    Reshape_layer = Reshape('Reshape layer')
    new_model_layer_list.append(Reshape_layer)

    for layer in model.classifier:
        new_model_layer_list.append(layer)

    new_model = nn.Sequential()
    for id,layer in enumerate(new_model_layer_list):
        new_model.add_module(str(id),layer)
    return new_model



'''
#converting model test
model_name = 'vgg16'
pretrained = False
batch_size = 1
input_shape = [batch_size,3,224,224]
#input_shape = [batch_size,64,224,224]
input = torch.from_numpy(np.zeros(input_shape).astype(np.float32))

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# create model
if pretrained:
   print("=> using pre-trained model '{}'".format(model_name))
   model = models.__dict__[model_name](pretrained=True)
else:
   print("=> creating model '{}'".format(model_name))
   model = models.__dict__[model_name]()

#summary(model,input_size=(3,224,224))

model_layer_org = []
for fea_layer in model.features:
    #print(fea_layer)
    model_layer_org.append(fea_layer)

model_layer_org.append(model.avgpool)

for cls_layer in model.classifier:
    #print(fea_layer)
    model_layer_org.append(cls_layer)

model_layer_id = {}
for id, layer in enumerate(model_layer_org):
    model_layer_id[layer] =id

model_input = []
model_output = []

fc_flag = True
layer_input = input
for id,layer in enumerate(model_layer_org):
    if type(layer).__name__ == 'Linear' and fc_flag:
        fc_flag = False
        layer_input = torch.reshape(layer_input,[-1,layer_input.shape[1]*layer_input.shape[2]*layer_input.shape[3]])
        #print(input.shape)
    #model_input[layer] = input.shape
    model_input.append(layer_input.shape)
    layer_output = layer(layer_input)
    layer_input = layer_output
    #print(output.shape)
    #model_output[layer] = output.shape
    model_output.append(layer_output.shape)

new_model = convert_model(model,model_input,model_output,model_layer_id)
output = new_model(input)
#summary(new_model,input_size=(3,224,224))
print('over')
'''
