import pdb
import os
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1, padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bnAfter, bnWidth):
        """
        a basic conv in RANet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bnAfter: the location of batch Norm
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bnAfter is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Conv2d(
                nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))
            if type == 'normal':
                layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                    stride=1, padding=1, bias=False))
            elif type == 'down':
                layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                    stride=2, padding=1, bias=False))
            else:
                raise ValueError
            layer.append(nn.BatchNorm2d(nOut))
            layer.append(nn.ReLU(True))        
        
        else:        
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.BatchNorm2d(nIn))
            layer.append(nn.ReLU(True))
            layer.append(nn.Conv2d(
                nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))        
            if type == 'normal':
                layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                    stride=1, padding=1, bias=False))
            elif type == 'down':
                layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                    stride=2, padding=1, bias=False))
            else:
                raise ValueError
        
        self.net = nn.Sequential(*layer)

    def forward(self, x):
        return self.net(x)


class ConvUpNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2, compress_factor, down_sample):
        '''
        The convolution with normal and up-sampling connection.
        '''
        super(ConvUpNormal, self).__init__()
        self.conv_up = ConvBN(nIn2, math.floor(nOut*compress_factor), 'normal',
                                bottleneck, bnWidth2)
        if down_sample:
            self.conv_normal = ConvBN(nIn1, nOut-math.floor(nOut*compress_factor), 'down',
                                    bottleneck, bnWidth1)
        else:
            self.conv_normal = ConvBN(nIn1, nOut-math.floor(nOut*compress_factor), 'normal',
                                    bottleneck, bnWidth1)
 
    def forward(self, x):
        res = self.conv_normal(x[1])
        _,_,h,w = res.size()
        res = [F.interpolate(x[1], size=(h,w), mode = 'bilinear', align_corners=True),
               F.interpolate(self.conv_up(x[0]), size=(h,w), mode = 'bilinear', align_corners=True),
               res]
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        '''
        The convolution with normal connection.
        '''
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal',
                                   bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0], self.conv_normal(x[0])]
        return torch.cat(res, dim=1)


class _BlockNormal(nn.Module):
    def __init__(self, num_layers, nIn, growth_rate, reduction_rate, trans, bnFactor):
        '''
        The basic computational block in RANet with num_layers layers.
        trans: If True, the block will add a transiation layer at the end of the block
                with reduction_rate.
        '''
        super(_BlockNormal, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.layers.append(ConvNormal(nIn + i*growth_rate, growth_rate, True, bnFactor))
        nOut = nIn + num_layers*growth_rate
        self.trans_flag = trans
        if trans:
            self.trans = ConvBasic(nOut, math.floor(1.0 * reduction_rate * nOut), kernel=1, stride=1, padding=0)
        
    def forward(self, x):
        output = [x]
        for i in range(self.num_layers):
            x = self.layers[i](x)
            # print(x.size())
            output.append(x)
        x = output[-1]
        if self.trans_flag:
            x = self.trans(x)
        return x, output
    
    def _blockType(self):
        return 'norm'


class _BlockUpNormal(nn.Module):
    def __init__(self, num_layers, nIn, nIn_lowFtrs, growth_rate, reduction_rate, trans, down, compress_factor, bnFactor1, bnFactor2):
        '''
        The basic fusion block in RANet with num_layers layers.
        trans: If True, the block will add a transiation layer at the end of the block
                with reduction_rate.
        compress_factor: There will be compress_factor*100% information from the previous
                sub-network.  
        '''
        super(_BlockUpNormal, self).__init__()

        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers-1):
            self.layers.append(ConvUpNormal(nIn + i*growth_rate, nIn_lowFtrs[i], growth_rate, True, bnFactor1, bnFactor2, compress_factor, False))

        self.layers.append(ConvUpNormal(nIn + (i+1)*growth_rate, nIn_lowFtrs[i+1], growth_rate, True, bnFactor1, bnFactor2, compress_factor, down))
        nOut = nIn + num_layers*growth_rate

        self.conv_last = ConvBasic(nIn_lowFtrs[num_layers], math.floor(nOut*compress_factor),  kernel=1, stride=1, padding=0)
        nOut = nOut + math.floor(nOut*compress_factor)
        self.trans_flag = trans
        if trans:
            self.trans = ConvBasic(nOut, math.floor(1.0 * reduction_rate * nOut), kernel=1, stride=1, padding=0)
            
    def forward(self, x, low_feat):
        output = [x]
        for i in range(self.num_layers):
            inp = [low_feat[i]]
            inp.append(x)
            x = self.layers[i](inp)
            output.append(x)
        x = output[-1]
        _,_,h,w = x.size()
        x = [x]
        x.append(F.interpolate(self.conv_last(low_feat[self.num_layers]), size=(h,w), mode = 'bilinear', align_corners=True))
        x = torch.cat(x, dim = 1)
        if self.trans_flag:
            x = self.trans(x)
        return x, output

    def _blockType(self):
        return 'up'


class RAFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args):
        '''
        RAFirstLayer gennerates the base features for RANet.
        The scale 1 means the lowest resoultion in the network.
        '''
        super(RAFirstLayer, self).__init__()
        _grFactor = args.grFactor[::-1] # 1-2-4
        _scale_list = args.scale_list[::-1] # 3-2-1
        self.layers = nn.ModuleList()
        if args.dataset.startswith('cifar'):
            self.layers.append(ConvBasic(nIn, nOut * _grFactor[0],
                                         kernel=3, stride=1, padding=1))
        elif args.dataset == 'imagenet':
            conv = nn.Sequential(
                    nn.Conv2d(nIn, nOut * _grFactor[0], 7, 2, 3),
                    nn.BatchNorm2d(nOut * _grFactor[0]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, 2, 1))
            self.layers.append(conv)

        nIn = nOut * _grFactor[0]
        
        s = _scale_list[0]
        for i in range(1, args.nScales):
            if s == _scale_list[i]:
                self.layers.append(ConvBasic(nIn, nOut * _grFactor[i],
                                         kernel=3, stride=1, padding=1))
            else:
                self.layers.append(ConvBasic(nIn, nOut * _grFactor[i],
                                         kernel=3, stride=2, padding=1))
                s = _scale_list[i]
            nIn = nOut * _grFactor[i]

    def forward(self, x):
        # res[0] with the smallest resolutions
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)
        return res[::-1]


class RANet(nn.Module):
    def __init__(self, args):
        super(RANet, self).__init__()
        self.scale_flows = nn.ModuleList()
        self.classifier = nn.ModuleList()
        
        # self.args = args
        self.compress_factor = args.compress_factor
        self.bnFactor = copy.copy(args.bnFactor)

        scale_list = args.scale_list # 1-2-3
        self.nScales = len(args.scale_list) # 3

        # The number of blocks in each scale flow
        self.nBlocks = [0]
        for i in range(self.nScales):
            self.nBlocks.append(args.block_step*i + args.nBlocks) # [0, 2, 4, 6]
        
        # The number of layers in each block
        self.steps = args.step

        self.FirstLayer = RAFirstLayer(3, args.nChannels, args)

        steps = [args.step] 
        for ii in range(self.nScales):

            scale_flow = nn.ModuleList()

            n_block_curr = 1
            nIn = args.nChannels*args.grFactor[ii] # grFactor = [4,2,1]
            _nIn_lowFtrs = []
            
            for i in range(self.nBlocks[ii+1]):
                growth_rate = args.growthRate*args.grFactor[ii]
                
                # If transiation
                trans = self._trans_flag(n_block_curr, n_block_all = self.nBlocks[ii+1], inScale = scale_list[ii])

                if n_block_curr > self.nBlocks[ii]:
                    m, nOuts = self._build_norm_block(nIn, steps[n_block_curr-1], growth_rate, args.reduction, trans, bnFactor=self.bnFactor[ii])
                    if args.stepmode == 'even':
                        steps.append(args.step)
                    elif args.stepmode == 'lin_grow':
                        steps.append(steps[-1]+args.step)
                    else:
                        raise NotImplementedError
                else:
                    if n_block_curr in self.nBlocks[:ii+1][-(scale_list[ii]-1):]:
                        m, nOuts = self._build_upNorm_block(nIn, nIn_lowFtrs[i], steps[n_block_curr-1], growth_rate, args.reduction, trans, down=True, bnFactor1=self.bnFactor[ii], bnFactor2=self.bnFactor[ii-1])
                    else:
                        m, nOuts = self._build_upNorm_block(nIn, nIn_lowFtrs[i], steps[n_block_curr-1], growth_rate, args.reduction, trans, down=False, bnFactor1=self.bnFactor[ii], bnFactor2=self.bnFactor[ii-1])

                nIn = nOuts[-1]
                scale_flow.append(m)
                
                if n_block_curr > self.nBlocks[ii]:
                    if args.dataset.startswith('cifar100'):
                        self.classifier.append(
                        self._build_classifier_cifar(nIn, 100))
                    elif args.dataset.startswith('cifar10'):
                        self.classifier.append(self._build_classifier_cifar(nIn, 10))
                    elif args.dataset == 'imagenet':
                        self.classifier.append(
                        self._build_classifier_imagenet(nIn, 1000))
                    else:
                        raise NotImplementedError
                
                _nIn_lowFtrs.append(nOuts[:-1])
                n_block_curr += 1
                
            nIn_lowFtrs = _nIn_lowFtrs
            self.scale_flows.append(scale_flow)
            
        args.num_exits = len(self.classifier)

        for m in self.scale_flows:
            for _m in m.modules():
                self._init_weights(_m)

        for m in self.classifier:
            for _m in m.modules():
                self._init_weights(_m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_norm_block(self, nIn, step, growth_rate, reduction_rate, trans, bnFactor=2):  
        
        block = _BlockNormal(step, nIn, growth_rate, reduction_rate, trans, bnFactor=bnFactor)
        nOuts = []
        for i in range(step+1):
            nOut = (nIn + i * growth_rate)
            nOuts.append(nOut)
        if trans:
            nOut = math.floor(1.0 * reduction_rate * nOut)
        nOuts.append(nOut)

        return block, nOuts

    def _build_upNorm_block(self, nIn, nIn_lowFtr, step, growth_rate, reduction_rate, trans, down, bnFactor1=1, bnFactor2=2):       
        compress_factor = self.compress_factor

        block = _BlockUpNormal(step, nIn, nIn_lowFtr, growth_rate, reduction_rate, trans, down, compress_factor, bnFactor1=bnFactor1, bnFactor2=bnFactor2)
        nOuts = []
        for i in range(step+1):
            nOut = (nIn + i * growth_rate)
            nOuts.append(nOut)
        nOut = nOut + math.floor(nOut*compress_factor)
        if trans:
            nOut = math.floor(1.0 * reduction_rate * nOut)
        nOuts.append(nOut)

        return block, nOuts

    def _trans_flag(self, n_block_curr, n_block_all, inScale):
        flag = False
        for i in range(inScale-1):
            if n_block_curr == math.floor((i+1)*n_block_all /inScale):
                flag = True
        return flag

    def forward(self, x, stage=None):
        inp = self.FirstLayer(x)
        res, low_ftrs = [], []
        classifier_idx = 0
        for ii in range(self.nScales):
            _x = inp[ii]
            _low_ftrs = []
            n_block_curr = 0
            for i in range(self.nBlocks[ii+1]):
                if self.scale_flows[ii][i]._blockType() == 'norm':
                    _x, _low_ftr = self.scale_flows[ii][i](_x)
                    _low_ftrs.append(_low_ftr)
                else:
                    _x, _low_ftr = self.scale_flows[ii][i](_x, low_ftrs[i])
                    _low_ftrs.append(_low_ftr)
                n_block_curr += 1
                
                if n_block_curr > self.nBlocks[ii]:
                    res.append(self.classifier[classifier_idx](_x))
                    if classifier_idx == stage:
                        return res
                    classifier_idx += 1
                
            low_ftrs = _low_ftrs
        return res

    def _build_classifier_cifar(self, nIn, num_classes):
        interChannels1, interChannels2 = 128, 128
        conv = nn.Sequential(
            ConvBasic(nIn, interChannels1, kernel=3, stride=2, padding=1),
            ConvBasic(interChannels1, interChannels2, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2),
        )
        return ClassifierModule(conv, interChannels2, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        conv = nn.Sequential(
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2)
        )
        return ClassifierModule(conv, nIn, num_classes)

class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.m = m
        self.linear = nn.Linear(channel, num_classes)
    def forward(self, x):
        res = self.m(x)
        res = res.view(res.size(0), -1)
        return self.linear(res)


class LinearBasic(nn.Module):
    def __init__(self, nIn, nOut, dropout=0.1):
        super(LinearBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(nIn, nOut, bias=False),
            nn.BatchNorm1d(nOut),
            nn.ReLU(True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class LinearBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bnAfter, bnWidth, dropout=0.1):
        """
        a basic linear layer in Tabular RANet, two types
        :param nIn: input features
        :param nOut: output features
        :param type: normal or down
        :param bnAfter: the location of batch Norm
        :param bnWidth: bottleneck factor
        """
        super(LinearBN, self).__init__()
        layer = []
        nInner = nIn
        if bnAfter is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Linear(nIn, nInner, bias=False))
            layer.append(nn.BatchNorm1d(nInner))
            layer.append(nn.ReLU(True))
            if type == 'normal':
                layer.append(nn.Linear(nInner, nOut, bias=False))
            elif type == 'down':
                layer.append(nn.Linear(nInner, nOut, bias=False))
            else:
                raise ValueError
            layer.append(nn.BatchNorm1d(nOut))
            layer.append(nn.ReLU(True))        
        
        else:        
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.BatchNorm1d(nIn))
            layer.append(nn.ReLU(True))
            layer.append(nn.Linear(nIn, nInner, bias=False))
            layer.append(nn.BatchNorm1d(nInner))
            layer.append(nn.ReLU(True))        
            if type == 'normal':
                layer.append(nn.Linear(nInner, nOut, bias=False))
            elif type == 'down':
                layer.append(nn.Linear(nInner, nOut, bias=False))
            else:
                raise ValueError
        
        self.net = nn.Sequential(*layer)

    def forward(self, x):
        return self.net(x)


class LinearUpNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2, compress_factor, down_sample, dropout=0.1):
        '''
        The linear layer with normal and up-sampling connection for tabular data.
        '''
        super(LinearUpNormal, self).__init__()
        self.linear_up = LinearBN(nIn2, math.floor(nOut*compress_factor), 'normal',
                                bottleneck, bnWidth2, dropout)
        if down_sample:
            self.linear_normal = LinearBN(nIn1, nOut-math.floor(nOut*compress_factor), 'down',
                                    bottleneck, bnWidth1, dropout)
        else:
            self.linear_normal = LinearBN(nIn1, nOut-math.floor(nOut*compress_factor), 'normal',
                                    bottleneck, bnWidth1, dropout)
 
    def forward(self, x):
        res = self.linear_normal(x[1])
        res = [x[1],
               self.linear_up(x[0]),
               res]
        return torch.cat(res, dim=1)


class TabularRANet(nn.Module):
    def __init__(self, args):
        super(TabularRANet, self).__init__()
        self.args = args
        self.nScales = len(args.scale_list)
        self.nBlocks = args.nBlocks
        self.step = args.block_step
        
        # First layer for tabular data
        self.first_layer = LinearBasic(args.num_features, args.nChannels * args.scale_list[0])
        
        # Build blocks
        self.blocks = nn.ModuleList()
        nIn = args.nChannels * args.scale_list[0]
        
        for i in range(self.nBlocks):
            block = self._build_block(nIn, args)
            self.blocks.append(block)
            nIn = args.nChannels * args.scale_list[-1]  # Output of last scale
        
        # Classifiers
        self.classifiers = nn.ModuleList()
        for i in range(self.nBlocks):
            classifier = nn.Sequential(
                nn.Linear(nIn, nIn // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(nIn // 2, args.num_classes)
            )
            self.classifiers.append(classifier)
    
    def _build_block(self, nIn, args):
        """Build a RANet block for tabular data"""
        layers = []
        nIn_curr = nIn
        
        for i in range(self.step):
            # Create layers for each scale
            scale_layers = []
            for j, scale in enumerate(args.scale_list):
                if j == 0:
                    # First scale
                    layer = LinearBasic(nIn_curr, args.nChannels * scale)
                else:
                    # Other scales
                    layer = LinearUpNormal(
                        nIn_curr, 
                        args.nChannels * args.scale_list[j-1], 
                        args.nChannels * scale,
                        args.bottleneck,
                        args.bnFactor[j-1],
                        args.bnFactor[j],
                        args.compress_factor,
                        False
                    )
                scale_layers.append(layer)
            
            layers.append(ParallelModule(scale_layers))
            nIn_curr = args.nChannels * args.scale_list[-1]
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # First layer
        x = self.first_layer(x)
        x = [x] * self.nScales  # Initialize all scales
        
        # Process through all blocks without early exit
        for i, block in enumerate(self.blocks):
            x = block(x)
        
        # Return only the final classification result
        return self.classifiers[-1](x[-1])


class ParallelModule(nn.Module):
    """Parallel module for tabular data"""
    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))
        return res


if __name__ == '__main__':
    from args_v5 import arg_parser
    from op_counter import measure_model    
    
    args = arg_parser.parse_args()
    # if args.gpu:
    #   os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    args.nBlocks = 2
    args.Block_base = 2
    args.step = 8
    args.stepmode ='even'
    args.compress_factor = 0.25
    args.nChannels = 64
    args.data = 'ImageNet'
    args.growthRate = 16
    
    args.grFactor = '4-2-2-1'
    args.bnFactor = '4-2-2-1'
    args.scale_list = '1-2-3-4'

    args.reduction = 0.5

    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    args.scale_list = list(map(int, args.scale_list.split('-')))
    args.nScales = len(args.grFactor)
    # print(args.grFactor)
    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']
    
    if args.data == 'cifar10':
        args.num_classes = 10
    elif args.data == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 1000

    inp_c = torch.rand(16,3,224,224)

    model = MSDNet(args)
    # output = model(inp_c)
    # oup = net_head(inp_c)
    # print(len(oup))

    n_flops, n_params = measure_model(model, 224, 224) 
    # net = _BlockNormal(num_layers = 4, nIn = 64, growth_rate = 24, reduction_rate = 0.5, trans_down = True)
    
