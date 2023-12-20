import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from loss import LossFunction
from model_la import MSPEC_Net
from decomposition import lplas_decomposition as decomposition
from model_la import MSPEC_Net
os.environ['CUDA_VISIBLE_DEVICES']='0'

class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)

        return illu


class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = input - fea

        return delta


class Network(nn.Module):

    def __init__(self, stage=3):
        super(Network, self).__init__()
        self.stage = stage
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        # self.calibrate = evaluate(MSPEC_Net)
        self._criterion = LossFunction()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):

        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input
        for i in range(self.stage):
            inlist.append(input_op)
            i = self.enhance(input_op)
            r = input / i
            r = torch.clamp(r, 0, 1)
            att = evaluate(MSPEC_Net, r)
            input_op = input + att
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist

    def _loss(self, input):
        i_list, en_list, in_list, _ = self(input)
        loss = 0
        for i in range(self.stage):
            loss += self._criterion(in_list[i], i_list[i])
        return loss


class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self._criterion = LossFunction()

        base_weights = torch.load(weights)
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        i = self.enhance(input)
        r = input / i
        r = torch.clamp(r, 0, 1)
        return i, r


    def _loss(self, input):
        i, r = self(input)
        loss = self._criterion(input, i)
        return loss


def exposure_correction(MSPEC_net,data_input):
	if data_input.dtype == 'uint8':	
		data_input = data_input/255	
	_,L_list = decomposition(data_input)	
	L_list = [torch.from_numpy(data).float().permute(2,0,1).unsqueeze(0).cuda() for data in L_list]	
	Y_list = MSPEC_net(L_list)	
	predict = Y_list[-1].squeeze().permute(1,2,0).detach().cpu().numpy()	
	return predict


def down_correction(MSPEC_net,data_input):	
	maxsize = max([data_input.shape[0],data_input.shape[1]])	
	insize = 512
	
	scale_ratio = insize/maxsize
	im_low = cv2.resize(data_input,(0, 0), fx=scale_ratio, fy=scale_ratio,interpolation = cv2.INTER_CUBIC)	
	top_pad,left_pad = insize - im_low.shape[0],insize - im_low.shape[1]	
	im = cv2.copyMakeBorder(im_low, top_pad, 0, left_pad, 0, cv2.BORDER_DEFAULT)	
	out = exposure_correction(MSPEC_net,im)	
	out = out[top_pad:,left_pad:,:]	
	final_out = out
	final_out = cv2.resize(final_out,(data_input.shape[1],data_input.shape[0]))	
	return final_out


def evaluate(MSPEC_net,image):
	detached_tensor = image.detach()
	numpy_array = detached_tensor[0].permute(1, 2, 0).cpu().numpy()
	cv2.imwrite('output_image.png', (numpy_array * 255).astype(np.uint8))
	data_input = cv2.imread('output_image.png')
	with torch.no_grad():	
		MSPEC_net = MSPEC_Net().cuda()
		MSPEC_net =torch.nn.DataParallel(MSPEC_net)	
		MSPEC_net.load_state_dict(torch.load('./snapshots/MSPECnet_woadv.pth'))
		MSPEC_net.eval()	
		output_image = down_correction(MSPEC_net,data_input)	
		image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  
		image = image.transpose((2, 0, 1))
		torch_tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0  
		torch_tensor = torch_tensor.to('cuda:0') 
		
		return torch_tensor
