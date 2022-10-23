from mmcv.cnn.bricks.norm import build_norm_layer
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from torch import nn
from torch.nn.modules.utils import _pair
import numpy as np

from mmseg.core import add_prefix
from mmseg.ops import resize
from ..builder import HEADS, build_loss
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .decode_head import BaseDecodeHead
#from .bn import InPlaceABNSync as BatchNorm2d
from mmcv.runner import BaseModule

class AttentionRefinementModule(BaseModule):
    """Attention Refinement Module (ARM) to refine the features of each stage.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        x_out (torch.Tensor): Feature map of Attention Refinement Module.
    """

    def __init__(self,
                 in_channels,
                 out_channel,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(AttentionRefinementModule, self).__init__(init_cfg=init_cfg)
        self.conv_layer = ConvModule(
            in_channels=in_channels,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.atten_conv_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None), nn.Sigmoid())

    def forward(self, x):
        x = self.conv_layer(x)
        x_atten = self.atten_conv_layer(x)
        x_out = x * x_atten
        return x_out


class AggregationModule(nn.Module):
    """Aggregation Module"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 aggress_dilation=1,
                 conv_cfg=None,
                 norm_cfg=None):
        super(AggregationModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        new_kernelsize = aggress_dilation*(kernel_size - 1) + 1
        padding = new_kernelsize // 2

        self.reduce_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU'))

        self.t1 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            dilation=aggress_dilation,
            padding=(padding, 0),
            groups=out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.t2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            dilation=aggress_dilation,
            padding=(0, padding),
            groups=out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)

        self.p1 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            dilation=aggress_dilation,
            padding=(0, padding),
            groups=out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.p2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            dilation=aggress_dilation,
            padding=(padding, 0),
            groups=out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        _, self.norm = build_norm_layer(norm_cfg, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward function."""
        x = self.reduce_conv(x)
        x1 = self.t1(x)
        x1 = self.t2(x1)

        x2 = self.p1(x)
        x2 = self.p2(x2)

        out = self.relu(self.norm(x1 + x2))
        return out


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,groups=1,conv_cfg=None,norm_cfg=None,*args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = ConvModule(
            in_chan,
            out_chan,
            ks,
            stride=stride,
            padding=padding,
            groups=groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
            bias = False)

        #self.conv = nn.Conv2d(in_chan,
        #        out_chan,
        #        kernel_size = ks,
        #        stride = stride,
        #        padding = padding,
        #        bias = False)
        # self.bn = BatchNorm2d(out_chan)
        #self.bn = BatchNorm2d(out_chan, activation='none')
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes,groups=1,conv_cfg=None,norm_cfg=None,*args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1,groups=groups,conv_cfg=conv_cfg,norm_cfg=norm_cfg)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

@HEADS.register_module()
class CPHeadPlus_V2(BaseDecodeHead):
    """Context Prior for Scene Segmentation.

    This head is the implementation of `CPNet
    <https://arxiv.org/abs/2004.01547>`_.
    """

    def __init__(self,
                 prior_channels,
                 prior_size,
                 am_kernel_size,
                 aggress_dilation=1,
                 groups=1,
                 loss_prior_decode=dict(type='AffinityLoss', loss_weight=1.0),
                 loss_detail_loss=None,
                 c0_in_channels=-1,
                 c0_channels=0,
                 c1_in_channels=-1,
                 c1_channels=0,
                 detail_index=1,
                 detail_channels=24,
                 arm_channels=-1,
                 seg_head=False,
                 concat_x=True,
                 out_index = 4,
                 **kwargs):
        super(CPHeadPlus_V2, self).__init__(**kwargs)
        self.prior_channels = prior_channels
        self.prior_size = _pair(prior_size)
        self.am_kernel_size = am_kernel_size
        self.detail_index = detail_index
        self.detail_channels = detail_channels
        self.arm_channels = arm_channels
        self.seg_head=seg_head
        self.groups = groups
        self.concat_x = concat_x
        self.out_index = out_index

        self.seg_head_conv = ConvBNReLU(self.num_classes, self.num_classes, ks=3, stride=1, padding=1, groups=self.groups,
                               conv_cfg=self.conv_cfg,
                               norm_cfg=self.norm_cfg)
        if self.arm_channels <0:
            self.bottle_channels=self.in_channels
            self.arm_conv = None
        else:
            self.bottle_channels=arm_channels
            self.arm_conv = AttentionRefinementModule(self.in_channels, arm_channels)



        self.aggregation = AggregationModule(self.in_channels, prior_channels,
                                             am_kernel_size,
                                             aggress_dilation=aggress_dilation,
                                             conv_cfg=self.conv_cfg,
                                             norm_cfg=self.norm_cfg)
        if loss_detail_loss is not None:
            self.conv_out_detail = BiSeNetOutput(self.detail_channels, 64, 1,groups=groups,conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg)
        else:
            self.conv_out_detail = None
        self.prior_conv = ConvModule(
            self.prior_channels,
            np.prod(self.prior_size),
            1,
            padding=0,
            stride=1,
            groups=groups,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.intra_conv = ConvModule(
            self.prior_channels,
            self.prior_channels,
            1,
            padding=0,
            stride=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.inter_conv = ConvModule(
            self.prior_channels,
            self.prior_channels,
            1,
            padding=0,
            stride=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if self.concat_x:
            self.bottleneck = ConvModule(
                self.bottle_channels + self.prior_channels * 2 + c1_channels + c0_channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else :
            self.bottleneck = ConvModule(
                self.prior_channels * 2 + c1_channels + c0_channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        self.loss_prior_decode = build_loss(loss_prior_decode)
        if loss_detail_loss is not None:
           self.loss_detail_loss = build_loss(loss_detail_loss)
        else :
           self.loss_detail_loss = None
        if c0_in_channels > 0 :
            self.c0_bottleneck = ConvModule(
                c0_in_channels,
                c0_channels,
                3,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c0_bottleneck = None
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                3,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None

    def forward(self, inputs): #inputs 基于in_index选择对应层的特征，通过transfrom_input选出来，inputs本身会把4层都传过来
        """Forward function."""
        x = self._transform_inputs(inputs)
        batch_size, channels, height, width = x.size()
        #print("cp_head_inputs[0]" + str(inputs[0].size()))
        #print("cp_head_inputs[1]" + str(inputs[1].size()))
        #print("cp_head_inputs[2]" + str(inputs[2].size()))
        #print("cp_head_inputs[3]" + str(inputs[3].size()))
        #print("cp_head_inputs[4]" + str(inputs[4].size()))
        #print("cp_head_inputs[4]" + str(inputs[5].size()))
        #print("cp_head_inputs[4]" + str(inputs[6].size()))
        #print("cpnet input height and width")
        #print(height)
        #print(width)
        assert self.prior_size[0] == height and self.prior_size[1] == width

        value = self.aggregation(x)

        context_prior_map = self.prior_conv(value)
        context_prior_map = context_prior_map.view(batch_size,
                                                   np.prod(self.prior_size),
                                                   -1)
        context_prior_map = context_prior_map.permute(0, 2, 1)
        context_prior_map = torch.sigmoid(context_prior_map)

        inter_context_prior_map = 1 - context_prior_map

        value = value.view(batch_size, self.prior_channels, -1)
        value = value.permute(0, 2, 1)

        intra_context = torch.bmm(context_prior_map, value)
        intra_context = intra_context.div(np.prod(self.prior_size))
        intra_context = intra_context.permute(0, 2, 1).contiguous()
        intra_context = intra_context.view(batch_size, self.prior_channels,
                                           self.prior_size[0],
                                           self.prior_size[1])
        intra_context = self.intra_conv(intra_context)

        inter_context = torch.bmm(inter_context_prior_map, value)
        inter_context = inter_context.div(np.prod(self.prior_size))
        inter_context = inter_context.permute(0, 2, 1).contiguous()
        inter_context = inter_context.view(batch_size, self.prior_channels,
                                           self.prior_size[0],
                                           self.prior_size[1])
        inter_context = self.inter_conv(inter_context)
        #print("print inter and intra size")
        #print(inter_context.size())
        #print(intra_context.size())
        if self.arm_conv is not None:
            x = self.arm_conv(x)

        if self.concat_x:
           cp_outs = torch.cat([x, intra_context, inter_context], dim=1)
        else:
           cp_outs = torch.cat([intra_context, inter_context], dim=1)


        #print("cpnet cpouts output")
        #print(cp_outs.shape[2:])
        output = cp_outs
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[self.out_index])
            #c1_output = inputs[self.out_index]
            #print("resnet layer6 output")
            #print(c1_output.shape[2:])
            c1_output_resize = resize(
                input=c1_output,
                size=cp_outs.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([c1_output_resize,cp_outs], dim=1)
        if self.c0_bottleneck is not None:
            c0_output = self.c0_bottleneck(inputs[self.detail_index])
            #print("resnet layer1 output")
            #print(c0_output.shape[2:])
            output = resize(
                input=output,
                size=c0_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c0_output], dim=1)
        output = self.bottleneck(output)
        output = self.cls_seg(output)
        if self.loss_detail_loss is not None:
            detail_loss_output = inputs[self.detail_index]
            detail_loss_output = self.conv_out_detail(detail_loss_output)
        else:
            detail_loss_output = None
        #print("cpnet output")
        #print(output.shape[2:])
        return output, context_prior_map, detail_loss_output

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``pam_cam`` is used."""
        return self.forward(inputs)[0]

    def _construct_ideal_affinity_matrix(self, label, label_size):
        scaled_labels = F.interpolate(
            label.float(), size=label_size, mode="nearest")
        scaled_labels = scaled_labels.squeeze_().long()

        scaled_labels[scaled_labels == 4] = self.num_classes
        scaled_labels[scaled_labels == 255] = self.num_classes
        one_hot_labels = F.one_hot(scaled_labels, self.num_classes + 1)
        one_hot_labels = one_hot_labels.view(
            one_hot_labels.size(0), -1, self.num_classes + 1).float()
        ideal_affinity_matrix = torch.bmm(one_hot_labels,
                                          one_hot_labels.permute(0, 2, 1))
        return ideal_affinity_matrix

    def losses(self, seg_logit, seg_label):
        """Compute ``seg``, ``prior_map`` loss."""
        seg_logit, context_prior_map,detail_loss_output = seg_logit
        logit_size = seg_logit.shape[2:]
        logit_size = torch.Size([self.prior_size[0],self.prior_size[1]])
        loss = dict()
        # if self.seg_head == True:
        #     seg_logit = resize(
        #         input=seg_logit,
        #         size=seg_label.shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        #     seg_logit = self.seg_head_conv(seg_logit)

        loss.update(super(CPHeadPlus_V2, self).losses(seg_logit, seg_label))
        prior_loss = self.loss_prior_decode(
            context_prior_map,
            self._construct_ideal_affinity_matrix(seg_label, logit_size))
        if detail_loss_output is not None:
           boundery_bce_loss,boundery_dice_loss = self.loss_detail_loss(detail_loss_output,seg_label.squeeze(1)) #seg_label 为（4，1，1，512，512）
           loss['loss_boundery_bce'] = boundery_bce_loss
           loss['loss_boundery_dice'] = boundery_dice_loss
        loss['loss_prior'] = prior_loss
        return loss
