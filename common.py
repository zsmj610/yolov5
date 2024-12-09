# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh
from utils.plots import color_list, plot_one_box
from utils.torch_utils import time_synchronized


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# Depthwise convolution 创建一个深度卷积层
def DWConv(c1, c2, k=1, s=1, act=True):
    """
        Args:
            c1:表示输入通道数。
            c2: 表示输出通道数。
            k: 表示卷积核大小，默认为1。
            s: 表示卷积步长，默认为1。
            act: 表示是否使用激活函数，默认为True
    """
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        '''
            c1 表示输入通道数。
            c2 表示输出通道数。
            k 表示卷积核大小，默认为1。
            s 表示卷积步长，默认为1。
            p 表示卷积填充大小，默认为None。
            g 表示分组卷积中的组数，默认为1。
            act 表示是否使用激活函数，默认为True。可以是bool类型的值，也可以是nn.Module类型的实例
        '''
        # 建立了一个卷积层 conv
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        # 定义了一个nn.BatchNorm2d
        self.bn = nn.BatchNorm2d(c2)
        # 如果act为True，即如果需要使用默认的SiLU激活函数，则将self.act设置为nn.SiLU()，SiLU是一种整流线性单元的激活函数。
        # 如果act不为True，那么会检查act是否是nn.Module类型的实例，
        # 如果是，则将self.act设置为act所表示的激活函数；
        # 如果不是，则将self.act设置为nn.Identity()，这意味着不使用任何激活函数
        if act is True:
            self.act = nn.SiLU()
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    # 函数定义了前向传播的操作
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    # 用于融合卷积和激活函数的前向传播操作
    def fuseforward(self, x):
        # 对输入张量 x 进行卷积操作-->将卷积后的结果通过激活函数 act 进行激活-->返回激活后的结果，不进行批归一化操作。
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        '''
            c1、c2分别表示输入通道数和输出通道数。
            shortcut 表示是否使用shortcut连接，默认为True。
            g 表示分组卷积的组数，默认为1。
            e 表示扩展因子，默认为0.5
        '''
        # 计算了一个中间隐层的通道数 c_
        c_ = int(c2 * e)
        # 调用Conv卷积模块，建立了2个卷积层 cv1 和 cv2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        # 根据c1与c2是否相等的布尔值与shortcut进行且运算的条件判断，确定是否使用shortcut连接
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # self.add为True，则将输入张量和两次卷积操作的结果相加。
        # 如果不满足这个条件，则直接返回两次卷积操作的结果。
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        '''
            c1：输入通道数。
            c2：输出通道数。
            n：Bottleneck模块重复次数。
            shortcut：是否使用shortcut连接。
            g：分组卷积的组数。
            e：expansion ratio，Bottleneck中hidden channels的扩展因子
        '''
        # 首先计算了一个中间隐层的通道数 c_，c_为输出通道数与扩展银子乘积取整。
        c_ = int(c2 * e)
        # 然后调用Conv卷积模块，建立了3个卷积层 cv1、cv2 和 cv3，，创建分别对应三个卷积操作。
        self.cv1 = Conv(c1, c_,1, 1)
        self.cv2 = Conv(c_, c_,1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        # 建立了一个包含n个Bottleneck模块的nn.Sequential对象
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat(self.m(self.cv1(x)), self.cv2(x), dim=1))

class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        '''
            c1 表示输入通道数。
            c2 表示输出通道数。
            k 是一个元组，表示需要进行最大池化的不同核大小，默认为(5, 9, 13)
        '''
        # 隐藏通道数c_，将输入通道数c1除以2取整
        c_ = c1 // c2
        # 用于将输入通道数c1缩减为c_
        self.cv1 = Conv(c1, c_, 1, 1)
        # 将通道数从c_ * (len(k) + 1)缩减为输出通道数c2
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        # 创建了一个nn.ModuleList，其中包含了多个最大池化层实例。
        # 每个最大池化层的核大小由k中的元素指定。使用列表推导式和循环创建最大池化层
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        # 将输入张量x通过第一个卷积层cv1，将通道数从c1缩减为c_
        x = self.cv1(x)
        # 使用列表推导式和循环，对每个最大池化层实例m都对输入x进行池化操作，并将结果存储在列表中。
        # 将列表中的所有结果和原始的x张量按照通道维度拼接起来，得到一个新的张量。
        # 将拼接后的张量通过第二个卷积层cv2，将通道数从c_ * (len(k) + 1)缩减为输出通道数c2
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        '''
            c1、c2分别表示输入通道数和输出通道数。
            k 表示卷积核大小，默认为1。
            s 表示卷积步长，默认为1。
            p 表示填充大小，默认为None。
            g 表示分组卷积的组数，默认为1。
            act 表示是否使用激活函数，默认为True。
        '''
        # 调用Conv方法创建Focus的卷积操作
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # 将输入张量按照不同的索引切片，得到四个子张量，并在通道维度上进行拼接。
        # 将输入张量按照不同的索引切片，得到四个子张量，并在通道维度上进行拼接。
        # x[..., ::2, ::2] 获取所有行和列索引都为偶数的像素；
        # x[..., 1::2, ::2] 获取所有行索引为奇数、列索引为偶数的像素；
        # x[..., ::2, 1::2] 获取所有行索引为偶数、列索引为奇数的像素；
        # x[..., 1::2, 1::2] 获取所有行和列索引都为奇数的像素。
        # 将拼接得到的张量输入到 conv 中进行卷积操作。
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        # gain表示压缩倍数，默认为2
        self.gain = gain

    def forward(self, x):
        # 获取输入张量 x 的形状信息，并将其分别赋值给 N、C、H、W
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        # 使用 self.gain 得到压缩倍数 s
        s = self.gain
        # 将输入张量 x 进行视图变换，将宽高进行压缩。
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        # 调整维度顺序，将通道维度放在前面。
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        # 将张量形状进行改变，并返回结果。
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        # dimension 表示拼接的维度，默认为1
        self.d = dimension

    def forward(self, x):
        # 使用 torch.cat 函数沿着指定维度 self.d 拼接张量，并返回结果
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                img.show(self.files[i])  # show
            if save:
                f = self.files[i]
                img.save(Path(save_dir) / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp')  # increment save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.display(save=True, save_dir=save_dir)  # save results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
