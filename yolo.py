# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # nc表示类别数
        self.no = nc + 5  # 表示每个anchor的输出数
        self.nl = len(anchors)  # 表示detection layers的数量
        self.na = len(anchors[0]) // 2  # 表示anchors的数量
        self.grid = [torch.zeros(1)] * self.nl  # 初始化为包含self.nl个长度为1的零张量的列表
        # 用于存储锚点信息，并将其转换为张量形式
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        # 用于将anchors和anchor_grid注册为模块的缓冲区（buffer）
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        # self.m是由nn.Conv2d组成的模块列表
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # 用于存储推理输出结果
        # 将训练模式和ONNX导出模式进行逻辑或运算，以确定当前模式
        self.training |= self.export

        for i in range(self.nl):
            # 对于每个检测层，通过使用self.m[i]对输入x[i]进行卷积操作。
            x[i] = self.m[i](x[i])
            # 获取x[i]的维度
            bs, _, ny, nx = x[i].shape
            # 将其重塑为(batch_size, self.na, self.no, ny, nx)的形状，并进行维度重排
            # x(bs,255,20,20) to x(bs,3,20,20,85),
            # 先调用view函数改变形状，然后调用permute调整张量X[i]的维度
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # 如果处于推理模式下
            if not self.training:
                # 检查self.grid[i]的形状是否与x[i]的形状匹配
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # 如果不匹配，则调用_make_grid方法重新生成网格
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                # 对x[i]进行sigmoid激活操作
                y  = x[i].sigmoid()
                # 并计算预测框的坐标和尺寸
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                # 将结果添加到列表z中
                z.append(y.view(bs, -1, self.no))
        # 根据训练模式判断返回值，如果处于训练模式，则返回x；否则返回拼接后的(torch.cat(z, 1), x)
        return x if self.training else (torch.cat(z, 1), x)

    # 生成指定大小的二维网格坐标
    @staticmethod
    def _make_grid(nx=20, ny=20):
        # nx和ny是可选参数，表示x和y方向上的网格数量，默认值为20
        # torch.arange(ny)和torch.arange(nx)分别生成0到ny-1和0到nx-1的整数序列
        # torch.meshgrid函数用于生成x和y方向上的坐标网格。
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        # 将x和y坐标堆叠在一起，并增加一个维度
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # 获取输入通道数，并将其保存到配置文件中
        if nc and nc != self.yaml['nc']:  # 如果指定了类别数并且与配置文件中的类别数不同
            # 打印日志信息，覆盖配置文件中的类别数
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            # 更新配置文件中的类别数
            self.yaml['nc'] = nc
        # 锚点信息
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            # 更新配置文件中的锚点信息，四舍五入为整数
            self.yaml['anchors'] = round(anchors)
        # 根据配置文件解析模型结构，得到模型和保存输出的层列表
        # 调用parse_model函数传递相关参数，生成模型于保存输出的层列表信息
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        # 默认类别名称列表
        self.names = [str(i) for i in range(self.yaml['nc'])]
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # 获取模型的最后一层检测器(Detect)
        m = self.model[-1]
        # 如果最后一层是检测器
        if isinstance(m, Detect):
            # 2倍的最小步长
            s = 256
            # 进行前向传播动态得到模型的步长
            # 第一步：创建一个形状为 (1, ch, s, s) 的全零张量，其中 ch、s 分别表示通道数和输入图像的尺寸
            # 将获得的张良传递给前向传播forward函数（从前向传播计算图中获取当前的计算图的输出结果）
            # 对模型的输出结果进行遍历，计算每个特征图相对于输入图像的缩放比例，得到一个列表。
            # 遍历得到的列表x，其中x.shape[-2] 表示特征图的高度。
            # 因此，通过计算s/x.shape[-2]得到特征图相对于输入图像的高度缩放比例。
            # 将上面计算出的缩放比例列表转换为张量
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            # 根据步长调整锚点
            m.anchors /= m.stride.view(-1, 1, 1)
            # 检查锚点顺序是否正确
            check_anchor_order(m)
            # 保存模型的步长
            self.stride = m.stride
            # 初始化权重和偏置
            self._initialize_biases()
            # print('Strides: %s' % m.stride.tolist())

        # 初始化权重和偏置
        initialize_weights(self)
        self.info()
        logger.info('')

    # 前向传播方法，接受输入张量和一些控制参数作为参数
    def forward(self, x, augment=False, profile=False):
        # 如果进行数据增强
        if augment:
            # 获取输入图像的尺寸
            img_size = x.shape[-2:] 
            # 获取输入图像的尺寸
            s = [1, 0.83, 0.67]
            # 不同翻转方式
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            # 输出列表
            y = []
            # 遍历缩放比例和翻转方式
            for si, fi in zip(s, f):
                # 对输入图像进行缩放和翻转
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                # 进行单尺度推理
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud

                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            # 将不同尺度的输出结果合并返回，用于数据增强时的推理和训练
            return torch.cat(y, 1), None
        else:
            # 进行单尺度推理，返回输出结果
            return self.forward_once(x, profile)

    # 单尺度推理方法，接受输入张量和是否进行性能分析作为参数
    def forward_once(self, x, profile=False):
        # 输出列表和时间列表
        y, dt = [], []
        # 遍历模型中的每一层
        for m in self.model:
            # 如果当前层不是来自前一层
            if m.f != -1:
                # 从之前的层中，获取当前层的输入
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f] 
            # 如果进行性能分析
            if profile:
                # 计算当前层的FLOPS
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                # 获取当前时间
                t = time_synchronized()
                # 运行10次前向传播
                for _ in range(10):
                    _ = m(x)
                # 计算前向传播时间
                dt.append((time_synchronized() - t) * 100)
                # 打印性能分析结果
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))
            # 调用m层执行一次前向传播计算
            x = m(x)
            # 将输出结果保存到列表中
            y.append(x if m.i in self.save else None)

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    # 对偏置项进行合理的初始化
    # 对于目标检测的最后一层（Detect() 模块），针对每个卷积层的偏置项进行了初始化
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # 获取模型中的最后一个模块，通常是目标检测任务中的Detect()模块
        m = self.model[-1]
        # 遍历最后一层模块中的卷积层和对应的步长
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


# 将模型字典d和输入通道数ch作为参数。
# 函数通过迭代模型字典中的每个模块，构建模型的层次结构，
# 并返回一个nn.Sequential对象和一个保存列表。
def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    # 获取模型字典中的锚点、类别数、深度倍数和宽度倍数等信息
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # 计算锚点数目，如果anchors是列表，则取第一个元素的长度除以2，否则直接使用anchors的值作为锚点数目
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    # 计算输出的通道数，即锚点数目乘以（类别数+5）
    no = na * (nc + 5)
    # 初始化layers（存储模型层）、save（用于保存需要的层索引）和c2（输出通道数）
    layers, save, c2 = [], [], ch[-1]

    # 迭代模型字典中的每个元素，包括backbone和head的元素
    # f是模块的来源索引，n是模块的数量，m是模块类型，args是模块参数
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        # 如果m是字符串，则使用eval函数将其转换为对应的模块类型，否则直接使用m本身
        m = eval(m) if isinstance(m, str) else m
        # 对args列表中的每个元素进行遍历，如果元素是字符串，则使用eval函数将其转换为对应的数据类型，否则保持不变
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        # 计算模块数量n的深度增益，根据深度倍数gd进行调整，取最大值为1
        n = max(round(n * gd), 1) if n > 1 else n

        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3, C3TR]:
            # 获取输入通道数c1和输出通道数c2，分别为ch[f]和args[0]
            c1, c2 = ch[f], args[0]
            # 如果c2不等于输出通道数no，则将c2乘以宽度倍数gw并进行可分割调整
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            # 将参数列表args重新组合，包括c1、c2和剩余的参数元素
            args = [c1, c2, *args[1:]]
            # 如果模块类型是BottleneckCSP、C3或C3TR之一，则在参数列表args的索引2处插入n的值，并将n设置为1
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats
                n = 1
        # 如果模块类型是nn.BatchNorm2d，则参数列表args只包含输入通道数ch[f]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        # 如果模块类型是Concat，则将输入通道数c2设为f中所有元素对应的输入通道数之和
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        # 如果模块类型是Detect，则将f中每个元素对应的输入通道数添加到参数列表args中
        # 如果args的第二个元素是整数，则将其转换为一个包含一组锚点索引的列表，并重复len(f)次
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        # 如果模块类型是Contract，则将输入通道数c2设为f对应的输入通道数乘以args的第一个元素的平方
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        # 如果模块类型是Expand，则将输入通道数c2设为f对应的输入通道数除以args的第一个元素的平方
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        # 对于其他模块类型，将输入通道数c2设为f对应的输入通道数
        else:
            c2 = ch[f]
        # 根据模块类型和参数构建模型层，如果n大于1，则使用nn.Sequential将模块重复n次，否则直接使用模块m
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        # 获取模块类型的字符串表示，去除开头的"__main__."并去除末尾的"()"
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # 计算模型层的参数数量，遍历模型层中的所有参数并累加元素数量
        np = sum([x.numel() for x in m_.parameters()])  # number params
        # 将模型层的索引、来源索引、类型和参数数量附加到模型层的属性上
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        # 使用logger记录模型层的信息，包括索引、来源索引、数量、类型和参数等
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        # 将需要保存的层索引添加到save列表中，如果f是整数，则将其添加到save列表中，否则遍历f的每个元素，过滤掉值为-1的元素，并将其添加到save列表中
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        # 将构建的模型层添加到layers列表中
        layers.append(m_)
        # 如果是第一个模块，则清空输入通道数列表ch，然后将输出通道数c2添加到ch中
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
