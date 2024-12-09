import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None):
    # 输出和保存超参数信息，并将其他训练选项保存到变量中
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # 创建用于保存模型权重和训练结果的目录和文件
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # 超参数字典hyp保存到文件中
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    # 将命令行参数字典opt保存到文件中，vars(opt)将opt对象转换为字典
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # 配置训练环境，包括创建绘图和初始种子，以及加载数据集的信息。
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    # 初始化随机种子，随机种子的值为2加上排名rank。
    # rank是在多GPU训练中指定每个GPU的ID，单GPU训练中rank的值为0。
    init_seeds(2 + rank)
    # 打开数据集配置文件opt.data
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    # 根据数据集文件的命名，判断是否使用的是coco数据集
    is_coco = opt.data.endswith('coco.yaml')

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        # 检查权重文件是否存在
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')  # 是否使用预训练权重
    if pretrained:
        # 加载预训练权重
        with torch_distributed_zero_first(rank):  # 查看本地是否有预训练权重
            attempt_download(weights)  # download if not found locally
        # 加载预训练权重文件到加载的模型所在的设备
        ckpt = torch.load(weights, map_location=device)
        # 调用Model方法，创建模型
        # 参数包括模型配置信息，输入图像的通道数为3，模型的类别数量为nc，anchors=hyp.get('anchors')表示锚框的配置。
        model = Model(opt.cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create 创建新模型

        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        # 将预训练权重文件中的模型参数与新创建的模型的状态字典进行交集运算，并排除指定的键。
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        # 加载交集后的模型参数。strict=False表示允许加载不完全匹配的键，即允许一部分参数不匹配。
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check

    # 训练和验证数据集的路径分别赋值给train_path和test_path变量
    train_path = data_dict['train']
    test_path = data_dict['val']

    # Freeze 固定那些层的参数
    freeze = []  # 包含需要冻结的层的全名或部分名称
    # model.named_parameters()返回一个由参数名称和参数本身组成的生成器
    for k, v in model.named_parameters():
        # 将所有层的梯度设置为可训练
        v.requires_grad = True
        # 检查参数名称中是否存在需要冻结的字符串，如果存在，则将梯度设置为不可训练。
        if any(x in k for x in freeze):
            # 输出被冻结的参数的名称。
            print('freezing %s' % k)
            # 将其梯度设置为不可训练
            v.requires_grad = False

    # 设置优化器的参数
    nbs = 64  # nominal batch size
    # accumulate表示在进行优化之前累积损失的次数。它的值被设置为nbs / total_batch_size的四舍五入结果，最小值为1
    accumulate = max(round(nbs / total_batch_size), 1)
    # 超参数中的权重衰减,通过缩放权重衰减，使其与实际使用的批量大小相关联
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            # pg2用于存储所有层的偏置项
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            # pg0用于存储所有批归一化层的权重，这些权重不会被进行权重衰减（weight decay）；
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            # pg1用于存储所有卷积层的权重，这些权重会被进行权重衰减
            pg1.append(v.weight)

    #  优化所有BN层的W
    if opt.adam:
        # 使用给定的学习率(hyp['lr0'])和动量参数(hyp['momentum'])，以及调整后的beta1参数(betas=(hyp['momentum'], 0.999))来初始化Adam优化器
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        # 使用给定的学习率(hyp['lr0'])、动量参数(hyp['momentum'])和Nesterov动量(nesterov=True)来初始化SGD优化器
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # 权重衰减只作用与卷积层的W参数，所以必须将参数分组
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    # 学习率衰减策略，以下是两种变化策略
    if opt.linear_lr:
        # 使用线性学习率调度器
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        # 余弦退火调度器
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']

    # lr_scheduler.LambdaLR()方法创建了一个LambdaLR类实例scheduler，用于动态调整优化器的学习率
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA 指数移动平均，在每次更新参数的时，考虑历史值对参数的影响，给训练带来帮助
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # 恢复训练
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # 加载之前保存的优化器状态和最佳性能指标
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        # 如果使用指数移动平均模型EMA（ema对象存在）并且之前保存的检查点包含EMA状态（ckpt.get('ema')为True），则将EMA状态加载到EMA对象中
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # 则将训练结果写入指定的文件results.txt

        # Epochs
        # 将起始轮数更新为之前保存的检查点中的轮数加1
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            # 如果opt.resume为True，则要求起始轮数必须大于0，否则抛出异常
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            # 总的训练轮数小于起始轮数，则表示之前的训练已经完成，并进行额外的微调训练。
            # 将总的训练轮数增加之前保存的检查点中的轮数，以便进行额外的微调训练。
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs
        # 删除不再需要的检查点和模型参数状态字典对象，以释放内存空间
        del ckpt, state_dict

    # Image sizes
    # 计算模型中最大步长（stride）
    gs = max(int(model.stride.max()), 32)  # grid size (max stride) 获取最高层特征相比于输入图片缩小的倍数
    # 获取模型中最后一个检测层的数量，用于缩放hyp['obj']
    # 在目标检测模型中，检测层通常负责输出检测框的数量或者类别的数量，因此这个数量对于模型的输出非常重要。
    nl = model.model[-1].nl
    # 检查是否满足gs的倍数，不满足则自动补全
    # 将检查过程应用到训练和测试时所使用的图片尺寸中
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]

    # DP mode 是否为多张显卡，是则进行数据并行化操作
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm 分布式训练
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader 创建训练集数据加载器（dataloader），并获取训练集中的最大标签类别数和批次数量
    # 调用create_dataloader()函数来创建自定义数据集和数据加载器
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    # 获取最大标签类别数（mlc）和批次数量（nb）
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0 加载验证集数据
    if rank in [-1, 0]:
        # 自定义验证集数据集和数据加载器
        testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.resume:
            # 取出所有标签并绘制出标签分布图
            #  使用NumPy库中的函数将dataset.labels沿第0轴（行方向）进行连接
            labels = np.concatenate(dataset.labels, 0)
            # 从 labels 数组中选择所有行的第一列数据,并转换成PyTorch张量
            c = torch.tensor(labels[:, 0])    # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors，检查是否合适
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # 将模型放入分布式数据并行环境中，以便在多个GPU上进行训练
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    # 针对模型的超参数 hyp 进行了一系列的缩放操作，以适配网络的层数、类别数量和输入图像尺寸
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    # 设置了标签平滑参数
    hyp['label_smoothing'] = opt.label_smoothing
    # 将类别数量 nc 附加到模型上
    model.nc = nc
    # 将超参数 hyp 附加到模型上
    model.hyp = hyp
    # 设置了交并比（IoU）损失的权重 gr，默认值为1.0，指示了在目标检测损失函数中对象损失和IoU损失的比例。
    model.gr = 1.0
    # 通过调用 labels_to_class_weights 函数计算了类别权重，并将其乘以类别数量后附加到模型上，用于在损失函数中考虑不同类别的权重。
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    # 将类别名称 names 附加到模型上
    model.names = names

    # Start training
    t0 = time.time() # 记录训练开始时间
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # 初始化mAP数组，每个类别一个mAP值
    # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)，初始化结果元组，包括准确率、召回率、mAP等指标
    results = (0, 0, 0, 0, 0, 0, 0)
    # 将学习率调度器的最后一个周期设置为起始周期-1
    scheduler.last_epoch = start_epoch - 1  # do not move
    # 初始化梯度缩放器，用于混合精度训练
    scaler = amp.GradScaler(enabled=cuda)
    # 初始化损失函数计算对象
    compute_loss = ComputeLoss(model)
    # 输出训练相关信息，包括图像尺寸、数据加载器工作线程数、日志保存路径以及训练周期数
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    # epoch ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        # 将模型设置为训练模式
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        # 创建大小4的全零张量,用于存储平均损失的张量
        mloss = torch.zeros(4, device=device)
        if rank != -1:
            # 设置数据加载器的采样器的当前轮次epoch,以确保每个训练周期的数据顺序不同
            train_loader.sampler.set_epoch(epoch)
        # 创建一个用于显示进度的迭代器，并记录日志信息
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar

        # 清除之前批次计算得到的梯度值
        optimizer.zero_grad()
        # 在每个训练周期内，遍历数据加载器的每个批次。
        # batch -------------------------------------------------------------

        for i, (imgs, targets, paths, _) in pbar:
            # 计算当前批次的累计编号，并将图像转移到设备上进行处理。
            ni = i + nb * epoch
            # 将图像数据异步地移动到指定设备上
            # imgs张量的数据类型转换为浮点型，并且将像素值从 [0, 255] 的整数范围归一化到[0, 1]的浮点数范围
            imgs = imgs.to(device, non_blocking=True).float() / 255

            # 在训练初期进行模型参数的预热（warmup）
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                # 根据当前累计批次编号计算累计梯度更新步数（accumulate）和学习率的变化。
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                # 遍历了优化器的参数组（param_groups），对每个参数组进行学习率和动量的更新
                # 在训练开始时逐渐增加学习率和动量的值，从而帮助模型更好地收敛并提高训练效果
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale 多尺度训练
            if opt.multi_scale:
                # 随机选择一个尺度并将图像进行缩放。
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor 随机化获得比例因子
                if sf != 1: # 用获得的比例因子，改变输入的图片尺寸
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward 使用自动混合精度（Automatic Mixed Precision）进行前向传播，并计算损失。
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # 调用scaler对损失值loss进行缩放，以适应混合精度训练所需的数据范围。执行反向传播
            scaler.scale(loss).backward()

            # Optimize 更新参数 ni负责计数，表示取到第几批数据
            if ni % accumulate == 0:
                # 如果达到了累计梯度更新步数，则执行一次梯度更新
                # 使用scaler缩放后的梯度值来更新模型的参数
                scaler.step(optimizer)
                #更新缩放器scaler的缩放因子
                scaler.update()
                # 将优化器optimizer中所有参数的梯度值清零
                optimizer.zero_grad()

                if ema:
                    # 更新模型的移动平均值
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                # 更新平均损失并显示训练进度信息
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    # 保存部分训练结果的图像
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()

                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # 调度器（Scheduler）动态地更新学习率
        # 首先通过列表推导式,获取当前优化器中所有参数组的学习率，并将其存储在列表 lr 中
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        # 更新学习率
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP 指数移动平均模型更新函数 update_attr()，对模型参数进行平均化处理
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            # 判断当前是否为最后一轮训练
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                # 如果未设置不进行测试，并且当前为最后一轮训练
                # 设置WandB日志记录器的当前训练轮次
                wandb_logger.current_epoch = epoch + 1
                # 使用 test.test() 函数对模型进行评估，返回评估结果，包括检测结果、指标值和时间等信息
                results, maps, times = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco)

            # 将评估结果以一定格式写入文件中，方便后续分析和记录
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # 定义要记录的日志标签，包括训练损失、指标和学习率等信息
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params

            # 将训练损失、指标和学习率等信息记录到TensorBoard和WandB中，方便可视化和分析
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP 加权求和获得拟合度值
            # 计算当前评估结果的拟合度值，并更新最佳拟合度值和相应的日志记录器
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model 保存本轮的模型
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # 将当前轮次的训练信息保存到指定位置（last）
                torch.save(ckpt, last)
                if best_fitness == fi:
                    # 如果当前为最佳模型，则将其也保存到指定位置（best）
                    torch.save(ckpt, best)

                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                # 删除字典形式的训练信息变量以释放内存
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})

        # 已完成的轮次数和总共耗时
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

        # 如果数据集是COCO数据集且类别数为80，则进行速度和mAP测试
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else (last):  # speed, mAP tests
                # Test best.pt
                results, _, _ = test.test(opt.data,
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=0.001,
                                          iou_thres=0.7,
                                          model=attempt_load(m, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=True,
                                          plots=False,
                                          is_coco=is_coco)

        # Strip optimizers
        # 根据最佳模型是否存在，确定最终模型（final）
        final = best if best.exists() else last  # final model
        for f in last, best:
            # 对最后模型（last）和最佳模型（best）进行去除优化器操作
            if f.exists():
                strip_optimizer(f)

        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    # 释放GPU缓存
    torch.cuda.empty_cache()
    # 返回评估结果
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/garbage.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    opt = parser.parse_args()

    # 设置分布式训练相关的变量，并进行一些前置的环境检查
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    # Checks rank=-1不进行分布式训练
    if opt.global_rank in [-1, 0]:
        # check_git_status()  # 检验YOLOV5的GitHub是否有更新
        check_requirements() # 检查环境的依赖包

    # Resume 表示从中断中恢复，因为使用的是预训练好的模型所以每一要再使用Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        # 获取要恢复的检查点路径ckpt，可能是用户指定的路径或者最新的检查点路径。
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        # 确保ckpt对应的文件存在
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'

        apriori = opt.global_rank, opt.local_rank
        # 读取opt.yaml文件
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        # 更新opt参数，包括模型配置、权重路径、恢复标志、批大小、全局和本地rank等信息
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        # 打印日志，指示从哪个检查点开始恢复训练
        logger.info('Resuming training from %s' % ckpt)
    else:
        # 检查并更新opt.data、opt.cfg和opt.hyp等文件路径
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)
        # 确保opt.cfg或opt.weights至少有一个被指定
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        # 根据情况扩展opt.img_size列表的长度至2
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)

        opt.name = 'evolve' if opt.evolve else opt.name
        # 增加保存路径
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # 分布式训练设置
    # opt.total_batch_size赋值给opt.batch_size，因为在分布式训练中，每个GPU处理的batch size应该是总batch size除以GPU数量
    opt.total_batch_size = opt.batch_size
    # 调用select_device函数获取当前使用的设备，
    device = select_device(opt.device, batch_size=opt.batch_size)  # 选择驱动
    # 如果opt.local_rank不为-1，表示进行分布式训练，需要将设备更改为cuda类型，并设置当前设备为opt.local_rank指定的GPU。
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        # 初始化分布式后端，使用nccl作为后端，且初始化方法为'env://'
        dist.init_process_group(backend='nccl', init_method='env://')
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        # opt.batch_size是opt.world_size的倍数
        opt.batch_size = opt.total_batch_size // opt.world_size

    # 加载超参数
    with open(opt.hyp) as f:
        # yaml.SafeLoader用于安全地加载yaml文件，防止可能的代码注入。
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    # 开始训练模型
    # 查看当前的训练配置
    logger.info(opt)
    # opt.evolve为False，表示不是进化训练模式
    if not opt.evolve:
        # 初始化日志记录器tb_writer为None
        tb_writer = None
        # 如果opt.global_rank为-1或0，表示当前进程是主进程，即全局排名为0的进程
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            # 创建SummaryWriter对象 tb_writer，指定保存目录为opt.save_dir，用于将训练过程的数据写入Tensorboard
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        # 开始训练
        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit) 超参数进化元数据
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3) 初始化学习率
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf) 最终OneCycleLR学习率
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay 优化器权重衰减
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok) 预热时期（分数可以）
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum  预热初始动量
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr 预热初始偏置
                'box': (1, 0.02, 0.2),  # box loss gain box损失增益
                'cls': (1, 0.2, 4.0),  # cls loss gain cls损耗增益
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight  cls BCE损失正重量
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)  obj损失增益（按像素缩放）
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold 训练阈值
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # 只对最后一个epoch进行测试和保存
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        # 设定超参数优化结果保存的路径和文件名
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        # 基于遗传算法的超参数优化
        for _ in range(300):
            # 在循环开始前，检查是否存在evolve.txt文件，该文件包含之前一些超参数优化的结果。
            # 如果evolve.txt存在，则加载其中记录的过去几代的超参数，并选择其中最优的几组作为父代
            if Path('evolve.txt').exists():
                # 选择父代并进行变异操作,parent变量指定了选择父代的方法 'single' or 'weighted'
                parent = 'single'
                x = np.loadtxt('evolve.txt', ndmin=2)
                # 确定要考虑的过去结果的数量，最多为5
                n = min(5, len(x))
                # 根据超参数的性能对x进行排序，选择性能最好的前n个超参数作为父代。
                x = x[np.argsort(-fitness(x))][:n]
                # w是超参数性能的权重，通过减去最小性能值来计算
                w = fitness(x) - fitness(x).min()
                if parent == 'single' or len(x) == 1:
                    # 如果parent是'single'或x中只有一组超参数，则随机选择其中一组超参数
                    # 使用random.choices函数从前n个超参数中根据权重进行加权随机选择，并将选择的超参数赋值给变量x
                    x = x[random.choices(range(n), weights=w)[0]]

                elif parent == 'weighted':
                    # 进行加权组合，计算加权平均值作为新一代的超参数。
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()

                # 超参数的变异操作
                mp, s = 0.8, 0.2  # mp是变异的概率, s是标准差，用于控制变异的幅度
                # 保证每次运行程序时，使用的随机数序列都是不同的，从而避免重复的随机数序列导致实验结果不可重复。
                npr = np.random
                npr.seed(int(time.time()))
                # g是一个包含超参数变化幅度的数组，取自meta字典中定义的范围
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                # meta字典中超参数的数量
                ng = len(meta)
                # 是一个长度为ng的数组，初始值为1
                v = np.ones(ng)

                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    # 为了控制超参数的取值范围，使用clip函数将变异系数限制在0.3到3.0之间
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                # 使用变异系数v对超参数进行变异
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # 将超参数限制在指定的范围内
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
