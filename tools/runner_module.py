import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils import factory
from utils.logger import *
from utils.data_manager import DataManager
from utils.AverageMeter import AverageMeter
import os
import ipdb
import numpy as np
from datasets import data_transforms
import cv2
from pointnet2_ops import pointnet2_utils
from torchvision import transforms
from tqdm import tqdm

train_transforms = transforms.Compose(
    [   
        # data_transforms.PointcloudScaleAndTranslate(),
        data_transforms.PointcloudRotate(),
    ]
)

test_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

def run_net(args, config, train_writer=None, val_writer=None):
    # cil settings
    init_cls = 0 if config.init_cls == config.increment else config.init_cls
    _set_random(config.seed)
    logger = get_logger(args.log_name)
    # build dataset
    data_manager = DataManager(
        config.dataset.train._base_.NAME,
        config.shuffle,
        config.seed,
        config.init_cls,
        config.increment,
        config,
        _trsf=False,
        join_thelast = True
    )

    config.nb_classes = data_manager.nb_classes # 40
    config.nb_tasks = data_manager.nb_tasks # is 10 incremental phase
    # build model
    # breakpoint()
    model = factory.get_model(config)    
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)
    
    cnn_curve = {"top1": [], "top2": [], "MAcc":[]}
    for task in range(data_manager.nb_tasks):
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()
        
        print(cnn_accy)
        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top2"].append(cnn_accy["top2"])
        cnn_curve["MAcc"].append(cnn_accy["MAcc"])

        print('Accuracy (CNN):', cnn_curve["top1"])
        print("MAcc:          ",cnn_curve["MAcc"])
        print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))


def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_dataloader)):
            points = data[0].cuda()
            label = data[1].cuda()

            points, idx = misc.fps(points, npoints)
            logits = base_model(points, batch_idx=batch_idx)
            target = label.view(-1)
            visualize = False
            if visualize == True:
                task = 'shift'
                np.save(f'./visualization/{task}/target-{batch_idx}', target.detach().cpu().numpy())
            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())


        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)


def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)

    return Acc_Metric(acc)



def test_net(args, config, config_cp):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger) # for finetuned transformer

    cp_model = builder.model_builder(config_cp.model)
    builder.load_model(cp_model, "./ckpts/pretrain.pth", logger = logger)
    cp_model.to(args.local_rank)
    cp_model.eval()
    # base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()
     
    test(base_model, test_dataloader, args, config, logger=logger, cp_model=cp_model)
    
def test(base_model, test_dataloader, args, config, logger = None,cp_model=None):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    #npoints = 1024

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
    
            points = misc.fps(points, npoints)
            #ipdb.set_trace()
            points = points[0]
            cp_feat, image_ori = cp_model(points, eval=True, label=label[0])
            logits, image = base_model(points, cp_feat=cp_feat, args=args)
            
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())
        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] acc = %.4f' % acc, logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

        print_log(f"[TEST_VOTE]", logger = logger)
        acc = 0.
        # for time in range(1, 300):
        #     this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
        #     if acc < this_acc:
        #         acc = this_acc
        #     print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (time, this_acc, acc), logger=logger)
        #print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)

def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)

                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    # print_log('[TEST] acc = %.4f' % acc, logger=logger)
    
    return acc

def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False