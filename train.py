#coding=utf-8
import os
import wandb
import shutil
from tqdm import tqdm
from timm.utils import CheckpointSaver
from timm.models import resume_checkpoint

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim

import models
import losses
from dataloaders import create_dataloader
from utils.logger import Logger
from utils.init import setup
from utils.parameters import get_parameters
from utils.misc import *
from utils.metric import cal_metrics
from utils.fps import farthest_point_sample_tensor


import cv2
cv2.setNumThreads(1)

args = get_parameters()
setup(args)
if args.local_rank == 0:
    if args.wandb.name is None:
        args.wandb.name = args.config.split('/')[-1].replace('.yaml', '')
    wandb.init(**args.wandb)
    allow_val_change = False if args.wandb.resume is None else True
    wandb.config.update(args, allow_val_change)
    wandb.save(args.config)
    if len(wandb.run.dir) > 1:
        args.exam_dir = os.path.dirname(wandb.run.dir)
    else:
        args.exam_dir = 'wandb/debug'
        if os.path.exists(args.exam_dir):
            shutil.rmtree(args.exam_dir)
        os.makedirs(args.exam_dir, exist_ok=True)
    logger = Logger(name='train', log_path=f'{args.exam_dir}/train.log')
    logger.info(args)

def load_model(args):
    if args.ckpt is not None:
        print(f'resume model from {args.ckpt}')
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        if getattr(args, 'transform', None) is None:
            args.transform = checkpoint['args'].transform
        if getattr(args, 'model', None) is None:
            args.model = checkpoint['args'].model
        model = models.__dict__[args.model.name](**args.model.params)
        state_dict = checkpoint if args.ckpt.endswith('pth') else checkpoint['state_dict']
        model.load_state_dict(state_dict)
    else:
        assert getattr(args, 'model', None) is not None
        model = models.__dict__[args.model.name](**args.model.params)
    print(args.model)
    return model, args
    
def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def model_forward(image, label, model, apply_shade, apply_wt, cal_covstat):
    outputs_catcls, outputs_catdepth, outputs_catfeat, outputs_shadecls, outputs_shadedepth, outputs_shadefeat = model(image, label, apply_shade, cal_covstat, apply_wt)
    return outputs_catcls, outputs_catdepth, outputs_catfeat, outputs_shadecls, outputs_shadedepth, outputs_shadefeat


def main():
    # Distributed traning
    if args.distributed:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method="env://")
        torch.cuda.set_device(args.local_rank)
        args.world_size = dist.get_world_size()

    # Create dataloader
    train_pos = create_dataloader(args, split='train',category='pos')
    train_neg = create_dataloader(args, split='train',category='neg')
    test_pos = create_dataloader(args, split='val',category='pos')
    test_neg = create_dataloader(args, split='val',category='neg')
    
    epoch_size = min(len(train_pos.dataset),len(train_neg.dataset)) // (args.train.batch_size * args.world_size)

    # Create model
    device = torch.device("cuda", args.local_rank)
    model = models.__dict__[args.model.name](**args.model.params)
    model = model.to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True, broadcast_buffers=False)
        if args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        model = nn.parallel.DataParallel(model)

    optimizer = optim.__dict__[args.optimizer.name](model.parameters(), **args.optimizer.params)
    scheduler = optim.lr_scheduler.__dict__[args.scheduler.name](optimizer, **args.scheduler.params)
    criterion = losses.__dict__[args.loss.name](**args.loss.params).to(device)
    criterion_2 = losses.__dict__[args.loss_2.name](**args.loss_2.params).to(device)
    criterion_3 = losses.__dict__[args.loss_3.name](**args.loss_3.params).to(device)
    criterion_4 = losses.__dict__[args.loss_4.name](**args.loss_4.params).to(device)
    criterion_5 = losses.__dict__[args.loss_5.name](**args.loss_5.params).to(device)
    criterion_6 = losses.__dict__[args.loss_5.name](**args.loss_5.params).to(device)

    # Resume from checkpoint
    start_epoch = 1
    if args.model.ckpt_path is not None:
        start_epoch = resume_checkpoint(model, args.model.ckpt_path, optimizer)
        if args.local_rank == 0:
            logger.info(f'resume model from {args.model.ckpt_path}')
    
    # Traing misc
    saver = None
    if args.local_rank == 0:
        wandb.watch(model, log='all')
        saver = CheckpointSaver(model, optimizer,
                                args=args,
                                checkpoint_dir=f'{args.exam_dir}',
                                recovery_dir=f'{args.exam_dir}',
                                max_history=10)
    
    train_loader_len = min(len(train_pos),len(train_neg))
    p_anneal = ExpAnnealing(0, 1, 0, alpha=args.train.alpha)
    # Training loop
    for epoch in range(start_epoch, args.train.epochs + 1):
        if args.distributed and args.debug == False:
            train_pos.sampler.set_epoch(epoch)
            train_neg.sampler.set_epoch(epoch)
        p = p_anneal.get_lr(epoch)

        # Warmup flag of shade augmentation
        apply_shade = False if epoch < (args.train.proto_select_epoch + 1) else True

        # Warmup flag of whitening loss
        apply_wt = False if epoch < (args.train.cov_stat_epoch + 1) else True

        if apply_shade==True and (epoch % args.train.proto_select_epoch)==0:
            validate_for_prototype(train_pos, train_neg, model, epoch, device, apply_shade, apply_wt)

        if apply_wt==True and (epoch % args.train.cov_stat_epoch)==0: # 2,4,6,8
        # if epoch==1 or ((epoch % (args.train.cov_stat_epoch+1))==0):# 1,3,5,7,9
            model.module.Shader.cov_matrix_layer_real.reset_mask_matrix()
            model.module.Shader.cov_matrix_layer_fake.reset_mask_matrix()
            validate_for_cov_stat(train_pos, train_neg, model, epoch, apply_shade, apply_wt, device)
            model.module.Shader.cov_matrix_layer_real.set_mask_matrix()
            model.module.Shader.cov_matrix_layer_fake.set_mask_matrix()

        train(train_pos, train_neg, model, criterion, criterion_2, criterion_3, criterion_4, criterion_5, criterion_6, p, apply_shade, apply_wt,
            optimizer, wandb, device, epoch, epoch_size, args.train.iteration_gap, test_pos, test_neg, saver, scheduler)

        validate(test_pos, test_neg, model, criterion, criterion_2, criterion_3, criterion_4, criterion_5, criterion_6, p, apply_shade, apply_wt,
                optimizer, scheduler, wandb, saver, device, epoch, iteration=train_loader_len)

        scheduler.step()
        
    if args.local_rank == 0:
        wandb.finish()
        logger.info(args)

def mix_pos_neg(datas_pos, datas_neg):
    img_PB, img_PN, img_PC, img_PH, img_PW = datas_pos[0].shape # torch.Size([2, 3, 6, 256, 256])
    img_NB, img_NN, img_NC, img_NH, img_NW = datas_neg[0].shape # torch.Size([2, 3, 6, 256, 256])

    # Image
    train_pos_img = datas_pos[0].reshape(-1, img_PC, img_PH, img_PW) # torch.Size([2*3, 6, 256, 256])
    train_neg_img = datas_neg[0].reshape(-1, img_NC, img_NH, img_NW) # torch.Size([2*3, 6, 256, 256])

    # Label
    label_PB,_ = datas_pos[1].shape # torch.Size([2, 3])
    label_NB,_ = datas_neg[1].shape # torch.Size([2, 3])

    train_pos_label = datas_pos[1].reshape(-1).long() # torch.Size([2*3])
    train_neg_label = datas_neg[1].reshape(-1).long() # torch.Size([2*3])

    # Depth
    dep_PB, dep_PN, dep_PH, dep_PW = datas_pos[2].shape # torch.Size([2, 3, 32, 32])
    dep_NB, dep_NN, dep_NH, dep_NW = datas_neg[2].shape # torch.Size([2, 3, 32, 32])

    train_pos_depth = datas_pos[2].reshape(-1, dep_PH, dep_PW) # torch.Size([2*3, 32, 32])
    train_neg_depth = datas_neg[2].reshape(-1, dep_NH, dep_NW)

    train_images = torch.cat((train_pos_img, train_neg_img),0)      # torch.Size([12, 6, 256, 256])
    train_targets = torch.cat((train_pos_label, train_neg_label),0) # torch.Size([12])
    train_depth = torch.cat((train_pos_depth, train_neg_depth),0)   # torch.Size([12, 32, 32])

    return train_images, train_targets, train_depth


def preprocess(datas):
    img_B, img_N, img_C, img_H, img_W = datas[0].shape # torch.Size([2, 3, 6, 256, 256])

    # Image
    train_image = datas[0].reshape(-1, img_C, img_H, img_W) # torch.Size([2*3, 6, 256, 256])

    # Label
    label_B,_ = datas[1].shape # torch.Size([2, 3])
    train_label = datas[1].reshape(-1).long() # torch.Size([2*3])

    # Depth
    dep_B, dep_N, dep_H, dep_W = datas[2].shape # torch.Size([2, 3, 32, 32])
    train_depth = datas[2].reshape(-1, dep_H, dep_W) # torch.Size([2*3, 32, 32])
    return train_image, train_label, train_depth

def train(train_pos, train_neg, model, criterion, criterion_2,  criterion_3, criterion_4, criterion_5, criterion_6, p, apply_shade, apply_wt,
          optimizer, wandb, device, epoch, epoch_size, iteration_gap, test_pos, test_neg, saver, scheduler):
    train_acces = AverageMeter('train_Acc', ':.5f')
    train_losses = AverageMeter('train_Class', ':.5f')
    train_losses2 = AverageMeter('train_Depth', ':.5f')
    train_losses3 = AverageMeter('train_Class2', ':.5f')
    train_losses4 = AverageMeter('train_Depth2', ':.5f')
    train_losses5 = AverageMeter('train_AIAW1', ':.5f')
    train_losses6 = AverageMeter('train_AIAW2', ':.5f')

    progress = ProgressMeter(epoch_size, [train_acces, train_losses, train_losses2, train_losses3, train_losses4, train_losses5, train_losses6])
    # progress = ProgressMeter(epoch_size,[train_acces])

    model.train()
    if apply_shade==False:
        for name, param in model.named_parameters():
            if 'CSA_layers' in name:
                param.requires_grad = False
            if 'conv_final' in name:
                param.requires_grad = False
    else: # apply_shade==True
        for name, param in model.named_parameters():
            if 'CSA_layers' in name:
                param.requires_grad = True
            if 'conv_final' in name:
                param.requires_grad = True


    train_loader_len = min(len(train_pos),len(train_neg))
    for i, (datas_pos, datas_neg) in enumerate(zip(train_pos, train_neg)):

        # ============ normal-train ============= #
        train_images, train_targets, train_depth = mix_pos_neg(datas_pos, datas_neg)
        
        train_images = train_images.to(device)
        train_targets = train_targets.to(device)

        train_depth = train_depth.to(device)

        outputs_catcls, outputs_catdepth, outputs_catfeat, outputs_shadecls, outputs_shadedepth, outputs_shadefeat = model_forward(train_images, train_targets, model, apply_shade, apply_wt, cal_covstat=False)

        batch_num, _ = outputs_catcls["out"].shape
        prediction = outputs_catcls["out"].argmax(dim=1)
        output_catcls = outputs_catcls["out"]
        output_catdepth = outputs_catdepth["out"]
        loss_1 = criterion(output_catcls, train_targets)
        loss_2 = criterion_2(output_catdepth.squeeze(), train_depth)
        
        if apply_shade==True:
            batch_num, _ = outputs_shadecls["out"].shape
            prediction = outputs_shadecls["out"].argmax(dim=1)
            output_augcls = outputs_shadecls["out"]
            output_augdepth = outputs_shadedepth["out"]
            loss_3 = criterion_3(output_augcls, train_targets)
            loss_4 = criterion_4(output_augdepth.squeeze(), train_depth)   
        if apply_wt==True:
            loss_5 = criterion_5(outputs_shadefeat["org_feat_real"], outputs_shadefeat["org_feat_fake"], 
                                 outputs_shadefeat["eye_real"], outputs_shadefeat["eye_fake"], 
                                 outputs_shadefeat["mask_matrix_real"], outputs_shadefeat["mask_matrix_fake"],
                                 outputs_shadefeat["margin_real"], outputs_shadefeat["margin_fake"],
                                 outputs_shadefeat["num_remove_cov_real"], outputs_shadefeat["num_remove_cov_fake"])   
            loss_6 = criterion_6(outputs_shadefeat["aug_feat_real"], outputs_shadefeat["aug_feat_fake"],
                                 outputs_shadefeat["eye_real"], outputs_shadefeat["eye_fake"],
                                 outputs_shadefeat["mask_matrix_real"], outputs_shadefeat["mask_matrix_fake"], 
                                 outputs_shadefeat["margin_real"], outputs_shadefeat["margin_fake"],
                                 outputs_shadefeat["num_remove_cov_real"], outputs_shadefeat["num_remove_cov_fake"],)        
        # all the loss
        train_loss = loss_1 + loss_2 * args.loss_2.weight
        if apply_shade==True:
            train_loss = train_loss + loss_3 * args.loss_3.weight + loss_4 * args.loss_4.weight
        if apply_wt==True:
            train_loss = train_loss + loss_5 * args.loss_5.weight + loss_6 * args.loss_6.weight

        acc = (prediction == train_targets).float().mean()

        if args.distributed:
            train_acces.update(reduce_tensor(acc.data).item(), train_targets.size(0))
            train_losses.update(reduce_tensor(loss_1.data).item(), train_targets.size(0))
            train_losses2.update(reduce_tensor(loss_2.data).item(), train_targets.size(0))
            if apply_shade==True:
                train_losses3.update(reduce_tensor(loss_3.data).item(), train_targets.size(0))
                train_losses4.update(reduce_tensor(loss_4.data).item(), train_targets.size(0))
            if apply_wt==True:
                train_losses5.update(reduce_tensor(loss_5.data).item(), train_targets.size(0))
                train_losses6.update(reduce_tensor(loss_6.data).item(), train_targets.size(0))
        else:
            train_acces.update(acc.item(), train_targets.size(0))
            train_losses.update(loss_1.item(), train_targets.size(0))
            train_losses2.update(loss_2.item(), train_targets.size(0))
            if apply_shade==True:
                train_losses3.update(loss_3.item(), train_targets.size(0))
                train_losses4.update(loss_4.item(), train_targets.size(0))
            if apply_wt==True:
                train_losses5.update(loss_5.item(), train_targets.size(0))
                train_losses6.update(loss_6.item(), train_targets.size(0))

        loss = train_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.local_rank == 0:
            if i % args.train.print_interval == 0:
                logger.info(f'Epoch-{epoch}: {progress.display(i)}')

        if i % args.train.iteration_gap == 0 and not i==0:
            validate(test_pos, test_neg, model, criterion, criterion_2, criterion_3, criterion_4, criterion_5, criterion_6, p, apply_shade, apply_wt,
                optimizer, scheduler, wandb, saver, device, epoch, i)

    if args.local_rank == 0:
        wandb.log({
            'train_Acc': train_acces.avg, 
            'train_Class1': train_losses.avg,
            'train_Depth1': train_losses2.avg,
            'train_Class2': train_losses3.avg,
            'train_Depth2': train_losses4.avg,
            'train_AIAW': train_losses5.avg,
        }, step=epoch)

def validate(test_pos, test_neg, model, criterion, criterion_2, criterion_3, criterion_4, criterion_5, criterion_6, p, apply_shade, apply_wt,
             optimizer, scheduler, writer, saver, device, epoch, iteration):
    acces = AverageMeter('Acc', ':.5f')
    losses = AverageMeter('Loss', ':.5f')
    losses2 = AverageMeter('Loss_2', ':.5f')

    y_preds = []
    y_trues = []
    y_depth_preds = []
    model.eval()
    val_loader_len = len(test_pos)+len(test_neg)

    if apply_shade==False:
        for name, param in model.named_parameters():
            # print(name, ':', param.shape)
            if 'CSA_layers' in name:
                param.requires_grad = False
            if 'conv_final' in name:
                param.requires_grad = False
    else: # apply_shade==True
        for name, param in model.named_parameters():
            if 'CSA_layers' in name:
                param.requires_grad = True
            if 'conv_final' in name:
                param.requires_grad = True

    for i, datas in enumerate(tqdm(test_pos)):
        with torch.no_grad():
            images = datas[0].to(device)
            targets = datas[1].to(device)
            dep_GT = datas[2].to(device)
            
            outputs_catcls, outputs_catdepth, outputs_catfeat, outputs_shadecls, outputs_shadedepth, outputs_shadefeat = model_forward(images, targets, model, apply_shade, apply_wt=False, cal_covstat=False)
            probs = torch.softmax(outputs_catcls["out"], dim=1)[:,0]
            bs, _, _, _ =  outputs_catdepth["out"].shape
            # torch.Size([6, 1, 32, 32]) --> torch.Size([6, 1024]) --> torch.Size([6])
            depth_probs = outputs_catdepth["out"].reshape(bs, -1).mean(dim=1) 
            probs =  probs + depth_probs

            y_preds.extend(probs)
            y_trues.extend(targets)
            # y_depth_preds.extend(depth_pred)

            loss_1 = criterion(outputs_catcls["out"], targets)
            loss_2 = criterion_2(outputs_catdepth["out"].squeeze(), dep_GT)
            batch_num, _ = outputs_catcls["out"].shape

            if args.distributed:
                losses.update(reduce_tensor(loss_1.data).item(), targets.size(0))
                losses2.update(reduce_tensor(loss_2.data).item(), targets.size(0))
            else:
                losses.update(loss_1.item(), targets.size(0))
                losses2.update(loss_2.item(), targets.size(0))
        if i==200:
            break

    for i, datas in enumerate(tqdm(test_neg)):
        with torch.no_grad():
            images = datas[0].to(device)
            targets = datas[1].to(device)
            dep_GT = datas[2].to(device)

            outputs_catcls, outputs_catdepth, outputs_catfeat, outputs_shadecls, outputs_shadedepth, outputs_shadefeat = model_forward(images, targets, model, apply_shade, apply_wt=False, cal_covstat=False)
            probs = torch.softmax(outputs_catcls["out"], dim=1)[:,0]
            bs, _, _, _ =  outputs_catdepth["out"].shape
            # torch.Size([6, 1, 32, 32]) --> torch.Size([6, 1024]) --> torch.Size([6])
            depth_probs = outputs_catdepth["out"].reshape(bs, -1).mean(dim=1)  
            probs =  probs + depth_probs


            y_preds.extend(probs)
            y_trues.extend(targets)
            # y_depth_preds.extend(depth_pred)

            loss_1 = criterion(outputs_catcls["out"], targets)
            loss_2 = criterion_2(outputs_catdepth["out"].squeeze(), dep_GT)

            batch_num, _ = outputs_catcls["out"].shape
           

            if args.distributed:
                losses.update(reduce_tensor(loss_1.data).item(), targets.size(0))
                losses2.update(reduce_tensor(loss_2.data).item(), targets.size(0))
            else:
                losses.update(loss_1.item(), targets.size(0))
                losses2.update(loss_2.item(), targets.size(0))
        if i==200:
            break    

    y_preds = torch.stack(y_preds)
    y_trues = torch.stack(y_trues)
    if args.distributed:
        gather_y_preds = [torch.ones_like(y_preds) for _ in range(dist.get_world_size())]
        gather_y_trues = [torch.ones_like(y_trues) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_y_preds, y_preds)
        dist.all_gather(gather_y_trues, y_trues)
        gather_y_preds = torch.cat(gather_y_preds)
        gather_y_trues = torch.cat(gather_y_trues)
    else:
        gather_y_preds = y_preds
        gather_y_trues = y_trues
        gather_y_depth_preds = y_depth_preds

    metrics = cal_metrics(gather_y_trues.cpu().tolist(), gather_y_preds.cpu().tolist(), threshold='auto')

    if args.local_rank == 0:
        epoch = f'{epoch}0{iteration}'
        epoch = int(epoch)
        best_acc, best_epoch = saver.save_checkpoint(epoch, metric=metrics.AUC)

        for k, v in metrics.items():
            logger.info(f'val_{k}: {100 * v:.4f}')
        logger.info(f'val_loss_1: {losses.avg:.4f}')
        logger.info(f'val_loss_2: {losses2.avg:.4f}')
        logger.info(f'best_val_auc: {best_acc:.4f} (Epoch-{best_epoch})')

        last_lr = [group['lr'] for group in scheduler.optimizer.param_groups][0]
        wandb.log({
            'val_ACER': metrics.ACER, 
            'val_APCER': metrics.APCER,
            'val_BPCER': metrics.BPCER,
            'val_AUC': metrics.AUC,
            'val_ACC': metrics.ACC,
            'val_loss': losses.avg,
            'lr': last_lr,
        }, step=epoch)
    model.train()

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def validate_for_prototype(train_pos, train_neg, model, epoch, device, apply_shade, apply_wt):
    model.eval()
    style_list_pos = torch.empty(size=(0, 2, args.model.params.style_dim)).cuda() # 0,2,C
    style_list_neg = torch.empty(size=(0, 2, args.model.params.style_dim)).cuda() # 0,2,C
    for trial in range(args.proto_trials):

        train_loader_len = min(len(train_pos),len(train_neg)) # 834
        # print('train_loader_len', train_loader_len)
        # import pdb
        # pdb.set_trace()
        for idx, datas_pos in enumerate(tqdm(train_pos)):
            # train_images, _, _ = mix_pos_neg(datas_pos, datas_neg)
            train_images, train_targets, _ = preprocess(datas_pos)
            train_images = train_images.to(device)
            train_targets = train_targets.to(device)
            
            with torch.no_grad():
                outputs_catcls, outputs_catdepth, outputs_catfeat, outputs_shadecls, outputs_shadedepth, outputs_shadefeat = model_forward(train_images, train_targets, model, apply_shade, apply_wt=False, cal_covstat=False)
            
            features = outputs_catfeat['cat_feat']

            img_mean = features.mean(dim=[2,3]) # B,C
            img_var = features.var(dim=[2,3]) # B,C
            img_sig = (img_var+1e-5).sqrt()
            img_statis_pos = torch.stack((img_mean, img_sig), dim=1) # B,2,C
            style_list_pos = torch.cat((style_list_pos, img_statis_pos), dim=0)
        
            del train_images
            del train_targets

            # # Logging
            if args.local_rank == 0:
                if idx % 100 == 0:
                    print('select pos proto idx:', idx)
                    # break
            #     if args.local_rank == 0:
            #         logger.info("trial {:d} \t validating for prototype: {:d} / {:d} ".format(trial, idx, len(train_loader)))
            # if idx > 10:
            #     break
            # if idx > 100:
            #     break

        for idx, datas_neg in enumerate(tqdm(train_neg)):
            # train_images, _, _ = mix_pos_neg(datas_pos, datas_neg)
            train_images, train_targets, _ = preprocess(datas_neg)
            train_images = train_images.to(device)
            train_targets = train_targets.to(device)
            
            with torch.no_grad():
                outputs_catcls, outputs_catdepth, outputs_catfeat, outputs_shadecls, outputs_shadedepth, outputs_shadefeat = model_forward(train_images, train_targets, model, apply_shade, apply_wt=False, cal_covstat=False)
            
            features = outputs_catfeat['cat_feat']

            img_mean = features.mean(dim=[2,3]) # B,C
            img_var = features.var(dim=[2,3]) # B,C
            img_sig = (img_var+1e-5).sqrt()
            img_statis_neg = torch.stack((img_mean, img_sig), dim=1) # B,2,C
            style_list_neg = torch.cat((style_list_neg, img_statis_neg), dim=0)
        
            del train_images
            del train_targets

            # # Logging
            if args.local_rank == 0:
                if idx % 100 == 0:
                    print('select neg proto idx:', idx)
                    # break
            #     if args.local_rank == 0:
            #         logger.info("trial {:d} \t validating for prototype: {:d} / {:d} ".format(trial, idx, len(train_loader)))
            # if idx > 10:
            #     break

    style_list_pos = concat_all_gather(style_list_pos) # N,2,C
    style_list_neg = concat_all_gather(style_list_neg) # N,2,C

    print('final style list_pos size : ',style_list_pos.size())
    print('final style list_neg size : ',style_list_neg.size())
    style_list_pos = style_list_pos.reshape(style_list_pos.size(0), -1).detach()
    style_list_neg = style_list_neg.reshape(style_list_neg.size(0), -1).detach()

    proto_styles_pos, centroids_pos = farthest_point_sample_tensor(style_list_pos, args.model.params.base_style_num) # C,2C
    proto_styles_neg, centroids_neg = farthest_point_sample_tensor(style_list_neg, args.model.params.base_style_num) # C,2C

    proto_styles_pos = proto_styles_pos.reshape(args.model.params.base_style_num, 2, args.model.params.style_dim)
    proto_styles_neg = proto_styles_neg.reshape(args.model.params.base_style_num, 2, args.model.params.style_dim)
    proto_mean_pos, proto_std_pos = proto_styles_pos[:,0], proto_styles_pos[:,1] # 384, 384
    proto_mean_neg, proto_std_neg = proto_styles_neg[:,0], proto_styles_neg[:,1] # 384, 384

    if args.local_rank == 0:
        print('style info first after calculation~~~')
        print('proto_mean_pos', proto_mean_pos)
        print('proto_std_pos', proto_std_pos)
        print('proto_mean_neg', proto_mean_neg)
        print('proto_std_neg', proto_std_neg)
        wandb.log({
            'len(style_list_pos)': style_list_pos.reshape(style_list_pos.size(0), 2, -1).size(), 
            'len(proto_styles_pos)': proto_styles_pos.shape,
            'len(style_list_neg)': style_list_neg.reshape(style_list_neg.size(0), 2, -1).size(), 
            'len(proto_styles_neg)': proto_styles_neg.shape,
        }, step=epoch)
    model.module.Shader.CSA_layers[0].CSA_norm.proto_mean_pos.copy_(proto_mean_pos)
    model.module.Shader.CSA_layers[0].CSA_norm.proto_std_pos.copy_(proto_std_pos)
    model.module.Shader.CSA_layers[0].CSA_norm.proto_mean_neg.copy_(proto_mean_neg)
    model.module.Shader.CSA_layers[0].CSA_norm.proto_std_neg.copy_(proto_std_neg)

    model.module.Shader.CSA_layers[1].CSA_norm.proto_mean_pos.copy_(proto_mean_pos)
    model.module.Shader.CSA_layers[1].CSA_norm.proto_std_pos.copy_(proto_std_pos)
    model.module.Shader.CSA_layers[1].CSA_norm.proto_mean_neg.copy_(proto_mean_neg)
    model.module.Shader.CSA_layers[1].CSA_norm.proto_std_neg.copy_(proto_std_neg)

    del style_list_pos, style_list_neg, proto_styles_pos, proto_styles_neg, proto_mean_pos, proto_std_pos, proto_mean_neg, proto_std_neg
    # Add Lines
    model.train()

def validate_for_cov_stat(train_pos, train_neg, model, epoch, apply_shade, apply_wt, device):
    model.eval()

    train_loader_len = min(len(train_pos),len(train_neg)) # 834
    # print('train_loader_len', train_loader_len)
    # import pdb
    # pdb.set_trace()
    for idx, (datas_pos, datas_neg) in enumerate(zip(train_pos, train_neg)):
        train_images, tarin_targets, _ = mix_pos_neg(datas_pos, datas_neg)
        train_images = train_images.to(device)
            
        with torch.no_grad():
            model_forward(train_images, tarin_targets, model, apply_shade, apply_wt, cal_covstat=True)
        
        del train_images

        # Logging
        if args.local_rank == 0:
            if idx % 100 == 0:
                print('update mask idx:', idx)
                # break
            # if idx > 10:
            #     break

                
if __name__ == '__main__':
    main()
