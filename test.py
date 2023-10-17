#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import argparse
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn

import models
from dataloaders import create_dataloader
from utils.metric import *

import warnings
warnings.filterwarnings("ignore")


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

def model_forward(image, label, model, apply_shade, apply_wt, cal_covstat):
    outputs_catcls, outputs_catdepth, outputs_catfeat, outputs_shadecls, outputs_shadedepth, outputs_shadefeat = model(image, label, apply_shade, cal_covstat, apply_wt)
    return outputs_catcls, outputs_catdepth, outputs_catfeat, outputs_shadecls, outputs_shadedepth, outputs_shadefeat

def eval_state(probs, labels, thr):
    predict = probs >= thr
    FP = np.sum((labels == 0) & (predict == False))
    TP = np.sum((labels == 1) & (predict == False))
    TN = np.sum((labels == 0) & (predict == True))
    FN = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP

def test(args, model, test_pos, test_neg, device):
    prob_dict = {}
    label_dict = {}

    output_dict_tmp = {}
    target_dict_tmp = {}

    model.eval()
    apply_shade = True

    for i, datas in enumerate(tqdm(test_pos)):
        with torch.no_grad():
            images = datas[0].to(device)
            targets = datas[1].to(device)
            dep_GT = datas[2].to(device)
            img_path = datas[3]

            outputs_catcls, outputs_catdepth, outputs_catfeat, outputs_shadecls, outputs_shadedepth, outputs_shadefeat = model_forward(images, targets, model, apply_shade, apply_wt=False, cal_covstat=False)
            probs = torch.softmax(outputs_catcls["out"], dim=1)[:,0]
            bs, _, _, _ =  outputs_catdepth["out"].shape

            depth_probs = outputs_catdepth["out"].reshape(bs, -1).mean(dim=1) 
            probs =  (probs + depth_probs).cpu().data.numpy()

            label = targets.cpu().data.numpy()

            for i in range(len(probs)):
                video_path = img_path[i].rsplit('/',1)[0]
                if(video_path in prob_dict.keys()):
                    prob_dict[video_path].append(probs[i])
                    label_dict[video_path].append(label[i])
                else:
                    prob_dict[video_path] = []
                    label_dict[video_path] = []
                    prob_dict[video_path].append(probs[i])
                    label_dict[video_path].append(label[i])

    for i, datas in enumerate(tqdm(test_neg)):
        with torch.no_grad():
            images = datas[0].to(device)
            targets = datas[1].to(device)
            dep_GT = datas[2].to(device)
            img_path = datas[3]

            outputs_catcls, outputs_catdepth, outputs_catfeat, outputs_shadecls, outputs_shadedepth, outputs_shadefeat = model_forward(images, targets, model, apply_shade, apply_wt=False, cal_covstat=False)
            probs = torch.softmax(outputs_catcls["out"], dim=1)[:,0]
            bs, _, _, _ =  outputs_catdepth["out"].shape

            depth_probs = outputs_catdepth["out"].reshape(bs, -1).mean(dim=1) 
            probs =  (probs + depth_probs).cpu().data.numpy()
            label = targets.cpu().data.numpy()

            for i in range(len(probs)):
                video_path = img_path[i].rsplit('/',1)[0]
                if(video_path in prob_dict.keys()):
                    prob_dict[video_path].append(probs[i])
                    label_dict[video_path].append(label[i])
                else:
                    prob_dict[video_path] = []
                    label_dict[video_path] = []
                    prob_dict[video_path].append(probs[i])
                    label_dict[video_path].append(label[i])

    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)

    print('len(label_list)', len(label_list))
    print('len(prob_list)', len(prob_list))

    metrics = cal_metrics(label_list, prob_list, threshold='auto')


    return metrics


def main():
    # set configs
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/test.yaml')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--pred_file', type=str, default='results.txt')
    parser.add_argument('--distributed', type=int, default=0)

    args = parser.parse_args()
    local_config = OmegaConf.load(args.config)
    for k, v in local_config.items():
        setattr(args, k, v)

    # set enviroment
    os.environ['TORCH_HOME'] = args.torch_home
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define the output dir
    if args.ckpt:
        args.output_dir = os.path.dirname(args.ckpt)

    # load model
    model, args = load_model(args)
    if torch.cuda.device_count() > 1:
        print(f'Let\'s use {torch.cuda.device_count()} GPUs!')
        model = nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))
    model = model.to(device)

    # dataloader
    test_pos = create_dataloader(args, split='test',category='pos')
    test_neg = create_dataloader(args, split='test',category='neg')

    result_list = args.output_dir+'/'+args.ckpt.split('/')[-1].split('.')[0]+'_'+args.pred_file
    result_list = open(result_list,'w')

    # all_test_list
    metrics = test(args, model, test_pos, test_neg, device)
    result_list.write(f'EER: {metrics.EER}\n')
    result_list.write(f'HTER: {metrics.ACER}\n')
    result_list.write(f'AUC: {metrics.AUC}\n')
    result_list.write(f'threshold: {metrics.Thre}\n')
    result_list.write(f'ACC: {metrics.ACC}\n')

    print('EER:',metrics.EER)
    print('HTER:',metrics.ACER)
    print('AUC:',metrics.AUC)
    print('threshold:',metrics.Thre)
    print('ACC:',metrics.ACC)


if __name__ == '__main__':
    main()
