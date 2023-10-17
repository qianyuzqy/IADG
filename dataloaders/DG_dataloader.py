try:
    from .DG_dataset import DG_Dataset
    from .transforms import create_data_transforms
except Exception:
    from DG_dataset import DG_Dataset
    from transforms import create_data_transforms

import torch.utils.data as data
import numpy as np

def _init_fn(worker_id):
    np.random.seed(1234)

def create_dataloader(args, split, category=None, input_type='multi'):
    '''
        define the domain generalization transforms accoding to different parameters
        Args:
            args ([type]): contain the specific parmaters
            category (str): 'pos' or 'neg'
            split (str, optinal):
                'train': to generate the domain generalization transforms for training
                'val': to generate the domain generalization transforms for validation
                'test': to generaate the domain generalization transforms for testing
    '''
    transform = create_data_transforms(args.transform, split, input_type)
    kwargs = getattr(args.dataset, args.dataset.name)
    # print(kwargs)
    print_info=False
    if 'local_rank' in args and args.local_rank==0:
        print_info=True
    dataset = eval(args.dataset.name)(split=split, transform=transform, category=category, print_info=print_info, **kwargs)
    
    sampler = None
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    shuffle = True if sampler is None and split == 'train' else False
    batch_size = getattr(args, split).batch_size

    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 sampler=sampler,
                                 num_workers=10,
                                 pin_memory=True,
                                 worker_init_fn=_init_fn)
    return dataloader


if __name__ == '__main__':
    import argparse
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/OCM2I.yaml')
    parser.add_argument('--distributed', type=int, default=0)
    args = parser.parse_args()

    local_config = OmegaConf.load(args.config)
    for k, v in local_config.items():
        setattr(args, k, v)

    train_pos = create_dataloader(args, split='train',category='pos')
    for i, datas in enumerate(train_pos):
        print(i, datas[0].shape, datas[1].shape)
        break

    train_neg = create_dataloader(args, split='train',category='neg')
    for i, datas in enumerate(train_neg):
        print(i, datas[0].shape, datas[1].shape)
        break

    test_pos = create_dataloader(args, split='val',category='pos')
    for i, datas in enumerate(test_pos):
        print(i, datas[0].shape, datas[1].shape)
        break

    test_neg = create_dataloader(args, split='val',category='neg')
    for i, datas in enumerate(test_neg):
        print(i, datas[0].shape, datas[1].shape)
        break
    
