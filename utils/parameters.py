import json
import argparse
from omegaconf import OmegaConf


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/ICM2O.yaml')
    parser.add_argument('--distributed', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--sync-bn', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    ## proto use seed
    parser.add_argument('--proto_select_epoch', type=int, default=1,
                        help='epoch to select proto')
    parser.add_argument('--online_proto', action='store_true', default=True,
                        help=' use online prototype')
    parser.add_argument('--dynamic_proto', action='store_true', default=True,
                        help='replace proto every several epochs')
    parser.add_argument('--set_proto_seed', action='store_true', default=True,
                        help='set seed for prototype dataloader')
    parser.add_argument('--proto_trials', type=int, default=1)
    args = parser.parse_args()

    _C = OmegaConf.load(args.config)
    _C.merge_with(vars(args))

    if _C.debug:
        _C.train.epochs = 2

    return _C


if __name__ == '__main__':
    args = get_parameters()
    print(args)
