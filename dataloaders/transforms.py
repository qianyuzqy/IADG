import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2

def create_data_transforms_alb(args, split='train', input_type=None):
    if input_type == 'multi':
        if split == 'train':
            return alb.Compose([
                alb.Resize(args.image_size, args.image_size),
                alb.Normalize(mean=args.mean, std=args.std),
                ToTensorV2(),
            ], additional_targets={'image0': 'image'})
        elif split == 'val':
            return alb.Compose([
                alb.Resize(args.image_size, args.image_size),
                alb.Normalize(mean=args.mean, std=args.std),
                ToTensorV2(),
            ], additional_targets={'image0': 'image'})
        elif split == 'test':
            return alb.Compose([
                alb.Resize(args.image_size, args.image_size),
                alb.Normalize(mean=args.mean, std=args.std),
                ToTensorV2(),
            ], additional_targets={'image0': 'image'})
        elif split == 'SDK':
            return alb.Compose([
                alb.Normalize(mean=args.mean, std=args.std),
                ToTensorV2(),
            ], additional_targets={'image0': 'image'})
    else:
        if split == 'train':
            return alb.Compose([
                alb.Resize(args.image_size, args.image_size),
                alb.Normalize(mean=args.mean, std=args.std),
                ToTensorV2(),
            ])
        elif split == 'val':
            return alb.Compose([
                alb.Resize(args.image_size, args.image_size),
                alb.Normalize(mean=args.mean, std=args.std),
                ToTensorV2(),
            ])
        elif split == 'test':
            return alb.Compose([
                alb.Resize(args.image_size, args.image_size),
                alb.Normalize(mean=args.mean, std=args.std),
                ToTensorV2(),
            ])
        elif split == 'SDK':
            return alb.Compose([
                alb.Normalize(mean=args.mean, std=args.std),
                ToTensorV2(),
            ])


create_data_transforms = create_data_transforms_alb
