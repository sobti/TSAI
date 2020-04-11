def album_transform(albumentations):
 train_transforms = albumentations.Compose([
                                       albumentations.PadIfNeeded(4,4),
                                       albumentations.RandomCrop(32,32),
                                       albumentations.HorizontalFlip(),
                                       albumentations.Cutout(num_holes=1, max_h_size=8, max_w_size=8),
                                       albumentations.augmentations.transforms.Normalize((0.4914, 0.48216, 0.4453), (0.24703, 0.24349, 0.26159))
                                         ])
 test_transforms=albumentations.Compose([albumentations.augmentations.transforms.Normalize((0.4914, 0.48216, 0.4453), (0.24703, 0.24349, 0.26159))])
 return train_transforms,test_transforms