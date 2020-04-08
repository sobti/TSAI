def album_transform(albumentations):
 train_transforms = albumentations.Compose([
                                       albumentations.RandomRotate90(),
                                       albumentations.HorizontalFlip(),
                                       albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
                                       #albumentations.GridDistortion(num_steps=5),
                                       albumentations.augmentations.transforms.Normalize((0.4914, 0.48216, 0.4453), (0.24703, 0.24349, 0.26159))
                                         ])
 test_transforms=albumentations.Compose([albumentations.augmentations.transforms.Normalize((0.4914, 0.48216, 0.4453), (0.24703, 0.24349, 0.26159))])
 return train_transforms,test_transforms