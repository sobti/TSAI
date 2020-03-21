from torchvision import datasets, transforms
def train_trans():
     train_transforms = transforms.Compose([
                                        #transforms.Resize((28, 28)),
                                        transforms.ColorJitter(brightness=0.20, contrast=0.16, saturation=0.15, hue=0.1),
                                        transforms.RandomRotation((-15,15)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.48216, 0.4453), (0.24703, 0.24349, 0.26159)) 
                                       ])
     return train_transforms                               

def test_trans():
     test_transforms = transforms.Compose([
                                       #transforms.Resize((28, 28)),
                                       #transforms.ColorJitter(brightness=0.20, contrast=0.1, saturation=0.10, hue=0.1),
                                      #transforms.RandomRotation((-10,10),fill=(0,)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.48216, 0.4453), (0.24703, 0.24349, 0.26159))])
     return test_transforms
                                
