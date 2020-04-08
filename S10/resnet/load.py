def load_Data(batch_size,train_transforms,test_transforms,torchvision,torch):
 trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transforms)
 testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transforms)                                        
 SEED = 1
 cuda = torch.cuda.is_available()
 print("CUDA Available?", cuda)
 torch.manual_seed(SEED)
 if cuda:
  torch.cuda.manual_seed(SEED)  
  
 return trainset,testset

def load_test_train(batch_size,trainset,testset,torchvision,torch):
 trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
 testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
 return trainloader,testloader                                