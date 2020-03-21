def model_run(epochs): 
 from torch.optim.lr_scheduler import StepLR
 model =  Net().to(device)
 optimizer = optim.SGD(model.parameters(), lr=0.046, momentum=0.90,weight_decay=0.001)
 #scheduler=StepLR(optimizer,step_size=4,gamma=0.1)
 #optimizer=optim.Adam(model.parameters(), lr=0.055, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

 print(torch.cuda.device_count())
 EPOCHS = epochs
 for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    #scheduler.step()