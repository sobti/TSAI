from tqdm import tqdm_notebook as tqdm
import torch
import torch.nn as nn
#tqdm._instances.clear()

class test_train():
    
 def train(model, device, train_loader, optimizer, epoch):
  train_losses = []    
  train_acc = []
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  cri = nn.BCEWithLogitsLoss()
  #cri=nn.MSELoss()
  for batch_idx, data in enumerate(pbar):
    # get samples
    data['bg'] = data['bg'].to(device)
    data['fgbg'] = data['fgbg'].to(device)
    data['mask'] = data['mask'].to(device)
    data['depth'] = data['depth'].to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred,y_pred1 = model(data['fgbg'],data['bg'])
    #y_pred = model(data['fgbg'],data['bg'])
    #y_pred = model(data['fgbg'])
   
    # Calculate loss
    
    loss1=cri(y_pred,data['mask'])
    loss2=cri(y_pred1,data['depth'])
    loss=(2*loss1)+loss2
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    #pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)
    
    pbar.set_description(desc= f'Loss={loss1.item()} Batch_id={batch_idx}')
    #train_acc.append(100*correct/processed)

 def test(model, device, test_loader):
    test_losses = []
    test_acc = []
    model.eval()
    map=[]
    test_loss = 0
    correct = 0
    cri = nn.BCEWithLogitsLoss()
    #cri=nn.MSELoss()
    with torch.no_grad():
        for data in test_loader:
            data['bg'] = data['bg'].to(device)
            data['fgbg'] = data['fgbg'].to(device)
            data['mask'] = data['mask'].to(device)
            data['depth'] = data['depth'].to(device)

            output,output1 = model(data['bg'],data['fgbg'])
            #output = model(data['fgbg'],data['bg'])
            l1=cri(output,data['mask'])
            l2=cri(output1,data['depth'])
            los=l1+l2
            test_loss += los.item()  # sum up batch loss
#            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#            correct += pred.eq(target.view_as(pred)).sum().item()
    
    
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    
#        100. * correct / len(test_loader.dataset)))
    
#    test_acc.append(100. * correct / len(test_loader.dataset))
  