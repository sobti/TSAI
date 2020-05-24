from tqdm import tqdm_notebook, tqdm, tnrange
from ModelStats import ModelStats
import torch.nn.functional as F
import torch
import sys

class Train:
  def __init__(self, model, trainloader, dataloader, optimizer, stats, scheduler=None, batch_scheduler=False, L1lambda = 0, LossFunction='CrossEntropyLoss', tb=None):
    self.model = model
    self.dataloader = dataloader
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.stats = stats
    self.L1lambda = L1lambda
    self.Loss=LossFunction
    self.tb=tb
    self.batch_scheduler = batch_scheduler
    if self.scheduler and hasattr(self.batch_scheduler,'best_lr'):
        self.trainset,self.valset=torch.utils.data.random_split(trainloader, [int(.70*len(trainloader)), int(.30*len(trainloader))])
        self.trainloader = dataloader.load(self.trainset)
        self.valloader = dataloader.load(self.valset)
        self.val=Val(model, self.valloader, self.stats, self.Loss)
    else:
        self.trainloader=trainloader

  def run(self):
    self.model.train()
    pbar = tqdm(self.trainloader)
    for data, target in pbar:
      # get samples
      if isinstance(data, dict):
        data['bg']=data['bg'].to(self.dataloader.device)
        data['fgbg']=data['fgbg'].to(self.dataloader.device)
      else:
        data=data.to(self.dataloader.device)
      if isinstance(target,dict):
        target['fgbgmask']=target['fgbgmask'].to(self.dataloader.device)
        target['dense']=target['dense'].to(self.dataloader.device)
      else:
        target=target.to(self.dataloader.device)
      #data, target = data.to(self.dataloader.device), target.to(self.dataloader.device)

      # Init
      self.optimizer.zero_grad()
      # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
      # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

      # Predict
      y_pred = self.model(data)
      # Calculate loss
      if isinstance(y_pred, dict) and isinstance(target,dict) and isinstance(self.Loss,dict):
        mask_loss=self.Loss['fgbgmask'](y_pred['fgbgmask'], target['fgbgmask'])
        dense_loss=self.Loss['dense'](y_pred['dense'], target['dense'])
        loss=mask_loss+dense_loss
      elif isinstance(y_pred, dict) and isinstance(target,dict):
        mask_loss=self.Loss(y_pred['fgbgmask'], target['fgbgmask'])
        dense_loss=self.Loss(y_pred['dense'], target['dense'])
        loss=mask_loss+dense_loss
      else:
        loss = self.Loss(y_pred, target)

      #Implementing L1 regularization
      if self.L1lambda > 0:
        loss += self.L1_Loss(Model=self.model, L1lambda=self.L1lambda)

      # Backpropagation
      loss.backward()
      self.optimizer.step()

      # Update pbar-tqdm
      lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
      if isinstance(y_pred, dict):
        correct=0 
      elif len(y_pred.size())>=3:
        correct=0
      else:
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item() 
      self.stats.add_batch_train_stats(loss.item(), correct, len(data['fgbg']) if isinstance(data, dict) else len(data), lr)
      if self.tb:
        self.stats.add_train_tb_data(mask_loss.item(), dense_loss.item(), loss.item(), len(data['fgbg']), data, target, y_pred)
      pbar.set_description(self.stats.get_latest_batch_desc())
      if hasattr(self.batch_scheduler,'best_lr'):
        self.val.run()
        flag = self.batch_scheduler.best_lr(self.stats,loss, lr)
        if flag:
            tqdm._instances.clear();
            pbar.close();
            self.stats.save();
            sys.exit();
      if self.scheduler:
        self.scheduler.step()
        
  def L1_Loss(self, Model, L1lambda):
    reg_loss=0
    l1_crit = torch.nn.L1Loss(size_average=False)
    for param in Model.parameters():
        target = torch.zeros_like(param)
        reg_loss += l1_crit(param, target)    
    return L1lambda*reg_loss
        
class Val:
  def __init__(self, model, dataloader, stats, LossFunction):
    self.model = model
    self.dataloader = dataloader
    self.stats = stats
    self.Loss=LossFunction

  def run(self):
    self.model.eval()
    loss=0
    correct=0
    length=0
    with torch.no_grad():
        for data, target in self.dataloader:
            data, target = data.to(self.model.device), target.to(self.model.device)
            output = self.model(data)
            loss += self.Loss(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct+= pred.eq(target.view_as(pred)).sum().item()
            length+=len(data)
    self.stats.add_batch_val_stats(loss/length, correct/length, length)
        
class Test:
  def __init__(self, model, dataloader, stats, LossFunction, tb):
    self.model = model
    self.dataloader = dataloader
    self.stats = stats
    self.Loss=LossFunction
    self.tb=tb

  def run(self):
    self.model.eval()
    losses=0
    with torch.no_grad():
        for data, target in self.dataloader:
            if isinstance(data, dict):
                data['bg']=data['bg'].to(self.model.device)
                data['fgbg']=data['fgbg'].to(self.model.device)
            else:
                data=data.to(self.model.device)
            if isinstance(target,dict):
                target['fgbgmask']=target['fgbgmask'].to(self.model.device)
                target['dense']=target['dense'].to(self.model.device)
            else:
                target=target.to(self.model.device)
            #data, target = data.to(self.model.device), target.to(self.model.device)
            output = self.model(data)
            if isinstance(output, dict) and isinstance(target,dict) and isinstance(self.Loss,dict):
                mask_loss=self.Loss['fgbgmask'](output['fgbgmask'], target['fgbgmask'])
                dense_loss=self.Loss['dense'](output['dense'], target['dense'])
                loss=mask_loss+dense_loss
            elif isinstance(output, dict) and isinstance(target,dict):
                mask_loss=self.Loss(output['fgbgmask'], target['fgbgmask'])
                dense_loss=self.Loss(output['dense'], target['dense'])
                loss=mask_loss+dense_loss
            else:
                loss = self.Loss(output, target)
            losses += loss.item()  # sum up batch loss
            if isinstance(output, dict):
                correct=0 
            elif len(output.size())>=3:
                correct=0
            else:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
            self.stats.add_batch_test_stats(losses, correct, len(data['fgbg']) if isinstance(data, dict) else len(data))
    if self.tb:
        self.stats.test_tb_data(data, target, output)

class ModelTrainer:
  def __init__(self, model, optimizer, train_loader, test_loader, dataloader, statspath, scheduler=None, batch_scheduler=False, L1lambda = 0, LossType='CrossEntropyLoss', tb=None):
    self.model = model
    self.scheduler = scheduler
    self.batch_scheduler = batch_scheduler
    self.optimizer = optimizer
    self.tb=tb
    self.stats = ModelStats(model, statspath, self.tb)
    if isinstance(LossType,dict):
        self.Loss_Function={'fgbgmask':self.LossFunction(LossType['fgbgmask']), 'dense':self.LossFunction(LossType['dense'])}
    else:
        self.Loss_Function = self.LossFunction(LossType)
    self.train = Train(model, train_loader, dataloader, optimizer, self.stats, self.scheduler if self.scheduler and self.batch_scheduler else None, self.batch_scheduler, L1lambda, self.Loss_Function, self.tb)
    self.test = Test(model, test_loader, self.stats, self.Loss_Function, self.tb)
    
  def run(self, epochs=10):
    pbar = tqdm_notebook(range(1, epochs+1), desc="Epochs")
    for epoch in pbar:
      self.train.run()
      self.test.run()
      try:
        scheduler_lr = self.scheduler.get_last_lr()[0]
      except:
        scheduler_lr = self.optimizer.param_groups[0]['lr']
      self.stats.next_epoch(scheduler_lr if self.scheduler else 0)
      pbar.write(self.stats.get_epoch_desc())
      if self.scheduler and not self.batch_scheduler:
        try:
            self.scheduler.step()
        except:
            self.scheduler.step(self.stats.avg_test_loss[-1])
      if self.scheduler:
        pbar.write(f"Learning Rate = {scheduler_lr:0.6f}")
    # save stats for later lookup
    self.stats.save()
    
  def LossFunction(self,Loss_Type="CrossEntropyLoss"):
    """L1Loss, MSELoss, CrossEntropyLoss, CTCLoss, NLLLoss, PoissonNLLLoss, KLDivLoss, BCELoss, BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss."""
    if Loss_Type=="CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()
    elif Loss_Type=="BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss()
    elif Loss_Type=="BCELoss":
        return torch.nn.BCELoss()
    elif Loss_Type=="MSELoss":
        return torch.nn.MSELoss()
    elif Loss_Type=="TripletMarginLoss":
        return torch.nn.TripletMarginLoss()