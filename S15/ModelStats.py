import json
import torch
from utils import *
#TODO: pass save format also, can be pickle or json
class ModelStats:
  def __init__(self, model, path, tb):
    self.model = model
    self.path = path
    self.batch_train_loss = []
    self.batch_train_acc = []
    self.batch_lr = []
    self.batch_train_samples=[]

    self.batch_val_loss=[]
    self.batch_val_acc=[]
    self.val_samples_seen=[]
    
    self.avg_test_loss = []
    self.test_acc = []

    self.train_acc = []
    self.avg_train_loss = []
    self.lr = []

    self.batches = 0
    self.epochs = 0
    self.tb=tb

    self.curr_train_acc = 0
    self.curr_train_loss = 0
    self.curr_test_acc = 0
    self.curr_test_loss = 0
    self.train_samples_seen = 0
    self.test_samples_seen = 0
    self.best_test_loss = 100000
    

  def add_batch_train_stats(self, loss, acc, cnt, lr):
    self.batches += 1
    self.batch_train_loss.append(loss)
    self.batch_train_acc.append(acc)
    self.batch_train_samples.append(cnt)
    self.curr_train_loss += loss
    self.curr_train_acc += acc
    self.train_samples_seen += cnt
    self.batch_lr.append(lr)
  
  def add_train_tb_data(self, mask_loss, dense_loss, loss, len_data, data, target, output):
    self.tb.add_scalar('Training loss', loss/len_data, self.batches)
    self.tb.add_scalar('Mask loss', mask_loss/len_data, self.batches)
    self.tb.add_scalar('Dense loss', dense_loss/len_data, self.batches)
    self.tb.add_scalar('Learning Rate', self.batch_lr[-1], self.batches)
    self.tb.add_scalar('Overall Training Loss', self.curr_train_loss/self.train_samples_seen, self.batches)
    if self.batches%600==0:
        self.tb.add_image('Train Input Bg Images', samples(data['bg'], mean=[0.49836101, 0.50576683, 0.48631948], std=[0.27918635, 0.27833673, 0.31944063]), global_step=self.batches)
        self.tb.add_image('Train Input FgBg Images', samples(data['fgbg'], mean=[0.49641467, 0.50198746, 0.48275208], std=[0.2798614 , 0.27848746, 0.31762387]), global_step=self.batches)
        self.tb.add_image('Train Output FgBg Images', samples(target['fgbgmask'], mean=[0.05250582], std=[0.22033702]), global_step=self.batches)
        self.tb.add_image('Train Output Dense Images', samples(target['dense'], mean=[0.41627218], std=[0.25850094]), global_step=self.batches)
        self.tb.add_image('Train Pred Dense Images', samples(output['dense'], mean=[0.41627218], std=[0.25850094]), global_step=self.batches)
        self.tb.add_image('Train Pred FgBg Images', samples(output['fgbgmask'], mean=[0.05250582], std=[0.22033702]), global_step=self.batches)
        for name, weight in self.model.named_parameters():
            self.tb.add_histogram(name, weight, self.batches)
            self.tb.add_histogram(f'{name}.grad', weight.grad, self.batches)
  
  def test_tb_data(self, data, target, output):
    self.tb.add_scalar('Test loss', self.curr_test_loss/self.test_samples_seen, self.epochs)
    self.tb.add_image('Test Input Bg Images', samples(data['bg'], mean=[0.49836101, 0.50576683, 0.48631948], std=[0.27918635, 0.27833673, 0.31944063]), global_step=self.epochs)
    self.tb.add_image('Test Input FgBg Images', samples(data['fgbg'], mean=[0.49641467, 0.50198746, 0.48275208], std=[0.2798614 , 0.27848746, 0.31762387]), global_step=self.epochs)
    self.tb.add_image('Test Output FgBg Images', samples(target['fgbgmask'], mean=[0.05250582], std=[0.22033702]), global_step=self.epochs)
    self.tb.add_image('Test Output Dense Images', samples(target['dense'], mean=[0.41627218], std=[0.25850094]), global_step=self.epochs)
    self.tb.add_image('Test Pred Dense Images', samples(output['dense'], mean=[0.41627218], std=[0.25850094]), global_step=self.epochs)
    self.tb.add_image('Test Pred FgBg Images', samples(output['fgbgmask'], mean=[0.05250582], std=[0.22033702]), global_step=self.epochs)
      
  def add_batch_test_stats(self, loss, acc, cnt):
    self.curr_test_loss += loss
    self.curr_test_acc += acc
    self.test_samples_seen += cnt
    
  def add_batch_val_stats(self, loss, acc, cnt):
    self.batch_val_loss.append(loss)
    self.batch_val_acc.append(acc)
    self.val_samples_seen.append(cnt)

  def next_epoch(self, lr):
    self.epochs += 1
    #print(self.curr_test_loss, self.test_samples_seen, self.curr_train_loss, self.train_samples_seen)
    self.avg_test_loss.append(self.curr_test_loss/self.test_samples_seen)
    self.test_acc.append(self.curr_test_acc/self.test_samples_seen)
    self.avg_train_loss.append(self.curr_train_loss/self.train_samples_seen)
    self.train_acc.append(self.curr_train_acc/self.train_samples_seen)
    self.lr.append(lr)
    self.curr_train_acc = 0
    self.curr_train_loss = 0
    self.curr_test_acc = 0
    self.curr_test_loss = 0
    self.train_samples_seen = 0
    self.test_samples_seen = 0

    if self.epochs == 1 or self.best_test_loss > self.avg_test_loss[-1]:
      print(f'Validation loss decreased ({self.best_test_loss:.6f} --> {self.avg_test_loss[-1]:.6f}).  Saving model ...')
      torch.save(self.model.state_dict(), f"{self.path}/{self.model.name}.pt")
      self.best_test_loss = self.avg_test_loss[-1]

  def save(self):
    s = {"batch_train_loss":self.batch_train_loss, "batch_train_acc":self.batch_train_acc,"batch_train_samples":self.batch_train_samples,
         "batch_lr":self.batch_lr, "avg_test_loss": self.avg_test_loss, "test_acc": self.test_acc, "batch_val_loss":self.batch_val_loss,
         "train_acc": self.train_acc, "avg_train_loss" : self.avg_train_loss, "lr": self.lr, "batch_val_acc":self.batch_val_acc,
         "val_samples_seen":self.val_samples_seen, "best_test_loss": self.best_test_loss, "epochs": self.epochs}
    with open(f'{self.path}/{self.model.name}_stats.json', 'w') as fp:
      json.dump(s, fp, sort_keys=True, indent=4)


  def get_latest_batch_desc(self):
    if len(self.batch_train_loss)==0:
      return "first batch"
    return f'Batch={self.batches} Loss={self.batch_train_loss[-1]:0.4f} Acc={100*self.curr_train_acc/self.train_samples_seen:0.2f}%'

  def get_epoch_desc(self):
    return f'Epoch: {self.epochs}, Train set: Average loss: {self.avg_train_loss[-1]:.4f}, Accuracy: {100*self.train_acc[-1]:.2f}%; Test set: Average loss: {self.avg_test_loss[-1]:.4f}, Accuracy: {100*self.test_acc[-1]:.2f}%'
