from torch.optim.lr_scheduler import _LRScheduler
class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, last_epoch=-1):
        self.last_epoch=last_epoch
        super(LinearLR, self).__init__(optimizer, last_epoch)
        

    def get_lr(self):
        r = 1.5**(self.last_epoch)
        return [r *  base_lr for base_lr in self.base_lrs]
