from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
import numpy as  np
import json

class LR_Finder():
	def __init__(self, optimizer):
		self.optimizer = optimizer
		self.history = {"lr": [], "loss": []}
		
	def range_test(self, start_lr=None, end_lr=10,num_iter=100,step_mode="exp", smooth_f=0.05, diverge_th=5):
         if start_lr:
             self._set_learning_rate(start_lr)
         if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
         elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)
         else:
            raise ValueError("expected one of (exp, linear), got {}".format(step_mode))

         if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1[")
         self.smooth_f = smooth_f
         self.diverge_th = diverge_th
         return self.optimizer, lr_schedule

	def _set_learning_rate(self, new_lrs):
         if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(self.optimizer.param_groups)
         if len(new_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                "Length of `new_lrs` is not equal to the number of parameter groups "
                + "in the given optimizer"
            )
        
         for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr

    
	def best_lr(self, stats, loss, lr):
         self.history["lr"].append(lr)
         loss=stats.batch_val_loss[-1]
         if stats.batches == 1:
            self.best_loss = loss
            self.best_learn = lr
         else:
             if self.smooth_f > 0:
                 loss = self.smooth_f * loss + (1 - self.smooth_f) * self.history["loss"][-1]
             if loss < self.best_loss:
                 self.best_loss = loss
                 self.best_learn = lr
		# Check if the loss has diverged; if it has, stop the test
         self.history["loss"].append(loss)
         if loss > self.diverge_th * self.best_loss:
             print("Stopping early, the loss has diverged")
             return True
         return 0
		
	def plot(self, load_path, cols, ylabel, legend_arr, title, save_path, skip_start=10, skip_end=5, log_lr=True, show_lr=None):
         """Plots the learning rate range test.
        Arguments:
            skip_start (int, optional): number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): number of batches to trim from the start.
                Default: 5.
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.
            show_lr (float, optional): if set, adds a vertical line to visualize the
                specified learning rate. Default: None.
            ax (matplotlib.axes.Axes, optional): the plot is created in the specified
                matplotlib axes object and the figure is not be shown. If `None`, then
                the figure and axes object are created in this method and the figure is
                shown . Default: None.
        Returns:
            The matplotlib.axes.Axes object that contains the plot.
        """

         if skip_start < 0:
             raise ValueError("skip_start cannot be negative")
         if skip_end < 0:
            raise ValueError("skip_end cannot be negative")
         if show_lr is not None and not isinstance(show_lr, float):
            raise ValueError("show_lr must be float")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
         with open(load_path) as f:
            data = json.load(f)
         lrs = data["batch_lr"]

        # Create the figure and axes object if axes was not already given
         fig, ax = plt.subplots(figsize=(15, 6))

        # Plot loss as a function of the learning rate
         for col in cols:
            param=data[col]
            if skip_end == 0:
                lra = lrs[skip_start:]
                param = param[skip_start:]
            else:
                lra = lrs[skip_start:-skip_end]
                param = param[skip_start:-skip_end]
            ax.plot(lra,param)
         ax.set(xlabel="Learning Rate", ylabel=ylabel)
         ax.legend(legend_arr)
         if log_lr:
            ax.set_xscale("log")

         if show_lr is not None:
            ax.axvline(x=show_lr, color="red")

        # Show only if the figure was created internally
         plt.show()
         fig.suptitle(title, fontsize=16)
         fig.savefig(save_path+'/'+title+'.jpg')
        
class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]
		