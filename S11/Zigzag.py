iterations = 500
lr_max = 0.81
lr_min = lr_max / 8
step_size = 50
cycle_size = step_size * 2
from math import floor
lrrrr = []
for curr_iter in range(iterations):
  cycle = floor(1 + curr_iter/cycle_size)
  x = abs(curr_iter/step_size - (2 * cycle) + 1)
  lr_t = lr_min + (lr_max - lr_min)*(1 - x)
  lrrrr.append(lr_t)
plt.plot(lrrrr)