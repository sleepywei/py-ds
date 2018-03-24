import numpy as np

n_person = 2000
n_times = 500

m = np.random.choice([-1, 1], size=(n_person, n_times))

print('Result type: ', m.dtype, ', Result shape: ', m.shape)
print('Filp coin result: ', m)
print('Flip coin result statistic: ', m.sum())