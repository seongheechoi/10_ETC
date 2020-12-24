import numpy as np
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

#softmax 출력 노드 10개
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
mse1 = mean_squared_error(np.array(y1), np.array(t))
print('mse1:',mse1)
mse2 = mean_squared_error(np.array(y2), np.array(t))
print('mse2:',mse2)

cee1 = cross_entropy_error(np.array(y1), np.array(t))
print('cee1:',cee1)
cee2 = cross_entropy_error(np.array(y2), np.array(t))
print('cee2:',cee2)
