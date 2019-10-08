import numpy as np

def blink_detect():
    x = np.arange(0, 14)
    y = np.array([
    1, 0.9, .5, 0.0, 0.1, 0.8, .7, .8, .9, .3, .2, .4, 1, .8
    ])


    epsilon = 0.01
    threshold = 0.4 * np.min(y) + 0.6*np.max(y)
    n = len(y)
    
    dy_dx = y[1:]-y[:-1]
    i = np.where(dy_dx==0)[0]
    if len(i) != 0:
        for k in i:
            if k==0:
                dy_dx[0] -= epsilon
            else:
                dy_dx[k] = epsilon * dy_dx[k-1]
    m = n-1
    c = dy_dx[1:m]*dy_dx[:m-1]
    x = np.where(c < 0)[0] + 1
    
    xtrema_ears = y[x]
    t = np.ones_like(x)
    t[xtrema_ears < threshold] =- 1
    
    
    t = np.concatenate(([1], t, [1]))
    xtrema_ears = np.concatenate(([y[0]],xtrema_ears,[y[n-1]]))
    xtrema_idx = np.concatenate(([0], x, [n-1]))
    
    z = t[1:]*t[:-1]
    z_idx = np.where(z < 0)[0]
    num_of_blinks = len(z_idx) // 2
    
    selected_ear = xtrema_ears[z_idx], xtrema_ears[z_idx+1]
    selected_idx = xtrema_idx[z_idx], xtrema_idx[z_idx+1]

