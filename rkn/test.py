import numpy as np

vec_size = 10
dt = 1e-6

test_A = np.random.uniform(low=-1, high=1, size=[vec_size, vec_size])


def f_outer(x):
    return x / np.linalg.norm(x)

def df_outer(x):
    s = np.linalg.norm(x)
    n = x / s
    return  (1 / s) * (np.eye(vec_size) - np.matmul(n, n.T))

def f_inner(x):
    return np.matmul(test_A, x)

def df_inner(x):
    return test_A

def f(x):
    return f_outer(f_inner(x))

def df(x):
    return np.matmul(df_outer(f_inner(x)), df_inner(x))

def central_diff(func, point):
    grad = np.zeros([vec_size, vec_size])
    for i in range(vec_size):
        x_temp = point.copy()
        x_temp[i] += dt
        u = func(x_temp)
        x_temp[i] -= 2*dt
        l = func(x_temp)
        grad[:, i] = np.squeeze((u - l) / (2 * dt))
    return grad



test_x = np.random.uniform(low=-1, high=1, size=[vec_size, 1])
test_sig = np.random.uniform(low = 0.01, high=1, size=[vec_size, vec_size])

Ax = np.matmul(test_A, test_x)

new_sig_1 = np.matmul(test_A, np.matmul(test_sig, test_A.T))
new_sig_1 = np.matmul(df_outer(Ax), np.matmul(new_sig_1, df_outer(Ax).T))

dAx = df(test_x)
new_sig_2 = np.matmul(dAx, np.matmul(test_sig, dAx.T))

assert np.all(np.isclose(new_sig_1, new_sig_2))

df_analytic = df(test_x)
df_numeric = central_diff(f, test_x)

#print(df_analytic)
#print(df_numeric)

new_sigma_3 = np.matmul(test_A, np.matmul(test_sig, test_A.T)) / (np.linalg.norm(test_x)**2)

print(np.abs(new_sig_2 - new_sigma_3))

assert np.all(np.isclose(df_numeric, df_analytic))


