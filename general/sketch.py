import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import taichi as ti
import time

seed = 2333
rng = np.random.RandomState(seed)
target_dimension = 10000
output_time = True

def sketch(ti_r, ti_m, ti_u, ti_J): # compute the sketch
    @ti.kernel
    def ti_sketch():
        for i in range(ti_r.shape[0]):
            indm = int(ti_r[i, 1])
            indu = int(ti_r[i, 2])
            sig = ti_r[i, 3]
            rnd = int(ti_r[i, 4])
            ti_J[rnd, 0] += sig * ti_r[i, 0]
            for j in range(ti_m.shape[1]):
                ti_J[rnd, 1 + j] += sig * ti_m[indm, j]
            ti_J[rnd, 1 + ti_m.shape[1]] += sig * ti_u[indu, 0]
    ti_sketch()

def covar(ti_J, ti_cov): # compute the covariance matrix
    @ti.kernel
    def ti_covar():
        for i in range(ti_J.shape[0]):
            ti_J[i, ti_J.shape[1] - 1] = 1
        for k in range(ti_J.shape[0]):
            for i in range(ti_cov.shape[0]):
                for j in range(ti_cov.shape[1]):
                    ti_cov[i, j] += ti_J[k, i] * ti_J[k, j]
    ti_covar()

def calc_mse(ti_r, ti_m, ti_u, ti_x, ti_mse): # calculate the mean squared error
    @ti.kernel
    def ti_calc_mse():
        ti_mse[None] = 0
        for i in range(ti_r.shape[0]):
            indm = int(ti_r[i, 1])
            indu = int(ti_r[i, 2])
            diff = -ti_r[i, 0]
            for j in range(ti_m.shape[1]):
                diff += ti_x[j] * ti_m[indm, j]
            diff += ti_x[ti_m.shape[1]] * ti_u[indu, 0]
            diff += ti_x[ti_m.shape[1] + 1]
            ti_mse[None] += diff ** 2
        ti_mse[None] /= ti_r.shape[0]
    ti_calc_mse()

class linear_solver():
    def __init__(self, cov):
        self.JTJ = cov[1:, 1:]
        self.JTb = cov[1:, 0]
        self.JTb = self.JTb.reshape(self.JTb.shape[0], 1)

    def solve(self, ridge): # solve ridge regression
        t = self.JTJ + ridge * np.identity(self.JTJ.shape[0])
        return np.dot(np.linalg.pinv(t), self.JTb)

def main():
    # prepare the dataset
    print('prepare dataset')
    df_m = pd.read_csv("movies.csv")
    df_r = pd.read_csv("ratings.csv")

    df_m['year'] = df_m['title'].str.extract(r'[(](\d*?)[)]', expand=False)
    df_m = df_m[pd.isnull(df_m['year']) == False]
    df_m['year'] = df_m['year'].astype('int')
    df_m = df_m.drop('title', axis=1)
    df_m['genres_list'] = df_m['genres'].str.split('|')
    list2series = pd.Series(df_m.genres_list)
    mlb = MultiLabelBinarizer()
    one_hot_genres = pd.DataFrame(mlb.fit_transform(list2series), columns=mlb.classes_, index=list2series.index)
    df_m = pd.merge(df_m, one_hot_genres, left_index=True, right_index=True)
    df_m = df_m.drop(['genres', 'genres_list'], axis=1)
    
    df_r = df_r.drop('timestamp', axis=1)

    train_set = df_r.sample(frac=0.8, random_state=233, axis=0)
    test_set = df_r[~df_r.index.isin(train_set.index)]
    train_set = train_set.reset_index().drop('index', axis=1)
    test_set = test_set.reset_index().drop('index', axis=1)

    df_u = train_set.drop('movieId', axis=1).groupby('userId').mean().reset_index().rename(columns={'rating': 'avg_user_rating'})
    df_m = df_m.merge(train_set.drop('userId', axis=1).groupby('movieId').mean().reset_index().rename(columns={'rating': 'avg_movie_rating'}))

    y = set(df_m.columns)
    y.remove('movieId')
    y = df_m.drop(y, axis=1)
    y['indm'] = np.array(range(y.shape[0]))
    z = df_u.drop('avg_user_rating', axis=1)
    z['indu'] = np.array(range(z.shape[0]))
    train_set = train_set.merge(y).merge(z).drop(['userId', 'movieId'], axis=1)
    test_set = test_set.merge(y).merge(z).drop(['userId', 'movieId'], axis=1)
    df_m = df_m.drop('movieId', axis=1)
    df_u = df_u.drop('userId', axis=1)

    train_set['sig'] = rng.randint(0, 2, size=train_set.shape[0]) * 2 - 1
    train_set['rnd'] = rng.randint(0, target_dimension, size = train_set.shape[0])
    train_set = train_set.values
    df_m = df_m.values
    df_u = df_u.values
    test_set = test_set.values

    # taichi initialization
    ti.init(arch = ti.gpu)
    ti_r = ti.field(ti.f64, train_set.shape)
    ti_m = ti.field(ti.f64, df_m.shape)
    ti_u = ti.field(ti.f64, df_u.shape)
    d = train_set.shape[1] - 4 + df_m.shape[1] + df_u.shape[1] + 1 # 1 for constant
    ti_J = ti.field(ti.f64, (target_dimension, d))
    ti_cov = ti.field(ti.f64, (d, d))
    ti_x = ti.field(ti.f64, d - 1)
    ti_test = ti.field(ti.f64, test_set.shape)
    ti_mse = ti.field(ti.f64, ())

    ti_r.from_numpy(train_set.astype(np.float64))
    ti_m.from_numpy(df_m.astype(np.float64))
    ti_u.from_numpy(df_u.astype(np.float64))
    ti_test.from_numpy(test_set.astype(np.float64))

    print('start computing covariance matrix')
    ti.sync()
    t1 = time.time()
    sketch(ti_r, ti_m, ti_u, ti_J)
    covar(ti_J, ti_cov)
    cov = ti_cov.to_numpy()
    linreg = linear_solver(cov)
    ti.sync()
    t2 = time.time()
    if output_time:
        print(t2 - t1)
    print('start solving')
    list_lambda = []
    list_mse = []
    # solving regression for different lambda
    for alpha in range(0, 500000, 5000):
        ti.sync()
        t1 = time.time()
        x = linreg.solve(ridge=alpha)
        ti.sync()
        t2 = time.time()
        if output_time:
            print(t2 - t1)
        ti_x.from_numpy(x.reshape(x.shape[0]).astype(np.float64))
        calc_mse(ti_test, ti_m, ti_u, ti_x, ti_mse)
        mse = ti_mse.to_numpy()
        list_lambda.append(alpha)
        list_mse.append(mse)
        print('lambda = {}, mse = {}'.format(alpha, mse))

    min_index = min(range(len(list_mse)), key=list_mse.__getitem__)
    print('minimum mse = {} at lambda = {}'.format(list_mse[min_index], list_lambda[min_index]))

if __name__ == '__main__':
    main()
    
