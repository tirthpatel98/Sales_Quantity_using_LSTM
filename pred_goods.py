from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy
import csv
from sklearn.preprocessing import StandardScaler
train_rows=1442
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = pd.read_csv('/home/tirth/PycharmProjects/demo/venv/bin/goods.csv', header=0, index_col=None)
values = dataset.values

# ensure all data is float
values = values.astype('float32')

# normalize features
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled = scaler.fit_transform(values)


w = numpy.sqrt(sum(values**2))
x_norm2 = values/w
inv_fac=w[-1]


# frame as supervised learning
reframed = series_to_supervised(x_norm2,4,2)
print(reframed.head())


# split into train and test sets
values = reframed.values

train = values[:train_rows, :]
test = values[train_rows:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1:]
test_X, test_y = test[:, :-1], test[:, -1:]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')



# fit network
history = model.fit(train_X, train_y, epochs=250, batch_size=30, validation_data=(test_X, test_y), verbose=2,shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#print(yhat.shape,test_y.shape)

# calculate RMSE
rmse = sqrt(mean_squared_error(yhat, test_y))
print('Test RMSE: %f' % rmse)

rev=yhat*inv_fac
#print("rrrrrrrrrrrrrrrr",rev)


trev=test_y*inv_fac
#print("ttttttttttt",trev)

rev=numpy.rint(rev)
trev=numpy.rint(trev)

#writing data to csv file
rev1=[]
trev1=[]
diff1=[]
avg_diff_per_weak=[]
total=0

for j in range(len(rev)):
    rev1.append(rev[j])
for k in range(len(trev)):
    trev1.append(trev[k])
for i in range(len(rev)):
    diff=(trev1[i]-rev1[i])
    diff1.append(diff)

count=0
for p in range(len(rev)):
    count=count+1
    total=total+(trev1[p]-rev1[p])
    if(count%7==0):
        avg_diff_per_weak.append(int(total/7))
        total=0
    else:
        avg_diff_per_weak.append("-")

with open('avg_predict.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(("pred", "actual","diff","avg_diff_per_weak"))
    wr.writerows(zip(rev1,trev1,diff1,avg_diff_per_weak))


