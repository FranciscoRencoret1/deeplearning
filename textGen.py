from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

# X1 = np.array([['A', 'person', 'is'], ['i', 'cant', 'sleep'], ['this', 'is', 'a']])
# X2 = np.array([1, 2, 3])
# Y = np.array([[1, 45, 78], [5, 50, 81]])

# gen = TimeseriesGenerator(X1, Y, length = 1, batch_size = 1)

# for i in gen:
# 	print(i)


def generate_data(Xp, Xq, Ys, Ye, mapper, maxContextLen, maxQuestionLen, batch_size):
  count = 0
  while True:
    X_mini_batch = [[], []]
    Y_mini_batch = [[], []]
    for i in range(batch_size):
      curr_index = count + i
      if curr_index > len(Xq) - 1:
        count = 0
        curr_index = count
      if maxPassageLen - len(Xp[mapper[curr_index]]) > 0:
        xp_temp = np.concatenate((Xp[mapper[curr_index]], np.zeros((maxPassageLen - len(Xp[mapper[curr_index]]), 300))))
      else: 
        xp_temp = Xp[mapper[curr_index]]       
      if maxQuestionLen - len(Xq[curr_index]) > 0:                  
        xq_temp = np.concatenate((Xq[curr_index], np.zeros((maxQuestionLen - len(Xq[curr_index]), 300))))
      else: 
        xq_temp = Xq[curr_index]              
      X_mini_batch[0].append(xq_temp)
      X_mini_batch[1].append(xp_temp)
      Y_mini_batch[0].append(Ys[curr_index])
      Y_mini_batch[1].append(Ye[curr_index])
    count += batch_size
    if np.array(X_mini_batch[0]).ndim < 3:
      continue
    yield ([np.array(X_mini_batch[0]), np.array(X_mini_batch[1])], [np.array(Y_mini_batch[0]), np.array(Y_mini_batch[1])])
