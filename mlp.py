import logging
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def print_separator(msg):
  print('\n<------------------ ' + msg + ' ------------------>\n')

if __name__ == '__main__':
  dataframe = pd.read_csv('dataset/glass.data', header=None)
  # Split train and tests subsets (80% train | 20% test) and shuffles the data

  # Diagnosis normal (N), altered (O)
  map_dic = {'building_windows_float_processed': 1, 
             'building_windows_non_float_processed': 2, 
             'vehicle_windows_float_processed': 3, 
             'containers': 4, 
             'tableware': 5, 
             'headlamps': 6} 
  labels_list = []
  for key in map_dic:
    labels_list.append(key)

  labels_len = len(labels_list)
  train_data, test_data = train_test_split(dataframe, test_size=0.3)

  print(len(train_data), 'trains examples')
  print(len(test_data), 'test examples')

  print_separator('Training data')
  print(train_data.head())
  print_separator('Test data')
  print(test_data.head())

  x_train_index = train_data.iloc[:,[0]]
  x_train = train_data.iloc[:,list(range(1,10))]
  y_train = train_data.iloc[:,[10]]
  
  print_separator('x_train')
  print(x_train.head())
  print_separator('y_train')
  print(y_train.head())

  x_test_index = test_data.iloc[:,[0]]
  x_test = test_data.iloc[:,list(range(1,10))]
  y_test = test_data.iloc[:,[10]]
  
  print_separator('x_test')
  print(x_test.head())
  print_separator('y_test')
  print(y_test.head())

  # # convert data to nparray
  x_train = x_train.values
  y_train = y_train.values
  x_test = x_test.values
  y_test = y_test.values

  # normalize data
  max_value = np.max([x_train.max(), x_test.max()])
  x_train = x_train / max_value 
  x_test = x_test / max_value

  print_separator('x_train as nparray')
  print(x_train)
  print_separator('y_train as nparray')
  print(y_train)
  print_separator('x_test as nparray')
  print(x_test)
  print_separator('y_test as nparray')
  print(y_test)

  train_label = np.array([[(1 if x == 1 else 0), (1 if x == 2 else 0), (1 if x == 3 else 0), (1 if x == 4 else 0), (1 if x == 5 else 0), (1 if x == 5 else 0)] for x in y_train])
  test_label = np.array([[(1 if x == 1 else 0), (1 if x == 2 else 0), (1 if x == 3 else 0), (1 if x == 4 else 0), (1 if x == 5 else 0), (1 if x == 5 else 0)] for x in y_test])

  print_separator('train_label')
  print(train_label)
  print_separator('test_label')
  print(test_label)

  neural_network_model = tf.keras.models.Sequential()
  x_train_row_len = len(x_train[0])

  neural_network_model.add(tf.keras.layers.Dense(x_train_row_len, input_shape=(x_train_row_len,), activation=tf.nn.tanh))
  neural_network_model.add(tf.keras.layers.Dense(30, activation=tf.nn.tanh))
  neural_network_model.add(tf.keras.layers.Dense(5, activation=tf.nn.tanh))
  neural_network_model.add(tf.keras.layers.Dense(labels_len, activation=tf.nn.softmax))

  lr = 0.01
  neural_network_model.compile(loss='binary_crossentropy',
                                optimizer=tf.keras.optimizers.Adamax(lr=lr),
                                metrics=['accuracy', 'mse'])

  batch_size = 1
#   num_classes = 10
  epochs = 100

  history = neural_network_model.fit(x_train, train_label,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     verbose=1,
                                     validation_data=(x_test, test_label))

  loss_value, accuracy_value, mse_value = neural_network_model.evaluate(x_test, test_label)
  print("Loss value=", loss_value, "Accuracy value =", accuracy_value, "MSE value = ", mse_value)

  print_separator('Predictions')
  predictions = neural_network_model.predict(x_test)
  print(predictions[0])
  print(np.argmax(predictions[0]))
  print(labels_list[np.argmax(predictions[0])])

  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()