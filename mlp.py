import logging
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

class MLP:
  def __init__(self):
    pass
  def get_data(self, data_path):
    # 'dataset/glass.data'
    self.dataframe = pd.read_csv(data_path, header=None)
    
    map_dic = {'building_windows_float_processed': 1, 
               'building_windows_non_float_processed': 2, 
               'vehicle_windows_float_processed': 3, 
               'containers': 4, 
               'tableware': 5, 
               'headlamps': 6}
    
    self._labels_list = []
    for key in map_dic:
      self._labels_list.append(key)
    self._labels_len = len(self._labels_list)

  def prepare_data(self, test_data_size):
    # Split train and tests subsets (70% train | 30% test) and shuffles the data
    train_data, test_data = train_test_split(self.dataframe, test_size=test_data_size)

    # Dataframe hold ID, 9 attributes and 1 class attribute (label)
    self._number_of_attributes = self.dataframe.shape[1] - 1
    self._label_column = self._number_of_attributes

    self.x_train_index, self.x_train, self.y_train = self._split_columns(train_data)
    self.x_test_index, self.x_test, self.y_test = self._split_columns(test_data)

    # convert data to nparray
    self.x_train, self.y_train = self.x_train.values, self.y_train.values
    self.x_test, self.y_test = self.x_test.values, self.y_test.values

    # normalize data
    max_value = np.max([self.x_train.max(), self.x_test.max()])
    self.x_train = self.x_train / max_value 
    self.x_test = self.x_test / max_value

    self.train_label = np.array(self._label_to_list(self.y_train))
    self.test_label = np.array(self._label_to_list(self.y_test))
  
  def create_model(self, hiden_layer_neurons, activation_functions, _loss, _metrics, _optimizer, lr):
    input_size = self._number_of_attributes - 1
    self.neural_network_model = tf.keras.models.Sequential()
    self.neural_network_model.add(tf.keras.layers.Dense(input_size,
                                  input_shape=(input_size,),
                                  activation=activation_functions[0]))
    # self.neural_network_model.add(tf.keras.layers.Dropout(0.1))
    self.neural_network_model.add(tf.keras.layers.Dense(32, activation=activation_functions[1]))
    # self.neural_network_model.add(tf.keras.layers.Dropout(0.1))
    self.neural_network_model.add(tf.keras.layers.Dense(self._labels_len, activation=activation_functions[2]))

    self.neural_network_model.compile(loss=_loss,
                                      optimizer=self._get_optimizer_from_name(_optimizer, lr),
                                      metrics=_metrics)

  def train(self, _batch_size, _epochs):
    self.history = self.neural_network_model.fit(self.x_train, self.train_label,
                                                 batch_size=_batch_size,
                                                 epochs=_epochs,
                                                 verbose=1,
                                                 validation_data=(self.x_test, self.test_label))
    self.loss_value, self.accuracy_value, self.mse_value = self.neural_network_model.evaluate(self.x_test, self.test_label)
    print("Loss value=", self.loss_value, "Accuracy value =", self.accuracy_value, "MSE value = ", self.mse_value)
  
  def show_results(self):
    # summarize history for accuracy
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Acccuracy')
    ax1.set_ylim(0.5, 1.0)
    ax1.plot(self.history.history['acc'], label='train')
    ax1.plot(self.history.history['val_acc'], label='test')
    ax1.set_label(ax1.legend(loc='lower right'))
    
    # summarize history for loss
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylim(0, 0.5)
    ax2.plot(self.history.history['loss'], label='train')
    ax2.plot(self.history.history['val_loss'], label='test')
    ax2.set_label(ax2.legend(loc='lower right'))
    
    plt.show()

  def _label_to_list(self, label_list):
    list_ = []
    for i in range(len(label_list)):
      aux_list = np.zeros(self._labels_len, dtype=int).tolist()
      index = label_list.item((i, 0)) - 1
      aux_list[index] = 1
      list_.append(aux_list)
    return list_

  def _split_columns(self, data):
    # Dataframe hold ID, 9 attributes and 1 class attribute (label)
    x_index = data.iloc[:,[0]]
    x = data.iloc[:,list(range(1, self._number_of_attributes))]
    y = data.iloc[:,[self._label_column]]
    return x_index, x, y

  def _get_optimizer_from_name(self, name, lr):
    optimizer_dic = {
      'SGD': tf.keras.optimizers.SGD(lr=lr),
      'RMSprop': tf.keras.optimizers.RMSprop(lr=lr),
      'Adagrad': tf.keras.optimizers.Adagrad(lr=lr),
      'Adadelta': tf.keras.optimizers.Adadelta(lr=lr),
      'Adam': tf.keras.optimizers.Adam(lr=lr),
      'Adamax': tf.keras.optimizers.Adamax(lr=lr),
      'Nadam': tf.keras.optimizers.Nadam(lr=lr)
    }
    return optimizer_dic[name]

if __name__ == '__main__':
  with open('hyper_parameters.yaml', 'r') as config_file:
    hyper_parameter = yaml.load(config_file, Loader=yaml.FullLoader) 
  mlp = MLP()
  mlp.get_data('dataset/glass.data')
  mlp.prepare_data(0.3)
  mlp.create_model(hyper_parameter['hiden_layer_neurons'],
                   hyper_parameter['activation_functions'],
                   hyper_parameter['loss'],
                   hyper_parameter['metrics'],
                   hyper_parameter['optimizer'],
                   hyper_parameter['lr'])
  mlp.train(hyper_parameter['batch_size'],
            hyper_parameter['epochs'])
  mlp.show_results()

#   print_separator('Predictions')
#   predictions = neural_network_model.predict(x_test)
#   print(predictions[0])
#   print(np.argmax(predictions[0]))
#   print(labels_list[np.argmax(predictions[0])])