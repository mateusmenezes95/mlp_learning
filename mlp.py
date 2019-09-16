import logging
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from os import sys

class MLP:
  def __init__(self):
    pass
  def get_data(self, data_path):
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
    # Split train and tests subsets ((1-test_data_size)*100% train | test_data_size*100% test) and shuffles the data
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

    self.train_label = tf.keras.utils.to_categorical(self.y_train - 1, num_classes=self._labels_len)
    self.test_label = tf.keras.utils.to_categorical(self.y_test - 1, num_classes=self._labels_len)
  
  def create_model(self, hiden_layer_neurons, activation_functions, dropout_parameters, _loss, _metrics, _optimizer, lr):
    input_size = self._number_of_attributes - 1
    self.neural_network_model = tf.keras.models.Sequential()
    self.neural_network_model.add(tf.keras.layers.Dense(input_size,
                                  input_shape=(input_size,),
                                  activation=activation_functions[0],
                                  kernel_initializer='he_uniform'))
    if dropout_parameters[0] == True:
      self.neural_network_model.add(tf.keras.layers.Dropout(dropout_parameters[1]))
    self.neural_network_model.add(tf.keras.layers.Dense(hiden_layer_neurons, activation=activation_functions[1]))
    if dropout_parameters[2] == True:
      self.neural_network_model.add(tf.keras.layers.Dropout(dropout_parameters[3]))
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
    self.loss_value, self.accuracy_value = self.neural_network_model.evaluate(self.x_test, self.test_label)
    print("Loss value=", self.loss_value, "Accuracy value =", self.accuracy_value)
  
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
  
  def save_model(self):
    is_to_save = input("Do you want save the model [y/n]: ")
    if is_to_save == 'y':
      self.neural_network_model.save('mlp_model.h5')
      print('Model saved in mlp_model.h5 file')
      return
    print('The current model was discarded!')
    return

  def evaluate_model(self, model):
    model_loaded = tf.keras.models.load_model(model)
    model_loaded.summary()
    loss_value, accuracy_value = model_loaded.evaluate(self.x_test, self.test_label)
    print("Loss value=", loss_value, "Accuracy value =", accuracy_value)

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
  print(type(sys.argv[1]))
  if sys.argv[1] != 'train_model' and sys.argv[1] != 'evaluate_model':
    print('Input argument <' + sys.argv[1] + '> is invalid! Options are: train_model ou evaluate_model')
    sys.exit()

  mlp = MLP()
  mlp.get_data('dataset/glass.data')
  mlp.prepare_data(0.3)

  if sys.argv[1] == 'train_model':
    with open('hyper_parameters.yaml', 'r') as config_file:
      hyper_parameter = yaml.load(config_file, Loader=yaml.FullLoader)

    mlp.create_model(hyper_parameter['hiden_layer_neurons'],
                     hyper_parameter['activation_functions'],
                     hyper_parameter['dropout_parameters'],
                     hyper_parameter['loss'],
                     hyper_parameter['metrics'],
                     hyper_parameter['optimizer'],
                     hyper_parameter['lr'])

    mlp.train(hyper_parameter['batch_size'],
              hyper_parameter['epochs'])

    mlp.show_results()
    mlp.save_model()
  else:
    mlp.evaluate_model('mlp_model.h5')


#   print_separator('Predictions')
#   predictions = neural_network_model.predict(x_test)
#   print(predictions[0])
#   print(np.argmax(predictions[0]))
#   print(labels_list[np.argmax(predictions[0])])