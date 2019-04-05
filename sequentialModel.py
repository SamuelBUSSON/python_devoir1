import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from sklearn.metrics import classification_report

import numpy as np

import pickle

from keras.datasets import mnist

########################################################################################################################
#                                                       Load Data                                                      #
########################################################################################################################

# Here, you define the file name with the path of the data that you want to load
fileName = 'dataset.pickle'

# Load the data
#with open(fileName, 'rb') as f:
 #   x_train, y_train,  x_test, y_test, IDPeopleForTraining, IDPeopleForTesting = pickle.load(f)

# Get all classes of the data
#classes = np.unique(y_train)

# Convert your class label to integer. Ex. ['paris', 'paris', 'tokyo', 'amsterdam']
#encoder = LabelEncoder()
#encoder.fit(y_train)  # Ex. ['paris', 'paris', 'tokyo', 'amsterdam']
#encoded_Y = encoder.transform(y_train)  # Ex. [1,1,2,3]
#y_train = np_utils.to_categorical(encoded_Y)  # Ex. [[1,0,0],[1,0,0],[0,1,0],[0,0,1]]

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])
print(x_train[1])
print(np.shape(x_train))
s = np.shape(x_train)
print(np.shape(x_train))

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

num_classes = 10

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

########################################################################################################################
#                                                    User Parameters                                                   #
########################################################################################################################

# The number of layers includes the first layer, the hidden layers and the output layer of the model.
# So, you have to set up correctly the number of neurones in the last layer.
layerNumber = 4

# Here, you define the number of neurones per layer. Warning : you have to set up correctly the number of neurones in
# the last layer according to the number of classes, for example.
neuroneNumberPerLayer = [512, 512, 512, 10]

# Here, you define the number of inputs of the sequential model. For example, they represent the number of features of
# the data set.
inputDim = 784

# Here, you define the type of activation function used for each layer of neurones.
activationTypePerLayer = ['relu', 'relu', 'sigmoid', 'softmax']
# The choices are :
#                   'elu' - Reference : https://arxiv.org/abs/1511.07289
#                   'selu' - Reference : https://arxiv.org/abs/1706.02515
#                   'relu'
#                   'softplus'
#                   'softsign'
#                   'tanh'
#                   'sigmoid'
#                   'hard_sigmoid'
#                   'exponential'
#                   'linear'
#                   'softmax'
#                   'sigmoid'

# Here, you define the type of optimizer to train the sequential model.
optimizer = 'sgd'
# The choices are :
#                   'sgd' - Stochastic gradient descent optimizer
#                   'rmsprop' - RMSProp optimizer
#                               Reference : http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
#                   'adagrad' - Adagrad optimizer
#                               Reference : http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
#                   'adadelta' - Adadelta optimizer - Reference : https://arxiv.org/abs/1212.5701
#                   'adam' - Adam optimizer - Reference : https://arxiv.org/abs/1412.6980v8
#                                             Reference : https://openreview.net/forum?id=ryQu7f-RZ
#                   'adamax' - Adamax optimizer - Reference : https://arxiv.org/abs/1412.6980v8
#                   'nadam' - Nesterov Adam optimizer - Reference : http://cs229.stanford.edu/proj2015/054_report.pdf
#                                                       Reference : http://www.cs.toronto.edu/~fritz/absps/momentum.pdf

# Here, you define a loss function (or objective function, or optimization score function) is one of the two
# parameters required to compile a model
loss = 'mean_absolute_error'
# The choices are :
#                   'mean_squared_error'
#                   'mean_absolute_error'
#                   'mean_absolute_percentage_error'
#                   'mean_squared_logarithmic_error'
#                   'squared_hinge'
#                   'hinge'
#                   'categorical_hinge'
#                   'logcosh'
#                   'categorical_crossentropy'
#                   'sparse_categorical_crossentropy'
#                   'binary_crossentropy'
#                   'kullback_leibler_divergence'
#                   'poisson'
#                   'cosine_proximity'

# A metric is a function that is used to judge the performance of your model.
# Here, you define the metrics that you want to evaluate the sequential model.
metrics = ['accuracy', 'mae']
# The choices are :
#                   'accuracy'
#                   'binary_accuracy'
#                   'categorical_accuracy'
#                   'sparse_categorical_accuracy'
#                   'top_k_categorical_accuracy'
#                   'sparse_top_k_categorical_accuracy'
#                   'binary_accuracy'

# Here, you define the number of iteration to train the sequential model.
epochs = 10

# Here, you define the number of instances to train the sequential model.
batchSize = 128
########################################################################################################################
#                                              Create the Sequential Model                                             #
########################################################################################################################

# Create a sequential model
model = Sequential()

# Create the layer of the sequential model
for i in range(0, layerNumber):

    if i == 1:

        model.add(Dense(neuroneNumberPerLayer[i], activation=activationTypePerLayer[i], input_dim=inputDim))

    else:

        model.add(Dense(neuroneNumberPerLayer[i], activation=activationTypePerLayer[i]))


########################################################################################################################
#                                            Configure the Learning Process                                            #
########################################################################################################################

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

########################################################################################################################
#                                                  Training the Model                                                  #
########################################################################################################################

model.fit(x_train, y_train, epochs=epochs, batch_size=batchSize)

########################################################################################################################
#                                                  Testing the Model                                                   #
########################################################################################################################

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# predictions = model.predict(x_test)
#
# predictions[predictions >= 0.5] = 1
# predictions[predictions < 0.5] = 0
#
# y_predictions = np.argmax(predictions, axis=1)
#
# y_test = encoder.transform(y_test)
#
# report = classification_report(y_test, y_predictions, target_names=encoder.classes_, digits=4)
#
# print(report)