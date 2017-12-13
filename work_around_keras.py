from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D
from keras.datasets import cifar10, mnist
from keras.preprocessing import image
from keras.utils import to_categorical
import numpy as np
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from keras.optimizers import  Adam
from keras.callbacks import History
import keras.backend as K
import  keras



def getOutputShape(imageSide, kernel, padding,stride):
    return  (imageSide+2*padding-kernel)/stride + 1

def plotSamples():
    for x_batch, y_batch in datagen.flow(x_train,y_train,batch_size=BATCH_SIZE,shuffle=True, seed=43):
        for i in range(9):
            plt.subplot(330+1+i)
            plt.imshow(x_batch[i].reshape((28,28)) , cmap=plt.get_cmap('gray'))
        break


def getModel():
    model=Sequential()
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), input_shape=input_shape, padding='same', name='block1_conv1'))
    model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), padding='same', name='block1_conv2'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='block1_maxpool1'))

    model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), input_shape=x_train.shape[1:], padding='same'))
    model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    model.add(Flatten()) #IMPORTANT
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.5, seed=12))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.5, seed=12))
    model.add(Dense(num_classes, activation='softmax'))
    return model



class Recorder(keras.callbacks.Callback):

    '''
        on_epoch_end: logs include acc and loss, and optionally include val_loss (if validation is enabled in fit), and  val_acc (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include size, the number of samples in the current batch.
        on_batch_end: logs include loss, and optionally acc (if accuracy monitoring is enabled).
    '''

    def on_train_begin(self, logs=None):
        self.accs = []
        self.losses = []

    def on_train_end(self, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        return
    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_batch_begin(self, batch, logs=None):
        return
    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('acc'))


class CustomizedGenerator():

    def __init__(self, setting):
        self.batch_size = setting['batch_size']
        self.setting = setting

    def fit(self):
        self.x_train = self.setting['x_train']
        self.y_train = self.setting['y_train']

    def generate(self):
        batch_id =  np.arange(x_train.shape[0]) // self.batch_size
        while True:
            for cur_batch_id in np.random.permutation(batch_id):
                batch_mask = batch_id[batch_id==cur_batch_id]
                x_batch = x_train[batch_mask,:]
                y_batch = y_train[batch_mask, :]
                yield x_batch, y_batch




# -------------------LOAD Data-------------------------------------------
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#------------------Paramtere Setting for Later Use-------------------------
input_shape = x_train.shape[1:]
num_classes = len(np.unique(y_train))
BATCH_SIZE =32
VERBOSE = 1
nb_train_samples = x_train.shape[0]




# Fit paramters--------------------------------------------------------
opt = SGD(lr=0.0001, momentum=0.95, decay=1e-6 ,nesterov=True)
opt=Adam(lr=1e-4)
loss = 'categorical_crossentropy'




model = getModel()
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])





# Using  keras provided ImageDataGenerator----------------------------------------------
datagen = image.ImageDataGenerator(zca_whitening=True, horizontal_flip=True)
datagen.fit(x_train)
keras_generator = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=43)
recorder_keras = Recorder()
model.fit_generator(keras_generator, verbose=1, steps_per_epoch=nb_train_samples//BATCH_SIZE, callbacks=[recorder_keras])  #nb_train_samples//BATCH_SIZE,)

# train record plot
plt.plot(recorder_keras.losses)
plt.plot(recorder_keras.accs)

# test performance
score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
test_score= score[0]
test_acc = score[1]
print ('accuracy: {0}'.format(test_acc))
# Using  keras provided ImageDataGenerator   END----------------------------------------------



#Using Customized  Generator--------------------------------------------------------
mygen = CustomizedGenerator({'x_train':x_train, 'y_train':y_train, 'batch_size': 32})
mygen.fit()
generator = mygen.generate()
recorder_cust = Recorder()
model.fit_generator(generator, verbose=1, steps_per_epoch=200, callbacks=[recorder_cust])  #nb_train_samples//BATCH_SIZE,)

# train record plot
plt.plot(recorder_cust.losses)
plt.plot(recorder_cust.accs)

# test performance
score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
test_score= score[0]
test_acc = score[1]
print ('accuracy: {0}'.format(test_acc))
#Using Customized  Generator END--------------------------------------------------------


