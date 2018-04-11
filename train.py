from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from matplotlib import pyplot as plt
from keras.callbacks import LearningRateScheduler, TensorBoard
from vis.utils import utils
from keras import activations
from vis.visualization import visualize_activation
from vis.visualization import get_num_filters
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='net1', help='net1 - (5x5); net2 - (11x11)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    return parser.parse_args()

args = get_args()

batch_size = 32
num_classes = 10
learning_rate=0.001
epochs = args.epochs
data_augmentation = True
num_predictions = 20
log_filepath  = './lenet_logs_' + args.network
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model_' + args.network + '.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def build_model(network):
    model = Sequential()
    if network == 'net1':
        model.add(Conv2D(6, (5,5), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=(32,32,3)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(16, (5,5), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    else:
        model.add(Conv2D(6, (11,11), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=(32,32,3)))
        model.add(MaxPooling2D((1, 1), strides=(2, 2)))
        model.add(Conv2D(16, (11,11), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
        model.add(MaxPooling2D((1, 1), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal'))
    sgd = optimizers.SGD(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


model = build_model(args.network)
print(model.summary())

#tensorboard logs
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
cbks = [tb_cb]

model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,callbacks=cbks,
                  validation_data=(x_test, y_test), shuffle=True)

model.save('lenet_'+args.network+'.h5')

layer_dict = dict([(layer.name, layer) for layer in model.layers])
print(layer_dict)
layer_name = 'conv2d_1'

layer_idx = utils.find_layer_idx(model, layer_name=layer_name)

plt.rcParams['figure.figsize'] = (18, 6)
img = visualize_activation(model, layer_idx, filter_indices=1)
plt.savefig('fil_in_vis.png')


filters = np.arange(get_num_filters(model.layers[layer_idx]))

# Generate input image for each filter.
vis_images = []
for idx in filters:
    img = visualize_activation(model, layer_idx, filter_indices=idx)

    # Utility to overlay text on image.
    img = utils.draw_text(img, ' {}'.format(idx))
    vis_images.append(img)

stitched = utils.stitch_images(vis_images, cols=8)
plt.axis('off')
plt.imshow(stitched)
plt.title(layer_name)
# plt.show()
plt.savefig("filter_vis_"+args.network+".png")
