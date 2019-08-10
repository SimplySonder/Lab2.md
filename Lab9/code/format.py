from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

from PIL import Image
import glob, os
import PIL.ImageOps


# im1 = Image.open("bag.jpg")
# im1.show()
# im2 = Image.open("shirt.jpg")
# im2.show()
# im3 = Image.open("dress.jpg")
# im3.show()

size = 28, 28
im_array = []
x = 0
for infile in glob.glob("*.jpg"):
    file, ext = os.path.splitext(infile)
    if (infile == "shirt.jpg" or infile == "bag.jpg" or infile == "dress.jpg"):
        im = Image.open(infile)
        im.thumbnail(size)
        #im.save(file + ".thumbnail", "JPEG")
        #im.show()
        out=im.convert("L")
        #out.show()

        inverted_image = PIL.ImageOps.invert(out)
        #inverted_image.save(str(x)+'.png')
        #inverted_image.show()
        im_array.append(inverted_image)
        x+=1

x = 0
for i in im_array:
    #i.show()
    # plt.figure()
    # plt.imshow(i)
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()
    np_im = np.array(i)
    print(np_im.shape)
    im_array[x] = np_im/255.0
    x+=1

train_images = train_images / 255.0
test_images = test_images / 255.0
print(test_images[0])
print("im_array")
im_array = np.array(im_array)
print(im_array[0])

# plt.figure(figsize=(10,10))
# for i in range(3):
#    plt.subplot(1,3,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(im_array[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
# plt.show()
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(im_array)
print(predictions[0])

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)





def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  print(predictions_array, predicted_label,class_names[predicted_label])
real_labels = [6,3,8]

# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, real_labels, im_array)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  real_labels)
# plt.show()

num_rows = 1
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, real_labels, im_array)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, real_labels)
plt.show()
