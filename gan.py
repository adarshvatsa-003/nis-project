import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import keras
from keras import backend as K


session=tf.Session()
K.set_session(session)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training numbers: %d" % len(x_train))
print("Test numbers: %d" % len(x_test))


no_classes=10
inds= np.array([y_train== i for i in range(no_classes)])
f,ax=plt.subplots(2, 5, figsize=(10, 5))
ax=ax.flatten()
for i in range(no_classes):
  ax[i].imshow(x_train[np.argmax(inds[i])].reshape(28,28))
  ax[i].set_title(str(i))
plt.show()


#flatten and normalize image before putting it into any model
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = x_train.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images=x_test.reshape((10000, 28*28))
test_images=test_images.astype('float32')/255

from keras.utils import to_categorical
train_labels=to_categorical(y_train)
test_labels=to_categorical(y_test)
from keras.callbacks import ModelCheckpoint
h=network.fit(train_images,
              train_labels,
              epochs=5,
              batch_size=128,
              shuffle=True,
              callbacks=[ModelCheckpoint('tutorial_MNIST.h5',save_best_only=True)])

# summarize history for accuracy
plt.plot(h.history['acc'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

score, acc = network.evaluate(test_images,
                            test_labels,
                            batch_size=128)

print ("Test Accuracy: %.5f" % acc)

network.save('tutorial_MNIST.h5')
##################################################################################
from keras.models import load_model
network = load_model('tutorial_MNIST.h5')
print(network.summary())

import numpy as np
from cleverhans.utils_keras import KerasModelWrapper
wrap = KerasModelWrapper(network)

x = tf.placeholder(tf.float32, shape=(None, 784))
y = tf.placeholder(tf.float32, shape=(None, 10))

from cleverhans.attacks import FastGradientMethod
fgsm = FastGradientMethod(wrap, sess=session)

fgsm_rate = 0.08
fgsm_params = {'eps': fgsm_rate,'clip_min': 0.,'clip_max': 1.}
adv_x = fgsm.generate(x, **fgsm_params)
adv_x = tf.stop_gradient(adv_x)
adv_prob = network(adv_x)

fetches = [adv_prob]
fetches.append(adv_x)
outputs = session.run(fetches=fetches, feed_dict={x: test_images})
adv_prob = outputs[0]
adv_examples = outputs[1]

adv_predicted = adv_prob.argmax(1)
adv_accuracy = np.mean(adv_predicted == y_test)

print("Adversarial accuracy: %.5f" % adv_accuracy)

n_classes = 10
f,ax=plt.subplots(2, 5, figsize=(10, 5))
ax=ax.flatten()
for i in range(n_classes):
    ax[i].imshow(adv_examples[i].reshape(28,28))
    ax[i].set_title("Adv: %d, Label: %d" % (adv_predicted[i], y_test[i]))
plt.show()