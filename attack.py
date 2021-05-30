import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
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
outputs = session.run(fetches=fetches, feed_dict={x:test_images})
adv_prob = outputs[0]
adv_examples = outputs[1]

adv_predicted = adv_prob.argmax(1)
adv_accuracy = np.mean(adv_predicted == y_test)

print("Adversarial accuracy: %.5f" % adv_accuracy)

n_classes = 10
f,ax=plt.subplots(2,5,figsize=(10,5))
ax=ax.flatten()
for i in range(n_classes):
    ax[i].imshow(adv_examples[i].reshape(28,28))
    ax[i].set_title("Adv: %d, Label: %d" % (adv_predicted[i], y_test[i]))
plt.show()

