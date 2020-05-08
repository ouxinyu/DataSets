'''
  @Author: Xinyu Ou
  @Date:   2020-05-07 11:04:20
  @Last Modified by:   Your name
  @Last Modified time: 2020-05-07 11:04:20
'''
import load_MNIST
import matplotlib.pyplot as plt


def load_datasets(show_examples=False):
    X_train = load_MNIST.load_train_images()
    y_train = load_MNIST.load_train_labels()
    X_test = load_MNIST.load_test_images()
    y_test = load_MNIST.load_test_labels()

    if show_examples is True:
        sample = X_train[1, :, :]
        plt.imshow(sample)
        plt.show()
        print('样例的矩阵形式为:\n {}'.format(sample))

    return X_train, X_test, y_train, y_test


load_datasets(show_examples=True)
