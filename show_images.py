from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

from util import getData

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def main():
    X, Y = getData(balance_ones=False)

    while True:
        for i in range(7):
            x, y = X[Y==i], Y[Y==i]
            N = len(y)
            j = np.random.choice(N)
            plt.imshow(x[j].reshape(48, 48), cmap='gray')
            plt.title(label_map[y[j]])
            plt.show()
        prompt = input('Quit? Enter Y:\n')
        if prompt == 'Y':
            break


if __name__ == '__main__':
    main()
