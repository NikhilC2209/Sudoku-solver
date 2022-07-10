import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

data = pd.read_csv('mnist_train_small.csv')

matrix = np.array(data.iloc[0,1:]).reshape(28,28)
#print(matrix)
plt.imshow(matrix)
plt.title('Sample digit')
plt.show()