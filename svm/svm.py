print("Starting up...")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.svm import SVC

#parse prepped data
data = pd.read_csv("prep_mush.csv")
data = data.to_numpy()

iterations = int(input("Number of iterations: "))

accTotal = 0
accV = np.zeros(iterations)
for it in tqdm(range(iterations)):
    #split into training data
    #setting this up is going to be very similar to shannon's MLP
    np.random.shuffle(data)

    size = int(data.shape[0]*0.8)
    training_labels = data[:size, 0]
    training_data = data[:size, 1:]

    test_labels = data[size: , 0]
    test_data = data[size:, 1:]

    model = SVC(gamma='auto')
    model.fit(training_data, training_labels)

    acc = 0
    for i in range(test_data.shape[0]):
        guess = model.predict([test_data[i]])
        if guess == test_labels[i]:
            acc += 1
    acc = acc/float(test_data.shape[0])
    accTotal += acc
    accV[it] = acc*100

accTotal = (accTotal/iterations)*100
accTotal = round(accTotal, 2)
print("Average Accuracy over " + str(iterations) + " iterations: " + str(accTotal))
ys = np.arange(iterations)
plt.bar(ys, accV)
plt.axis([-1, iterations, 98, 100])
plt.title("SVM Ran " + str(iterations) + " times. Average Accuracy = " + str(accTotal))
plt.ylabel("Accuracy (in %)")
plt.savefig(str(iterations) + "iterations.png")
plt.show()
