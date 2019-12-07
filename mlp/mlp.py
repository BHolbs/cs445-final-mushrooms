import csv
#import matplotlib.pyplot as plt
import mlplib as mlp
import numpy as np

#Constants
N_INPUTS = 126
N_HIDDEN = 20
N_OUTPUTS = 2
LEARN_RATE = 0.1
EPOCHS = 50
MOMENTUM = 0.5
TRAIN_RATIO = 0.8

#Get the data from the csv file
with open('prep_mush.csv', newline = '') as data_file:
  reader = csv.reader(data_file)
  data = []
  for row in reader:
    data.append(row)
  #data[:, 0] is the labels
  #data[:, 1:] is the data
  data = np.array(data).astype(int)

#Randomize the order before splitting into training and test
np.random.shuffle(data)

#Calculate the size of the training data
tr_size = int(len(data)*TRAIN_RATIO)

train_labels = data[:tr_size, 0]
train_data = data[:tr_size, 1:]
#Append bias to the start
bias = np.ones((tr_size, 1))
train_data = np.hstack((bias, train_data))

test_labels = data[tr_size: , 0]
test_data = data[tr_size:, 1:]
#Append bias to the start
bias = np.ones((len(data)-tr_size, 1))
test_data = np.hstack((bias, test_data))


#Randomly initialize weights between -0.05 and 0.05
weights_to_hidden = mlp.gen_weights(0.05, N_HIDDEN, N_INPUTS + 1)
weights_to_output = mlp.gen_weights(0.05, N_OUTPUTS, N_HIDDEN + 1)

#Use this code for breaking the training data in smaller groups
#train_data = train_data[:int(len(train_data)/2)]

#Train the network and save the outputs of the tests on the training and 
#test data set to a file.
with open("output/acc-lr-"+str(LEARN_RATE)+"-nhid-"+str(N_HIDDEN)+"-mom-"+str(MOMENTUM)+".csv", mode='w') as data_file:
  data_file = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  data_file.writerow(['epoch', 'train acc', 'test acc'])
  data_file.writerow([0,mlp.test(weights_to_hidden, weights_to_output, train_data, train_labels),mlp.test(weights_to_hidden, weights_to_output, test_data, test_labels)])
  
  for i in range(1, EPOCHS+1):
    mlp.train(weights_to_hidden, weights_to_output, train_data, train_labels, train_labels, LEARN_RATE, MOMENTUM)
    train_acc = mlp.test(weights_to_hidden, weights_to_output, train_data, train_labels)
    test_acc = mlp.test(weights_to_hidden, weights_to_output, test_data, test_labels)
    print("EPOCH ", i, ": ",train_acc,", ",test_acc)
    data_file.writerow([i, train_acc, test_acc])

#Generate the confusion matrix
cm = mlp.conf_matrix(weights_to_hidden, weights_to_output, test_data, test_labels)

#Write the confusion matrix out to a file
with open("output/cm-lr-"+str(LEARN_RATE)+"-nhid-"+str(N_HIDDEN)+"-mom-"+str(MOMENTUM)+".csv", mode='w') as conf_file:
  conf_file = csv.writer(conf_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  conf_file.writerow([' ','0','1'])
  for i in range(0,2):
    conf_file.writerow([i,cm[i][0],cm[i][1]])
