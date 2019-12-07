import csv
import random
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

# Total of 8125 entries in this data set
# Using 6000 for training and 2125 for testing
num_training = 7000
num_testing = 8124 - num_training

# Initialize empty np array to store the data points
all_data = []

# Gather data from file.
with open('mushrooms.csv', 'r') as f:
    reader = csv.reader(f)
    i = 0

    for row in reader:
        if i >= 1:
            all_data.append(row)
        i += 1

    random.shuffle(all_data)

feature_table = []

# Split features into separate lists
for i in range(0, 23):
    feature_table.append([])
    for data in all_data:
        feature_table[i].append(data[i])


labels = []
new_feature_table = []
# Now encode each feature into ints
for f in range(0, len(feature_table)):

    encoded_features = preprocessing.LabelEncoder().fit_transform(feature_table[f])

    if f == 0:
        labels = encoded_features
    else:
        new_feature_table.append(encoded_features)


zipped_data = zip(*new_feature_table)

# Split into testing and training sets
training_data = zipped_data[:num_training]
training_labels = labels[:num_training]

testing_data = zipped_data[num_training:]
testing_labels = labels[num_training:]


# Now that the data is split, train the model then test all the testing data
model = GaussianNB()
model.fit(training_data, training_labels)

num_correct = 0
for data in range(0, len(list(testing_data))):
    predicted = model.predict([list(testing_data[data])])

    if predicted[0] == testing_labels[data]:
        num_correct += 1

print("")
print("Training set consists of {} mushrooms".format(num_training))
print("Testing set consists of {} mushrooms".format(num_testing))
print("The percentage accuracy for this Naive Bayes Model is {}%".format(100*(num_correct/float(num_testing))))
print("")
