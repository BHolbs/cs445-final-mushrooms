import csv
import numpy as np

#Encoding function
def onehot_encoder(var, var_list):
  #Create a list of 0s
  encoded = [0]*len(var_list)
  #Set the matching index to 1
  encoded[var_list.index(var)] = 1

  return encoded

encoder = {
  "cap-shape" : ('b', 'c', 'x', 'f', 'k', 's'),
  "cap-surface": ('f', 'g', 'y', 's'),
  "cap-color" : ('n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'),
  "bruises" : ('t', 'f'),
  "odor" : ('a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'),
  "gill-attachment" : ('a', 'd', 'f', 'n'),
  "gill-spacing" : ('c', 'w', 'd'),
  "gill-size" : ('b', 'n'),
  "gill-color" : ('k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'),
  "stalk-shape" : ('e', 't'),
  "stalk-root" : ('b', 'c', 'u', 'e', 'z', 'r', '?'),
  "stalk-surface-above-ring" : ('f', 'y', 'k', 's'),
  "stalk-surface-below-ring" : ('f', 'y', 'k', 's'),
  "stalk-color-above-ring" : ('n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'),
  "stalk-color-below-ring" : ('n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'),
  "veil-type" : ('p', 'u'),
  "veil-color" : ('n', 'o', 'w', 'y'),
  "ring-number" : ('n', 'o', 't'),
  "ring-type" : ('c', 'e', 'f', 'l', 'n', 'p', 's', 'z'),
  "spore-print-color" : ('k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'),
  "population" : ('a', 'c', 'n', 's', 'v', 'y'),
  "habitat" : ('g', 'l', 'm', 'p', 'u', 'w', 'd')
}

#Preprocess our mushroom data. Binary variables become -1 or 1
#Multicategorical data is converted using one-hot encoding
with open('mushrooms.csv', newline = '') as data_file:
  reader = csv.reader(data_file)
  #Holds the column data
  meta = next(reader)

  #Convert each row and append to our output
  output = []
  for row in reader:
    #Class
    if row[0] == 'p':
      encoded = [0]
    else:
      encoded = [1]

    #For other points, use onehot encoding
    for i in range(1,len(row)):
      encoded.extend(onehot_encoder(row[i], encoder[meta[i]]))

    #Append to our output
    output.append(encoded)

output = np.array(output)
print(output.shape)
print(output)

#Save it back out to a csv
with open('prep_mush.csv', 'w', newline='') as out:
  writer = csv.writer(out)
  writer.writerows(output)
