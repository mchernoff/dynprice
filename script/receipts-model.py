import pickle as pkl
import random
import os
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from dateutil import parser
import bson
import json

random.seed()

home = os.getenv("HOME")

with open(home + '/dump/porton/receipts.bson','rb') as datafile:
    data = bson.decode_all(datafile.read())

with open(home + '/dynprice/bin/receipts-dict.json', 'r') as dictfile:
    obj = json.loads(dictfile.read())
    specialties = obj['specialties']
    eventTypes = obj['eventTypes']
    types = obj['types']

features = []
labels = []

random.shuffle(data)

for row in data:
    price = (int)((float)(row['price']))        
    labels.append(price)
    row['price'] = price

labels.sort()
q1 = labels[(int)(len(labels)/4)]
q2 = labels[(int)(len(labels)/2)]
q3 = labels[(int)(3*len(labels)/4)]
tlo = q1 - 1.5*(q3-q1)
thi = q3 + 1.5*(q3-q1)


labels = []


for row in data:
    try:
        np.array([row['price']]).astype(float)
    except:
        print(row['price'])
        continue
    try:
        duration = (parser.parse(row['end']) - parser.parse(row['start'])).seconds
    except:
        duration = None
        continue
    if(row['price'] < tlo or row['price'] > thi):
        continue
    if(row['specialty'] == None):
        continue
    
    vector = np.asarray([duration])
    speciality = np.zeros(len(specialties), dtype=int)
    if(row['specialty'] not in specialties):
        print('Specialty "' + row['specialty'] + '" not mapped')
        print('Run receipts-dict-gen.py or add ' + row['type'] + ' to receipts-dict.json manually')
        exit(1)
    if(row['type'] not in types):
        print('Type "' + row['type'] + '" not mapped')
        print('Run receipts-dict-gen.py or add ' + row['type'] + ' to receipts-dict.json manually')
        exit(1)
    if(row['eventType'] not in eventTypes):
        print('Event type "' + row['type'] + '" not mapped')
        print('Run receipts-dict-gen.py or add ' + row['eventType'] + ' to receipts-dict.json manually')
        exit(1)
    speciality[specialties[row['specialty']]] = 1
    typ = np.zeros(len(types), dtype=int)
    typ[types[row['type']]] = 1
    eventType = np.zeros(len(eventTypes), dtype=int)
    eventType[eventTypes[row['eventType']]] = 1
    
    vector = np.concatenate((vector, speciality,eventType,typ))
    
    features.append(vector)
    labels.append(row['price'])

print((str)(len(features)) + " usable training entries")

training_epochs = 50000

labels = np.array(labels).astype(float)

fakedata_file = '../bin/receipts-fake.json'
if(os.path.exists(fakedata_file)):
    with open(fakedata_file, 'r') as fakefile:
        fileobj = json.loads(fakefile.read())
        fakedata = fileobj['features']
        fakeprices = fileobj['labels']
        labels = np.concatenate((labels, fakeprices))
        features = np.concatenate((features, fakedata))
        
    print("Plus " + (str)(len(fakedata)) + " fake entries")

print(len(labels), len(features))

reg = LinearRegression().fit(features,labels)
print(reg.score(features, labels))
predictions = reg.predict(features)
print(mean_squared_error(labels, predictions))

model_file = home + '/dynprice/bin/receipts-model.pkl'

with open(model_file, 'wb') as dumpfile:
    pkl.dump(reg, dumpfile)
    print("Created new model at " + model_file)



##fig, ax = plt.subplots()
##ax.scatter(labels, predictions)
##ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=3)
##ax.set_xlabel('Measured')
##ax.set_ylabel('Predicted')
##plt.show()
