import bson
from dateutil import parser
import pickle as pkl
import json
import numpy as np
from sklearn.metrics import mean_squared_error
from dateutil import parser
import random
import os

home = os.getenv("HOME")

with open(home + '/dump/porton/receipts.bson','rb') as infile:
    data = bson.decode_all(infile.read())

with open('../bin/receipts-dict.json', 'r') as dictfile:
    obj = json.loads(dictfile.read())
    specialties = obj['specialties']
    eventTypes = obj['eventTypes']
    types = obj['types']


newsize = 500
newentries = []

features = []
labels = []
    
model_file = '../bin/receipts-model.pkl'
with open(model_file, 'rb') as dumpfile:
    reg = pkl.load(dumpfile)


minDuration = -1
maxDuration = 0


for row in data:
    try:
        duration = (parser.parse(row['end']) - parser.parse(row['start'])).seconds
        if(minDuration < 0):
            minDuration = duration
        elif(minDuration > duration):
            minDuration = duration
        maxDuration = max(maxDuration, duration)
    except:
        duration = None
        continue
    
prices = []

for i in range(0, newsize):
    duration = random.randint(minDuration, maxDuration)
    newrow = np.asarray([duration])
    
    type_choice = random.choice(list(types.values()))
    eventType_choice = random.choice(list(eventTypes.values()))
    specialty_choice = random.choice(list(specialties.values()))

    specialty = np.zeros(len(specialties), dtype=int)
    specialty[specialties[row['specialty']]] = 1
    typ = np.zeros(len(types), dtype=int)
    typ[types[row['type']]] = 1
    eventType = np.zeros(len(eventTypes), dtype=int)
    eventType[eventTypes[row['eventType']]] = 1

    newrow = np.concatenate((newrow, specialty,eventType,typ))    
    price = reg.predict(newrow.reshape(1,-1))[0]
    prices.append(price)
    newentries.append(newrow.tolist())


fileobj = {
            'features': newentries,
            'labels': prices
        }
fakefile = '../bin/receipts-fake.json'
with open(fakefile, 'w') as outfile:
    outfile.write(json.dumps(fileobj))
    print((str)(newsize) + " fake entries created in " + fakefile)
