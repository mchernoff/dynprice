import bson
import json

with open('./receipts.bson','rb') as file:
    data = bson.decode_all(file.read())

specialties = {}
eventTypes = {}
types = {}
for row in data:
    if(row['specialty'] not in specialties and row['specialty'] is not None):
        specialties[row['specialty']] = len(specialties)
    if(row['eventType'] not in eventTypes and row['eventType'] is not None):
        eventTypes[row['eventType']] = len(eventTypes)
    if(row['type'] not in types and row['type'] is not None):
        types[row['type']] = len(types)

obj = {
    "specialties": specialties,
    "eventTypes": eventTypes,
    "types": types,
}

jsontxt = json.dumps(obj)

with open('./receipts-dict.json', 'w') as file:
    file.write(jsontxt)
