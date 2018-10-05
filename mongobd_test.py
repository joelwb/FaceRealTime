import train_data
from pymongo import MongoClient


cliente = MongoClient('localhost', 27017)
banco = cliente.face_encodings

group = banco.group

collections, X, y = train_data.get_data("","fotos_pessoas",True)

group.insert_one(collections)


try:
    doc = group.find({"person_group":{ "$exists": True}}).next()
    print(doc.pretty())
    #group.delete_one(doc)

except:
    pass


#doc = group.find().next()