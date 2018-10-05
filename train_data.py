import face_recognition
import os
import os.path
import shutil
import numpy as np
import time
import psycopg2
from random import randint
from face_recognition.face_recognition_cli import image_files_in_folder

con = psycopg2.connect(host='localhost', database='IAmHere', user='postgres', password='leds123')
cursor = con.cursor()


def get_train_data(train_dir,verbose=False):
    """
    :param train_dir: Diretorio que possui pastas com o nome da pessoas, e dentro delas as imagens de cada uma


     Estrutura:
        <train_dir>/
        ├── <pessoa1>/
        │   ├── <img1>.jpeg
        │   ├── <img2>.jpeg
        │   ├── ...
        ├── <pessoa2>/
        │   ├── <img1>.jpeg
        │   └── <img2>.jpeg
        └── ...
    """

    # X é uma lista de 'encodings' dos rostos das pessoas
    # Y é uma lista com o nome das pessoas
    # 

    # Se existir os arquivos com os dados, carrega os dados
    if os.path.isfile("X.npy"): X = np.load("X.npy").tolist()
    else: X = []

    if os.path.isfile("y.npy"): y = np.load("y.npy").tolist()
    else: y = []
 
    
    # Passa por todas a pessoas
    for class_dir in os.listdir(train_dir):

        #Se não for um diretório pula para o próximo loop
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Passa por cada imagem da pessoa
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):

            # Carrega a imagem, descobre as faces e seus limites 
            # Ex: [(x1,y1,w1,h1), (x2,y2,w2,h2), ...]
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            # Verifica se a quantidade de faces reconhecidas foi diferente de 1
            if len(face_bounding_boxes) != 1:
                # Ignora imagem e se verbose = true, faz um print
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Faz um 'enconding da face e adiciona na lista de faces'
                enconding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]

                X.append(enconding)
                # Adiciona o nome da pessoa na lista de nome de pessoas
                y.append(class_dir)

        #Apaga a pasta quando terminar de processa-lá
        shutil.rmtree(train_dir+"/"+class_dir)

    # Salva a lista em arquivos
    np.save("X.npy",X)
    np.save("y.npy",y)
    
    return X, y

def get_data(parent, collection, verbose):
    path = os.path.join(parent, collection)
    
    collection_dict = {}
    X, y = [], []
    if os.path.isdir(path):
        collection_path = path.replace("/",".") if parent != "" else collection
        sql = "INSERT INTO collection (nome, path) VALUES ('%s','%s')" % (collection, collection_path)
        sql = sql + " RETURNING id;"
        cursor.execute(sql)
        id_collection = cursor.fetchone()[0]

        con.commit()
        
        for subcollection in os.listdir(path):
            sub_dict, sub_x, sub_y = get_data(path, subcollection, verbose)

            X, y = X + sub_x, y + sub_y

            if (len(sub_dict) == 0): 
                collection_dict, sub_x, sub_y = get_person_encondings(id_collection,collection,path,verbose)
                X, y = X + sub_x, y + sub_y
                break
            
            else:
                collection_dict[subcollection] = sub_dict
        
        if (parent != ""): 
            return collection_dict, X, y
        else:
            return {"person_group":collection_dict}, X, y
    
    else:
        return {}, [], []

def get_person_encondings(fk_collection, collection, path, verbose):
    doc = []
    X = []


    for img_path in image_files_in_folder(path):
        # Carrega a imagem, descobre as faces e seus limites 
        # Ex: [(x1,y1,w1,h1), (x2,y2,w2,h2), ...]
        image = face_recognition.load_image_file(img_path)
        face_bounding_boxes = face_recognition.face_locations(image)

        # Verifica se a quantidade de faces reconhecidas foi diferente de 1
        if len(face_bounding_boxes) != 1:
            # Ignora imagem e se verbose = true, faz um print
            if verbose:
                print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
        else:
            # Faz um 'enconding da face e adiciona na lista de faces'
            enconding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]

            doc.append(enconding.tolist())

            str_enc = str(enconding.tolist()).replace("[","{").replace("]","}")
            sql = "INSERT INTO face_encoding (fk_collection, encod) VALUES "
            sql = sql + ("(%d" % fk_collection) + ",'" + str_enc + "')"

            cursor.execute(sql)
            con.commit()

            X.append(enconding)

    return doc, X, [collection]  * len(X)

def get_data_from_db(collection):
    sql = "SELECT nome, encod FROM face_encoding "
    sql = sql + "INNER JOIN collection on (face_encoding.fk_collection = collection.id) "
    sql = sql + "WHERE '" + collection +"' @> path"

    cursor.execute(sql)
    dados = cursor.fetchall()

    return[sublist[1] for sublist in dados], [sublist[0] for sublist in dados]