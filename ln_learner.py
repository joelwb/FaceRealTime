import math
import pickle
import os
import face_recognition
from sklearn import linear_model
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_data,model_save_path=None, verbose=False, loss="modified_huber"):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    # Create and train the KNN classifier
    ln_clf = linear_model.SGDClassifier(loss=loss)
    ln_clf.fit(train_data[0], train_data[1])

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(ln_clf, f)

    return ln_clf

def predict(np_img, ln_clf=None, model_path=None, threshold=0.96):
    """
    Identifica imagem as faces usando um classificador linear já treinado

    :param np_img: imagem que será usada para identificar as faces
    :param ln_clf: (opcional) um classificador knn. Se não for especificado, o model_path precisa ser informado
    :param model_path: (opcional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param threshold: (opcional) limiar que define o quão similar as faces devem ser para o match. 
    Quanto maior, mais rígido, além de ser em porcentagem
    :return: retorna uma lista com os nomes e a localização das faces encontradas na imagem: [(nome, localizacao da face), ...].
        Faces não reconhecidas, será colocado '?' no nome.
    """
    if ln_clf is None and model_path is None:
        raise Exception("Informe o classificador linear ou o caminho do modelo")

    # Load a trained KNN model (if one was passed in)
    if ln_clf is None:
        with open(model_path, 'rb') as f:
            ln_clf = pickle.load(f)

    # Se o classificador for None carrega o modelo do arquivo
    faces_locations = face_recognition.face_locations(np_img)

    # Se não achar nenhuma face, retorna lista vazia
    if len(faces_locations) == 0:
        return []

    #Pega os 'encodings' das faces
    faces_encodings = face_recognition.face_encodings(np_img, faces_locations)

    # Usa o classificador calcular a similaridade das faces com cada pessoa
    # Se a similaridade mais alta for maior que limiar definido, faz um match
    predict = ln_clf.predict_proba(faces_encodings)
    are_matches = [max(predict[i]) >= threshold for i in range(len(faces_encodings))]
    
    # Faz a classificação
    return [(pred, loc) if rec else ("?", loc) for pred, loc, rec in zip(ln_clf.predict(faces_encodings), faces_locations, are_matches)]
