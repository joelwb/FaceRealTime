import math
from sklearn import neighbors
import pickle
import face_recognition

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_data, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Treina um classificador

    :param model_save_path: (opcional) caminho em que o modelo será salvo
    :param n_neighbors: (opcional) número de neighbors, se não for informado será calculado
    :param knn_algo: (opcional) algoritmo que será usado
    :param verbose: (opcional) define se serão feitos prints
    :return: retorna o classificador treinado com os dados do train_data.
    """
    # Calcula o número de neighbors se se não for informado
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(train_data[0]))))
        if verbose: # Printa o valor calculado se verbose = True
            print("Chose n_neighbors automatically:", n_neighbors)

    # Cria e treina o classificador
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(train_data[0], train_data[1])

    # Salva o classificador no arquivo informado
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(np_img, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Identifica imagem as faces usando um classificador knn já treinado

    :param np_img: imagem que será usada para identificar as faces
    :param knn_clf: (opcional) um classificador knn. Se não for especificado, o model_path precisa ser informado
    :param model_path: (opcional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (opcional) limiar que define o quão similar as faces devem ser para o match. 
    Quanto maior, menos rígido, mas esse valor não é linear e nem porcentagem
    :return: retorna uma lista com os nomes e a localização das faces encontradas na imagem: [(nome, localizacao da face), ...].
        Faces não reconhecidas, será colocado '?' no nome.
    """

    # Lança exceção caso knn e o model path não forem informados
    if knn_clf is None and model_path is None:
        raise Exception("Informe o classificador knn ou o caminho do modelo")

    # Se o knn for None carrega o modelo do arquivo
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Pega as localizações das faces
    faces_locations = face_recognition.face_locations(np_img)

    # Se não achar nenhuma face, retorna lista vazia
    if len(faces_locations) == 0:
        return []

    #Pega os 'encodings' das faces
    faces_encodings = face_recognition.face_encodings(np_img, faces_locations)

    # Usa o knn para fazer os matches baseado na rigidez do limiar informado
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(faces_encodings))]

    # Faz a classificação
    return [(pred, loc) if rec else ("?", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), faces_locations, are_matches)]


