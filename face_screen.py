import cv2
import numpy as np
from mss import mss
from PIL import Image
import knn_learner
import ln_learner
import train_data
import time
import pyautogui

"""
                        #####IMPORTANTE#####

    Quanto mais faces encontradas nas imagens mais pesado fica o processamento
    Sendo a parte mais pesada é fazer o reconhecimento das faces e não fazer os matches
"""


process_this_frame = True

# Instancia o capturador de tela
sct = mss()

# Escala da tela que sera processado em porcentagem
scale = 1

# Define a fonte dos nomes e FPS
font = cv2.FONT_HERSHEY_DUPLEX

# Descobre a resolução da tela
screen_width, screen_height =  pyautogui.size()


# Retorna a região da tela que será capturada, centralizando na posição do mouse
def get_box_capture():
    # Descobre a posição do mouse
    mouse_x, mouse_y = pyautogui.position()

    # Define o tamanho da tela que sera capturado
    w,h = 640,480

    # Centraliza o mouse
    x,y = int(mouse_x-(w/2)), int(mouse_y-(h/2))

    # Faz verficações para evitar problemas com as bordas da tela
    if x < 0: x = 0
    elif x + w > screen_width: x = screen_width-w

    if y < 0: y = 0
    elif y + h > screen_height: y = screen_height-h

    # Região da tela que deve ser capturada
    return {"top": y, "left": x, "width": w, "height": h}


# Captura um frame e retorna a imagem original e a imagem redimensionada
def captureScreen():
    # Pega um frame de uma região da tela
    box = get_box_capture()
    frame = sct.grab(box)

    orig_img = np.array(frame)
    np_img = np.array(Image.frombytes('RGB', frame.size, frame.rgb))

    # Redimensiona imagem de acordo com a escala
    small_frame = cv2.resize(np_img, (0, 0), fx=scale, fy=scale)

    # Converte imagem de BGR para RGB
    rgb_small_frame = small_frame[:, :, ::-1]

    return rgb_small_frame, orig_img


# Renderiza o quadrado e o nome em cada face
def show_faces(predictions, img):
    """
    :param predictions: lista com nomes e posição da face. Ex: [(nome1, (top,right,bottom,left), ...)]
    :param img: imagem que será feito os desenhos
    :return: None 
    """

    for name, (top, right, bottom, left) in predictions:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= int(1/scale)
        right *= int(1/scale)
        bottom *= int(1/scale)
        left *= int(1/scale)

        # Desenha o quadrado na face
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

        # Desenha o nome na face
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


if __name__ == "__main__":
    # Pega os dados para treinar o classificador 
    print("Getting train data")
    train_data = train_data.get_train_data("fotos_pessoas")

    # Treina o classificador
    print("Trainando o classificador...")
    classifier = knn_learner.train(train_data, model_save_path="trained_knn_model.clf", verbose=True)
    # classifier = ln_learner.train(train_data,"trained_knn_model.clf",True)
    print("Treino completo!")

    start = time.time()
    cont = 0 # Contador de quadros processados a cada segundo
    fps = ""
    while True:
        resized_img, orig_img = captureScreen()

        # Processa frame sim, frame não - pode ajudar no desempenho
        if process_this_frame:
            cont += 1

            # Analiza a imagem reduzida e identifica as pessoas
            predictions = knn_learner.predict(resized_img, knn_clf=classifier, model_path="trained_knn_model.clf")
            #predictions = ln_learner.predict(resized_img,classifier,"trained_knn_model.clf")

            # Renderiza na imagem original quadrados nos rostos das pessoas e seu nome
            show_faces(predictions, orig_img)

            # Calcula FPS a cada 1 segundo e reseta variaveis
            final_time = time.time()
            if (final_time - start >= 1):
                fps = "{:.2f}".format(cont / (final_time - start))
                cont = 0
                start = time.time()

            # Exibe na tela o FPS
            cv2.putText(orig_img, fps+" FPS", (16, 30), font, 1.0, (255, 0, 0), 2)

            # Exibe a imagem original contendo FPS, regiões dos rostos e seu nome
            cv2.imshow('Video', orig_img)
            
        #alterna se irá ou não processar o próximo frame
        process_this_frame = not process_this_frame

        # Quando a tecla 'q' for presionada sai do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Fecha todas as janelas do opencv
    cv2.destroyAllWindows()
