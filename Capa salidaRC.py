import cv2 as cv
import os
import numpy as np
dataRuta=r'C:'#Directorio donde esta la carpeta creada de las datas de los rostros
listData=os.listdir(dataRuta)
entrenamientomodelo1=cv.face.EigenFaceRecognizer_create()
entrenamientomodelo1.read('archivo.xml')#aca pones el archivo xml del entranamiento antes hecho
ruidos=cv.CascadeClassifier(cv.data.haarcascades +"haarcascade_frontalface_default.xml")#esto son los ruidos
camara=cv.VideoCapture(1)#1 si es camara externa y 0 si es la camara de tu laptop
while True:
    _,captura=camara.read()
    grises= cv.cvtColor(captura,cv.COLOR_BGR2GRAY)
    idcaptura=grises.copy()
    cara= ruidos.detectMultiScale(grises,1.2,5)
    for(x,y,e1,e2) in cara:
        cv.rectangle(captura,(x,y),(x+e1,y+e2),(0,255,0),2)#Color RGB el ultimo digito es el grosor
        capturado=idcaptura[y:y+e1,x:x+e2]
        capturado=cv.resize(capturado,(160,160),interpolation=cv.INTER_CUBIC)
        resultado=entrenamientomodelo1.predict(captura)
        cv.putText(captura,'{}'.format(resultado),(x,y-6),1,1.2,(0,255,0),1,cv.LINE_AA)
        if resultado[1]<9000:#El rango de los numeros producidos en tu archivo xml
            cv.putText(captura,"No encontrado",(x,y-20),1,0.7,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura,(x,y),(x+e1,y+e2),(0,255,0),2)
        else:
            cv.putText(captura,'{}'.format(listData[resultado[0]]),(x,y-20),1,1.2,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura,(x,y),(x+e1,y+e2),(0,255,255),2)
    cv.imshow('Resultado',captura)
    if(cv.waitKey(1)==ord('s')):# para frenar el programa 's' pero tu puedes cambiar de letra
        break
camara.release()
cv.destroyAllWindows

