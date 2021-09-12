import cv2 as cv
import os 
import numpy as np
dataRuta=r'C:'#ruta de la carpeta que guardara los datos de los rostros capturados carpeta data
listData=os.listdir(dataRuta)
#print('data',listData)
ids=[]
rostrosData=[]
id=0
for fila in listData:
    rutacompleta=dataRuta+'/'+fila
    for archivo in os.listdir(rutacompleta):
        ids.append(id)
        rostrosData.append(cv.imread(rutacompleta+'/'+archivo,0))
       # imagen=cv.imread(rutacompleta+'/'+archivo,0)
    id=id+1
entrenamientomodelo1=cv.face.EigenFaceRecognizer_create()
print('iniciando el entrenamiento..espere')
entrenamientomodelo1.train(rostrosData,np.array(ids))
entrenamientomodelo1.write('Entrenamiento.xml')#nombre del archivo xml que seran los rostros ya entrenados
print('Entrenamiento Cumplido')