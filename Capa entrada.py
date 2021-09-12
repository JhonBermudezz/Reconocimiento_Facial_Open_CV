import cv2 as cv
import os 

modelos='FotosNombre'#Nombre del modelo al cual se le va estudiar su rostro a cada rostro se cambia el nombre y se incia de nuevo el programa
ruta1= 'C:/'#Ruta donde se van a crear las carpetas y cargar con las imagenes capturadas de preferiencia donde hayas guardado este archivo python
rutacompleta= ruta1+'/'+ modelos
if not os.path.exists(rutacompleta):
    os.makedirs(rutacompleta)


camara=cv.VideoCapture(1)#
ruidos=cv.CascadeClassifier(cv.data.haarcascades +"haarcascade_frontalface_default.xml")
id=0
while True:
    rv,captura=camara.read()
    if rv==False:break
    grises=cv.cvtColor(captura,cv.COLOR_BGR2GRAY)
    cara= ruidos.detectMultiScale(grises,1.2,5)#jugar con el 2 poner mas o menos, calibrarlo segun te funcione
    idcaptura=captura.copy()
    for(x,y,e1,e2) in cara:
        cv.rectangle(captura,(x,y),(x+e1,y+e2),(0,255,0),2)#Color RGB el ultimo digito es el grosor
        capturado=idcaptura[y:y+e1,x:x+e2]
        capturado=cv.resize(capturado,(160,160),interpolation=cv.INTER_CUBIC)
        cv.imwrite(rutacompleta+'/imagen_{}.jpg'.format(id),capturado)#ruta donde se van a guardar
        id=id+1
    cv.imshow("Resultado Rostro",captura)
    if id==351:#Limites de fotos a estudiar que serian 350 puedes variar, poner mas o poner menos
        break
camara.release()
cv.destroyAllWindows()
