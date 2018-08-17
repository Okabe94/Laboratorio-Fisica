# Organizar los imports
import cv2
import imutils
import numpy as np

# Variables globales
bg = None

#-------------------------------------------------------------------------------
# Function - Para separar el segundo plano del primero
#-------------------------------------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # Inicializar el segundo plano
    if bg is None:
        bg = image.copy().astype("float")
        return

    # Computar el promedio, acumularlo y actualizar el segundo plano
    cv2.accumulateWeighted(image, bg, aWeight)

#-------------------------------------------------------------------------------
# Function - Segmentar la región de la mano en la imagen
#-------------------------------------------------------------------------------
def segment(image, threshold=25):
    global bg
    # Encontrar la diferencia absoluta entre el segundo plano y el cuadro actual
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # Resaltar la diferencia en la imagen para obtener el primer plano
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # Obtener los contornos en la imagen resltada
    (_, cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Entregar Null si no se encuentran contornos
    if len(cnts) == 0:
        return
    else:
        # Basado en el contorno del área obtener la máxima, en este caso, la mano
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    # Inicializar el peso para sacar los promedios
    aWeight = 0.5

    # Obtener referencias de la webcam
    camera = cv2.VideoCapture(0)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
    space = 125

    # Coordinadas de la ROI(Region de interes)
    top, right, bottom, left = (height-space), (width-space),(height+space),(width+space)

    # Inicializar el numero de cuadros
    num_frames = 0

    # Ciclo continuo hasta ser interrumpido
    while(True):
        # Obtener el cuadro actual
        (grabbed, frame) = camera.read()

        # Cambiar tamaño del cuadro
        frame = imutils.resize(frame, width=700)

        # Rotar hasta no estar invertido
        frame = cv2.flip(frame, 1)

        # Clonar el cuadro
        clone = frame.copy()

        # Obtener el largo y ancho del cuadro
        (height, width) = frame.shape[:2]

        # Obtener el ROI
        roi = frame[top:bottom, right:left]

        # Convertir el ROI a escala de grises y difuminar
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Para obtener el segundo plano, seguir mirando hasta obtener pasar el punto de critico
        # Para calibrar nuestro modelo de promedios
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # Segmentar la region de la mano
            hand = segment(gray)

            # Asegurar que la region de la mano este segmentada
            if hand is not None:
                # De estarlo, mostrar la imagen limitada y la region segmentada
                (thresholded, segmented) = hand

                # Dibujar el segmento y mostrar el cuadro
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

        # Dibujar la mano segmentada
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # Incrementar el numero de cuadros
        num_frames += 1

        # Mostrar el cuadro con la mano segmentada
        cv2.imshow("Video Feed", clone)

        # Esperar por una tecla presionada
        keypress = cv2.waitKey(1) & 0xFF

        # Si el usuario presiona "q", detener
        if keypress == ord("q"):
            break

# Liberar memoria
camera.release()
cv2.destroyAllWindows()
