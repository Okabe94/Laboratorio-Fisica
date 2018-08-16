# organizar los imports
import cv2
import imutils
import numpy as np

# variables globales
bg = None

#-------------------------------------------------------------------------------
# Function - Para separar el segundo plano del primero
#-------------------------------------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # inicializar el segundo plano
    if bg is None:
        bg = image.copy().astype("float")
        return

    # Computar el promedio, acumularlo y actualizar el segundo plano
    cv2.accumulateWeighted(image, bg, aWeight)
