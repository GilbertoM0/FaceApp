from deepface import DeepFace
import cv2

# Cargar una imagen de prueba (coloca la imagen en el mismo directorio del script)
image_path = "./img/test_image.webp"  # Cambia este nombre por el de tu archivo de imagen
image = cv2.imread(image_path)

# Mostrar la imagen
cv2.imshow("Imagen", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Analizar emociones con DeepFace
analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'])

# Mostrar los resultados
print("Resultados del análisis de emociones:")
print(analysis[0]["emotion"])

