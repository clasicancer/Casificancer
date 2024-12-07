import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

class CancerClassifier:
    def __init__(self, model_path: str):
        """
        Inicializa el clasificador cargando el modelo.

        :param model_path: Ruta del archivo del modelo (.h5).
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocesa una imagen para hacerla compatible con el modelo.

        :param image: Imagen en formato PIL.
        :return: Imagen preprocesada como un array de NumPy.
        """
        image = image.resize((224, 224))  # Redimensionar la imagen
        image = img_to_array(image)  # Convertir a array
        image = np.expand_dims(image, axis=0)  # Añadir dimensión para batch
        return tf.keras.applications.resnet50.preprocess_input(image)  # Preprocesar para ResNet50

    def predict(self, img_array: np.ndarray) -> dict:
        """
        Realiza una predicción en la imagen preprocesada.

        :param img_array: Imagen preprocesada como un array de NumPy.
        :return: Diccionario con el resultado de la predicción.
        """
        # Hacer predicción con el modelo
        logits = self.model.predict(img_array)  # Logits del modelo

        # Aplicar softmax para convertir logits en probabilidades
        probabilities = tf.nn.softmax(logits[0]).numpy()

        # Interpretar resultados
        class_names = ['Benigno', 'Maligno']  # Clase 0 y Clase 1 respectivamente
        predicted_class = np.argmax(probabilities)  # Índice de la clase con mayor probabilidad
        confidence = probabilities[predicted_class]  # Probabilidad de la clase predicha

        # Retornar resultado en formato JSON-friendly
        return {
            'resultado': f"Esta imagen parece ser {class_names[predicted_class]} "
            f"con un {confidence * 100:.2f}% de confianza."
        }
