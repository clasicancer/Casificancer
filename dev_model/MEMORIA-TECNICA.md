# Memoria Técnica

## Portada
- **Nombre del Proyecto**: Detector Automatizado de Tumores Cáncer de Mama con Redes Neuronales Convolucionales
- **Fecha**: 5 de diciembre de 2024.
- **Integrantes**:  
  - Leslie Guadalupe Pérez Montes
  - Marían de los Ángeles Partida Contreras
  - José Emanuel Tellez Soto

![Lucha](../images/lucha-contra-cancer1.jpg)

## Índice
- [Portada](#portada)
- [Alcance del proyecto](#alcance-del-proyecto)
  - [Objetivo](#objetivo)
  - [Introducción](#introducción)
- [Fuentes de información y procedimientos aplicados](#fuentes-de-información-y-procedimientos-aplicados)
  - [Construcción del modelo](#construcción-del-modelo)
  - [Resultados modelo](#resultados-modelo)
  - [Pruebas sobre el modelo](#pruebas-sobre-el-modelo)
  - [Conclusiones](#conclusiones)
- [Conclusiones generales](#conclusiones-generales)
- [Anexos](#anexos)
- [Glosario](#glosario)

## Alcance del proyecto

### Objetivo
El propósito central del proyecto es desarrollar un modelo automatizado basado en Redes Neuronales Convolucionales (CNN) para clasificar imágenes histopatológicas en dos categorías: "maligno" y "benigno". Este sistema está diseñado para asistir a profesionales de la salud en el diagnóstico temprano del cáncer de mama, reduciendo el tiempo de análisis y mejorando la precisión en la detección de tumores. El proyecto busca contribuir al ámbito médico mediante la integración de técnicas avanzadas de inteligencia artificial en el proceso de diagnóstico.

### Introducción

El cáncer de mama es una de las principales causas de muerte en mujeres a nivel mundial. El diagnóstico temprano y preciso es esencial para mejorar las probabilidades de tratamiento exitoso y supervivencia. Tradicionalmente, el diagnóstico se basa en la evaluación manual de imágenes histopatológicas bajo microscopio, un proceso que requiere experiencia y tiempo y que a menudo está sujeto a variabilidad en la interpretación.

Los avances en inteligencia artificial, particularmente en deep learning, han abierto nuevas posibilidades para automatizar y mejorar la precisión del diagnóstico. Este proyecto implementa una **Red Neuronal Convolucional (CNN)**, una arquitectura especializada en el procesamiento de datos visuales para construir un clasificador de imágenes histopatológicas que detecte automáticamente tumores benignos y malignos en muestras de tejido mamario, optimizando el diagnóstico y reduciendo la carga de trabajo del personal médico.


## Fuentes de información y procedimientos aplicados

### **MODELO 1**
El conjunto de datos, descargado de BreakHis, pertenece a la clasificación de imágenes histopatológicas del cáncer de mama. Este conjunto de datos esta compuesto por los siguientes archivos:

  - **images**.
    - `imagenes_benigno`: Imagenes benignas.
    - `imagenes_maligno`: Imagenes malignas. 
    - Total de imágenes: 7954
---

## Pipeline de Preparación
### 1. Crear Etiquetas Binarias
- Clasificamos las imágenes basándonos en si son benignas o malignas.
  ![Distribucion binaria](../images/Proportion1.png)
### 2. División del Conjunto de Datos
- Los datos se dividen en entrenamiento y validación con proporciones del 66% y 31% respectivamente.

  | Conjunto         | Cantidad de Imágenes |
  |-------------------|----------------------|
  | Entrenamiento     | 6364               |
  | Validación        | 1590                 |
  

  - **Total de imágenes:** 7954
  - **Etiquetas:**
    - **Imágenes benignas:** 1976 imágenes
    - **Imágenes malignas:** 4388 imágenes

### 3. Preprocesamiento
Antes de usar las imágenes, se realiza un escalado de sus píxeles para mejorar el rendimiento del modelo:
- **Escalado:** Los valores de los píxeles se convierten de `[0, 255]` a `[0, 1]`.

### 4. Manejo de desequilibrio entre clases
Cuando una clase tiene más ejemplos que otra, el modelo puede inclinarse a favorecer la clase más frecuente. Para evitarlo:

1. Se calculan pesos de clase que equilibran la importancia de ambas clases.
2. Estos pesospenalizan al modelo por cometer errores en la clase menos representada.


---



## Construcción del MODELO 1

La red que construiremos se basa en una arquitectura de **red neuronal convolucional (CNN)**, una técnica ideal para problemas de visión por computadora. Las CNNs son capaces de extraer características clave de las imágenes, como texturas, bordes y patrones complejos, que son esenciales identificar tumores malignos y benignos en muestras de tejido mamario.

Este fue el primer modelo que hicimos, el cual tiene tres capas convolucionales y tiene una función de perdida.
### Aquitectura

| **Layer (type)**           | **Output Shape**         | **Param #** |
|----------------------------|--------------------------|-------------|
| `rescaling (Rescaling)`     | (None, 224, 224, 3)     | 0           |
| `conv2d (Conv2D)`          | (None, 224, 224, 16)    | 448         |
| `max_pooling2d (MaxPooling2D)` | (None, 112, 112, 16)    | 0           |
| `conv2d_1 (Conv2D)`        | (None, 112, 112, 32)    | 4,460       |
| `max_pooling2d_1 (MaxPooling2D)` | (None, 56, 56, 32)     | 0           |
| `conv2d_2 (Conv2D)`        | (None, 56, 56, 64)      | 18,496      |
| `max_pooling2d_2 (MaxPooling2D)`        | (28,28, 64)          | 0           |
| `dropout (Dropout)`          | (28,28, 64)               | 0         |
| `flatten (Flatten)`            | (None, 50176)             | 0  |
| `dense (Dense)`          | (None, 128)               | 6,422,656         |
| `dense_1 (Dense)`          | (None, 2)               | 258         |

#### **Totales**
- **Total params:** 6,446,498
- **Trainable params:** 6,446,498
- **Non-trainable params:** 0

### Funcionamiento del Modelo
1. Las imágenes se pasan a través de tres capas convolucionales para extraer características espaciales.
1. Las capas de MaxPooling reducen las dimensiones de las características.
1. La capa de Flatten transforma los datos en un vector plano.
1. Las capas densas realizan la clasificación basada en las características extraídas.
1. Capa de salida. Una sola neurona con una función de activación sigmoide produce una probabilidad entre 0 y 1.

### Justificación 

Esta red no es adecuada para la tarea de clasificación de tumores benignos y malignos porque:

- Tiene solo tres capas convolucionales y tiene una función de pérdida

## Resultados modelo

El modelo muestra signos claros de sobreajuste.

![trainvstest](../images/TrainVS.png)

### **Precisión (Accuracy)**:

  - **Entrenamiento**: La precisión no alcanza valores cercanos a 1.0 rápidamente, indicando un ajuste malo a los datos de entrenamiento.
  - **Validación**: La precisión de validación es 0.8575.

### **Pérdida (Loss)**:

- **Entrenamiento**: La pérdida disminuye en 0.4052
- **Validación**: La pérdida de validación es baja, pero fluctúa ligeramente a partir de la mitad del entrenamiento, lo que sugiere que el modelo podría beneficiarse de técnicas adicionales de regularización para mayor estabilidad.
 

## Pruebas sobre el modelo

### **Matriz de Confusión**

![Matriz de confusión](../images/matrix.png)

## **Métricas de Evaluación**

### **1. Precisión (Precision)**
De todas las predicciones como "Malaria", el 98.9% fueron correctas. Una alta precisión significa que el modelo tiene una baja tasa de falsos positivos

### **2. Recall (Sensibilidad)**

El modelo identificó correctamente el 99.8% de los casos de malaria. Esto es crucial en diagnósticos médicos, ya que minimiza los casos de malaria no detectados.

### **3. F1-Score**

Un F1-Score de 0.994 indica un balance excelente entre precisión y recall, lo que demuestra que el modelo es confiable y robusto para detectar malaria.

## Conclusiones

- El modelo tiene un excelente desempeño tanto en entrenamiento como en validación, y parece generalizar bien. Las oscilaciones en las métricas pueden abordarse con pequeños ajustes, pero no afectan significativamente el rendimiento general.

- El modelo es altamente efectivo para detectar malaria en imágenes, con métricas que reflejan una excelente precisión y sensibilidad. Este rendimiento lo hace adecuado para aplicaciones en entornos clínicos, donde el diagnóstico rápido y preciso es crucial.

### **Modelo 2**


### Modelo 1
El conjunto de datos, descargado de BreakHis, pertenece a la clasificación de imágenes histopatológicas del cáncer de mama. Este conjunto de datos esta compuesto por los siguientes archivos y carpetas:

- **images**.
Este directorio contiene las imágenes utilizadas para la tarea de clasificación. Incluye imágenes microscópicas de tejido tumoral de mama recogidas de 82 pacientes utilizando diferentes factores de aumento (40X, 100X, 200X y 400X).

- **SampleSubmission.csv**.
Este archivo es un ejemplo del formato esperado para las predicciones de clasificación.

- **Test.csv**.
Contiene información sobre las imágenes de prueba. 

- **Train.csv**.
Este archivo contiene los datos de entrenamiento, que son una lista de imágenes junto con sus etiquetas correspondientes. 

  - **Columnas:**
    - `Image_ID`: Cada nombre de archivo de imagen almacena información sobre la imagen en sí: método de biopsia del procedimiento, clase tumoral, tipo de tumor, identificación del paciente y factor de aumento.
    - `class`: Clase de la imagen (e.g., bening, para benignas).
    - `ymin`, `xmin`, `ymax`, `xmax`: Coordenadas de un cuadro delimitador (bounding box) para objetos relevantes en la imagen.

  - **Cantidad de filas:** 23,530 entradas, pero solo 2,747 imágenes únicas (indicando múltiples objetos por imagen).

### Distribución de Clases en Train.csv

La gráfica a continuación muestra la distribución de las clases en el conjunto de entrenamiento:

![Distribución Original de Clases](../images/original_distribution.png)

### Observaciones:
- `Trophozoite` tiene la mayoría de los registros.
- Las otras clases, `NEG` y `WBC`, tienen menos ejemplos en comparación.
- Combinaciones: 
  - No hay imágenes que contengan las tres clases simultáneamente.
  - La combinación más común es Trophozoite y WBC, mientras que no hay combinaciones que incluyan NEG con otra clase.
  - Una proporción significativa de imágenes tiene solo una clase (Trophozoite, WBC, o NEG).

![Combinaciones](../images/combinaciones.png)

## Ejemplo de Imágenes con Bounding Boxes

Los **bounding box coordinates** son valores que definen un rectángulo alrededor de un objeto de interés dentro de una imagen. Este rectángulo se utiliza comúnmente en tareas de visión por computadora, como la detección de objetos, para localizar y delimitar objetos específicos dentro de una imagen.

Las imágenes a continuación muestran ejemplos del conjunto de datos con sus respectivas bounding boxes dibujadas.

### Ejemplo 1: Imagen `id_q18tfhfneh.jpg`
![id_q18tfhfneh](../images/id_q18tfhfneh_marked.jpg)

### Ejemplo 2: Imagen `id_zz4ga0557e.jpg`
![id_zz4ga0557e](../images/id_zz4ga0557e_marked.jpg)

### Ejemplo 3: Imagen `id_2pye2ftpl6.jpg`
![id_2pye2ftpl6](../images/id_2pye2ftpl6_marked.jpg)

---

## Pipeline de Preparación
### 1. Crear Etiquetas Binarias
- Clasificamos las imágenes basándonos en si contienen al menos un trofozoíto.
  ![Distribucion binaria](../images/proportion.png)
### 2. División del Conjunto de Datos
- Los datos se dividen en entrenamiento y validación con proporciones del 80% y 20% respectivamente.

  | Conjunto         | Cantidad de Imágenes |
  |-------------------|----------------------|
  | Entrenamiento     | 2,197               |
  | Validación        | 550                 |
  

  - **Total de imágenes:** 2,747
  - **Etiquetas:**
    - **malaria_SI:** 2,018 imágenes
    - **malaria_NO:** 729 imágenes

### 3. Preprocesamiento
Antes de usar las imágenes, se realiza un escalado de sus píxeles para mejorar el rendimiento del modelo:
- **Escalado:** Los valores de los píxeles se convierten de `[0, 255]` a `[0, 1]`.

### 4. Manejo de desequilibrio entre clases
Cuando una clase tiene más ejemplos que otra, el modelo puede inclinarse a favorecer la clase más frecuente. Para evitarlo:

1. Se calculan pesos de clase que equilibran la importancia de ambas clases.
2. Estos pesospenalizan al modelo por cometer errores en la clase menos representada.


---



## Construcción del modelo

La red que construiremos se basa en una arquitectura de **red neuronal convolucional (CNN)**, una técnica ideal para problemas de visión por computadora. Las CNNs son capaces de extraer características clave de las imágenes, como texturas, bordes y patrones complejos, que son esenciales para identificar trofozoítos, glóbulos blancos y otras estructuras relevantes en las imágenes de microscopio.

### Aquitectura

| **Layer (type)**           | **Output Shape**         | **Param #** |
|----------------------------|--------------------------|-------------|
| `input_1 (InputLayer)`     | (None, 224, 224, 3)     | 0           |
| `conv2d (Conv2D)`          | (None, 222, 222, 32)    | 896         |
| `max_pooling2d (MaxPooling2D)` | (None, 111, 111, 32)    | 0           |
| `conv2d_1 (Conv2D)`        | (None, 109, 109, 32)    | 9,248       |
| `max_pooling2d_1 (MaxPooling2D)` | (None, 54, 54, 32)     | 0           |
| `conv2d_2 (Conv2D)`        | (None, 52, 52, 64)      | 18,496      |
| `flatten (Flatten)`        | (None, 173056)          | 0           |
| `dense (Dense)`            | (None, 128)             | 22,151,136  |
| `dense_1 (Dense)`          | (None, 1)               | 129         |

#### **Totales**
- **Total params:** 22,179,905
- **Trainable params:** 22,179,905
- **Non-trainable params:** 0

### Funcionamiento del Modelo
1. Las imágenes se pasan a través de tres capas convolucionales para extraer características espaciales.
1. Las capas de MaxPooling reducen las dimensiones de las características.
1. La capa de Flatten transforma los datos en un vector plano.
1. Las capas densas realizan la clasificación basada en las características extraídas.
1. Capa de salida. Una sola neurona con una función de activación sigmoide produce una probabilidad entre 0 y 1, indicando si la imagen pertenece a la clase positiva (`Malaria_SI`).

### Justificación 

Esta red es adecuada para la tarea de clasificación de malaria porque:

- Extrae características relevantes de las imágenes.
- Reduce la dimensionalidad de manera eficiente.
- Se adapta bien a problemas de clasificación binaria.
- Tiene una estructura simple y eficiente que puede ser mejorada según las necesidades del problema.

## Resultados modelo

El modelo no muestra signos claros de sobreajuste, ya que las métricas de validación son similares a las de entrenamiento.

![trainvstest](../images/trainVSval.png)

### **Precisión (Accuracy)**:

  - **Entrenamiento**: La precisión alcanza valores cercanos a 1.0 rápidamente, indicando un ajuste muy bueno a los datos de entrenamiento.
  - **Validación**: La precisión de validación es alta (~0.97-0.99), pero muestra ligeras oscilaciones en algunas épocas, lo que puede deberse a variaciones en los datos o a la falta de estabilidad.

### **Pérdida (Loss)**:

- **Entrenamiento**: La pérdida disminuye consistentemente y se estabiliza en valores muy bajos (~0.01), lo que refleja que el modelo está aprendiendo adecuadamente.
- **Validación**: La pérdida de validación es baja, pero fluctúa ligeramente a partir de la mitad del entrenamiento, lo que sugiere que el modelo podría beneficiarse de técnicas adicionales de regularización para mayor estabilidad.
 

## Pruebas sobre el modelo

### **Matriz de Confusión**

![Matriz de confusión](../images/matrix.png)

## **Métricas de Evaluación**

### **1. Precisión (Precision)**
De todas las predicciones como "Malaria", el 98.9% fueron correctas. Una alta precisión significa que el modelo tiene una baja tasa de falsos positivos

### **2. Recall (Sensibilidad)**

El modelo identificó correctamente el 99.8% de los casos de malaria. Esto es crucial en diagnósticos médicos, ya que minimiza los casos de malaria no detectados.

### **3. F1-Score**

Un F1-Score de 0.994 indica un balance excelente entre precisión y recall, lo que demuestra que el modelo es confiable y robusto para detectar malaria.

## Conclusiones

- El modelo tiene un excelente desempeño tanto en entrenamiento como en validación, y parece generalizar bien. Las oscilaciones en las métricas pueden abordarse con pequeños ajustes, pero no afectan significativamente el rendimiento general.

- El modelo es altamente efectivo para detectar malaria en imágenes, con métricas que reflejan una excelente precisión y sensibilidad. Este rendimiento lo hace adecuado para aplicaciones en entornos clínicos, donde el diagnóstico rápido y preciso es crucial.



## Conclusiones generales

El clasificador de imágenes histopatológicas del cáncer de mama basado en redes neuronales convolucionales constituye un avance crucial en la automatización del diagnóstico oncológico. Con un alto nivel de precisión y capacidad de generalización, este modelo puede optimizar la detección temprana y el manejo del cáncer de mama, mejorando significativamente la calidad del diagnóstico en contextos médicos, especialmente en regiones con recursos limitados y alta demanda de atención especializada.

Este proyecto demuestra cómo la inteligencia artificial puede revolucionar el diagnóstico del cáncer de mama mediante imágenes histopatológicas:

- **Rápido y Preciso:** El modelo permite identificar tumores benignos y malignos en tiempo real, optimizando la detección temprana y mejorando las tasas de diagnóstico efectivo.
- **Económico:** La automatización reduce la dependencia de especialistas altamente capacitados, haciéndolo viable para clínicas y hospitales con recursos limitados.
- **Escalable:** Esta tecnología puede adaptarse a la clasificación de otras patologías basadas en imágenes, ampliando su impacto en diversas áreas de la medicina.


## Anexos
- [Repositorio Github](https://github.com/clasicancer/Casificancer)
- [Conjunto de Datos para la Detección de Cáncer de Mama](http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz)

## Glosario

Este glosario proporciona una breve descripción de los términos relacionados con la clasificación de cáncer de mama utilizados en el proyecto.

### **1. Cáncer de Mama**
- **Definición:** El cáncer de mama es una enfermedad en la que las células del tejido mamario crecen de manera descontrolada, formando un tumor maligno que puede invadir tejidos cercanos o diseminarse a otras partes del cuerpo a través de la sangre o el sistema linfático.
- **Importancia:** Es el tipo de cáncer más común entre las mujeres a nivel mundial, representando una causa significativa de mortalidad y morbilidad. La detección temprana y el tratamiento adecuado son esenciales para mejorar las tasas de supervivencia y reducir el impacto en la calidad de vida de las pacientes.
### **2. Imágenes Histopatológicas**
- **Definición:** Las imágenes histopatológicas son representaciones visuales de cortes finos de tejido que han sido teñidos y observados bajo un microscopio. Estas imágenes permiten estudiar la arquitectura celular y tisular, identificando características relacionadas con enfermedades, como el cáncer.
- **Importancia:** Son fundamentales en el diagnóstico y clasificación de enfermedades, ya que proporcionan información detallada sobre la estructura y comportamiento celular. En el caso del cáncer, estas imágenes permiten diferenciar entre tejidos normales, benignos y malignos, guiando decisiones clínicas y estrategias de tratamiento.
### **3. Clasificación de Imágenes**
- **Definición:** Proceso automatizado en el que un modelo de aprendizaje profundo, como una red neuronal convolucional (CNN), categoriza imágenes en clases específicas, como "maligno" o "benigno", basándose en patrones detectados en los datos visuales.
### **4. Histología Mamaria**
- **Definición:** Estudio de la estructura microscópica de los tejidos mamarios. Permite identificar diferencias clave entre tejidos normales, benignos y malignos, lo cual es crucial para el entrenamiento de modelos de clasificación.

### **5. Tipos Histológicos de Tumores Mamarios**
- **Definición:** Clasificación de los tumores según la apariencia de las células bajo el microscopio. Ejemplos incluyen adenosis, fibroadenoma, carcinoma ductal y carcinoma lobular. Estas categorías son las etiquetas utilizadas en los modelos de clasificación de imágenes.

### **6. Datos de Entrenamiento**
- **Definición:** Conjunto de imágenes histopatológicas anotadas con información sobre la clase y tipo de tumor. Es esencial para que el modelo aprenda a diferenciar entre tumores benignos y malignos.

### **7. Magnificación (Factor de Aumento)**
- **Definición:** Nivel de zoom aplicado en las imágenes histopatológicas (como 40X, 100X, 200X, 400X). La variación en los factores de aumento permite al modelo analizar características a diferentes escalas y mejorar la precisión de la clasificación.

### **8. Características Microscópicas de Tumores**
- **Definición:** Rasgos visuales observados en las imágenes histopatológicas, como la disposición celular, tamaño de los núcleos y presencia de mitosis. Estas características son clave para entrenar al modelo de clasificación.
### **9. Etiquetado de Imágenes**
- **Definición:** Proceso de asignar una clase y tipo de tumor a cada imagen en el dataset. Este etiquetado es la base para que el modelo aprenda a asociar patrones visuales con categorías específicas.

### **10. Sensibilidad y Especificidad**
- **Definición:**
  - **Sensibilidad:** Capacidad del modelo para identificar correctamente las imágenes de tumores malignos.
  - **Especificidad:** Capacidad del modelo para clasificar correctamente las imágenes de tumores benignos.
- Ambas métricas son críticas para evaluar el rendimiento del modelo.

### **11. Lesión Premaligna**
- **Definición:** Cambios celulares que aún no son malignos pero presentan un riesgo elevado de convertirse en cáncer. Su identificación precisa es importante para la detección temprana y la intervención.
### **12. Métodos de Biopsia**
- **Definición:** Procedimientos para obtener muestras de tejido mamario, como la biopsia excisional (SOB). Estas muestras se utilizan para crear las imágenes histopatológicas empleadas en el modelo de clasificación.
### **13. Red Neuronal Convolucional (CNN)**
- **Definición:** Arquitectura de redes neuronales artificiales diseñada específicamente para procesar datos con estructuras de cuadrícula, como imágenes. Utiliza operaciones de convolución para extraer automáticamente características relevantes.
### **14. Clasificación Binaria**
- **Definición:** Tarea de aprendizaje automático en la que el modelo clasifica los datos en una de dos categorías, como "maligno" o "benigno".
### **15. Tumor Maligno**
- **Definición:** Crecimiento anormal de células que puede invadir tejidos cercanos y propagarse a otras partes del cuerpo.
- **Importancia:** Representa una amenaza significativa para la salud debido a su capacidad de invadir órganos vitales y diseminarse (metástasis), lo que puede conducir a complicaciones graves e incluso la muerte. Su identificación temprana y tratamiento adecuado son esenciales para mejorar los resultados clínicos.
### **16. Tumor Benigno**
- **Definición:** Masa celular no cancerosa que no invade tejidos circundantes ni se disemina a otras áreas del cuerpo.
- **Importancia:** Aunque no representa un riesgo directo de metástasis como los tumores malignos, su crecimiento puede causar complicaciones dependiendo de su tamaño y ubicación, como presión sobre órganos o estructuras cercanas. Además, algunos tumores benignos tienen el potencial de transformarse en malignos, lo que resalta la importancia de su monitoreo y tratamiento oportuno.
### **17. Dataset**
Definición: Conjunto estructurado de datos utilizado para entrenar y evaluar el modelo de clasificación.
### **18. Entrenamiento**
- **Definición:** Proceso mediante el cual la red neuronal aprende patrones en los datos de entrada ajustando los pesos de sus conexiones a través de algoritmos de optimización.
### **19. Validación**
- **Definición:** Proceso de evaluación del rendimiento del modelo en un conjunto de datos separado del de entrenamiento, utilizado para prevenir el sobreajuste y medir la capacidad de generalización.
### **20. Sobreajuste (Overfitting)**
- **Definición:** Situación en la que un modelo se adapta demasiado a los datos de entrenamiento, lo que deteriora su desempeño en datos nuevos o no vistos.
### **21. Precisión (Accuracy)**
- **Definición:** Métrica de evaluación del desempeño de un clasificador, definida como la proporción de clasificaciones correctas entre el total de clasificaciones realizadas.
### **22. Tasa de Falsos Positivos (False Positive Rate)**
- **Definición:** Proporción de casos benignos que son clasificados incorrectamente como malignos.
### **23. Tasa de Falsos Negativos (False Negative Rate)**
- **Definición:** Proporción de casos malignos que son clasificados incorrectamente como benignos.
### **24. Keras**
- **Definición:** Biblioteca de alto nivel en Python diseñada para construir y entrenar modelos de aprendizaje profundo de manera sencilla e intuitiva.
### **25. TensorFlow**
- **Definición:** Plataforma de código abierto para construir, entrenar y desplegar modelos de aprendizaje automático y redes neuronales profundas.

