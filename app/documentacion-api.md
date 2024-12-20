
# Documentación General de la API

## Tabla de Contenidos
1. [Introducción](#introducción)
2. [Requisitos Previos](#requisitos-previos)
3. [Instalación](#instalación)
   - [Clonación del Repositorio](#clonación-del-repositorio)
   - [Configuración del Entorno](#configuración-del-entorno)
4. [Uso de Docker](#uso-de-docker)
   - [Construcción de la Imagen](#construcción-de-la-imagen)
   - [Ejecución del Contenedor](#ejecución-del-contenedor)
   - [Detener y Eliminar Contenedores](#detener-y-eliminar-contenedores)
5. [Endpoints de la API](#endpoints-de-la-api)
   - [POST /clasification_image](#post-clasification_image)
6. [Manejo de Errores](#manejo-de-errores)

---

## Introducción
Esta API permite clasificar imágenes para detectar tumores benignos o tumores malignos. 
Es ideal para integrarse en aplicaciones que requieran un sistema de diagnóstico automatizado.

### Fecha de Actualización
Última actualización: 3-12-2024

---

## Requisitos Previos
- Docker y Docker Compose instalados.
- Git instalado.
- Python 3.10 o superior (para desarrollo).
- Configuración básica del sistema operativo.

## Instalación

### Clonación del Repositorio
Clona el repositorio del proyecto:
```bash
git clone https://github.com/clasicancer/Clasificancer.git
cd proyecto-cancer
```
## Carga del modelo

El archivo de modelo necesario para ejecutar la API es bastante pesado y no está incluido directamente en el repositorio. Puedes descargarlo desde el siguiente enlace:
 
[Descargar cancer_modelo_2.h5](https://drive.google.com/file/d/17XrqdCtbny6RukYmdqZqthdpPddUPYFi/view?usp=share_link)

[Descargar cancer_modelo_1.h5](https://drive.google.com/file/d/1soqhuJiuDvAopDfQRg8jR2RjYfnlnK0b/view?usp=share_link)

[Descargar cancer_modelo_trans_1.h5](https://drive.google.com/file/d/1RyIMKE782dmjdUphPpUdTMRAW4SVLkeC/view?usp=share_link)

Por favor, guarda el archivo en la ubicación indicada en la configuración del modelo (/app/cancer_modelo_2.h5 si usas Docker).
 
## Uso de Docker

### Construcción de la Imagen
Construye la imagen de Docker:
```bash
docker build -t cancer-api .
```

### Ejecución del Contenedor
Inicia el contenedor con Docker:
```bash
docker run -d -p 80:80 cancer-api
```
Accede a la API en `http://localhost:80/docs` para la documentación interactiva.

### Detener y Eliminar Contenedores
Para detener un contenedor:
```bash
docker stop id_contenedor
```
Para eliminarlo:
```bash
docker rm id_contenedor
```

---

## Endpoints de la API

### POST /clasification_image
**Descripción:** Clasifica una imagen en tumor maligno o tumor benigno.

**Request Body:**
- `img_base64` (string): Imagen codificada en Base64.

**Ejemplo:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"img_base64": "base64_string"}' http://localhost:80/clasification_image
```

**Response:**
```json
{
  "exactitud": 0.98,
  "predicted_class": "Benign"
}
```

---

## Manejo de Errores
La API devuelve respuestas HTTP con códigos de error estándar:
- **400**: Error en la solicitud (p. ej., imagen inválida o no soportada).
- **404**: Endpoint no encontrado.
- **500**: Error interno del servidor.

## Observaciones

- Los modelos deben estar entrenados y disponibles en el directorio `/app`.
