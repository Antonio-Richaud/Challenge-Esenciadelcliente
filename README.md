# Challenge-Esencia-del-cliente 📊🤝

Para este segundo proyecto del bootcamp de Data Science, se desarrollo un análisis de datos bastante completo. Este proyecto se enfoca en comprender mejor a los clientes mediante el análisis de datos y técnicas avanzadas de clustering. A través de la implementación de diversas técnicas de machine learning y reducción de dimensionalidad, el objetivo es identificar patrones y segmentar a los clientes en diferentes grupos, permitiendo a las empresas personalizar sus estrategias de marketing y mejorar la experiencia del cliente.

Los datos utilizados para este proyecto fueron extraídos del conjunto de datos disponible en [Kaggle](https://www.kaggle.com/datasets/ramjasmaurya/medias-cost-prediction-in-foodmart), que proporciona información detallada sobre los costos y ventas de productos en Foodmart. Este conjunto de datos nos permitió realizar un análisis profundo y obtener insights valiosos sobre el comportamiento de los clientes.

El desarrollo y la ejecución de todo el proyecto se realizaron en Google Colab, una herramienta potente y accesible para el análisis de datos y el desarrollo de modelos de machine learning. Google Colab nos permitió aprovechar sus recursos computacionales y colaborar de manera eficiente durante todo el proceso.

Durante este proyecto, se utilizaron varias metodologías y herramientas de Python, incluyendo K-Means, PCA (Análisis de Componentes Principales) y métricas de evaluación como Silhouette, Davies-Bouldin y Calinski-Harabasz. Estas técnicas nos permitieron obtener insights valiosos sobre los datos de los clientes y agruparlos en clústeres bien definidos.

La importancia de este proyecto radica en su capacidad para transformar datos en información accionable. Al identificar y comprender los diferentes segmentos de clientes, las empresas pueden desarrollar estrategias de marketing más efectivas, mejorar la personalización de sus servicios y, en última instancia, aumentar la satisfacción y lealtad de sus clientes.

Este proyecto es una contribución con todo el amor del mundo para aquellos que buscan formarse en el fascinante ámbito de la Ciencia de Datos. Espero que mi trabajo pueda servir como una guía y recurso valioso para cualquier persona interesada en mejorar sus habilidades y conocimientos en esta área.


[@Antonio Richaud](https://www.antonio-richaud.com/)


![Logo](https://www.aluracursos.com/assets/img/challenges/logos/challenges-logo-data.1712144089.svg)

![Insignia](#)


## Pasos que se siguieron para el desarrollo del Challenge 

### 1. Configuración del Ambiente

Para desarrollar este proyecto, trabajamos en Google Colab. Primero, creamos una cuenta en Gmail si no se tiene. Luego, accedemos a Google Colab y creamos un nuevo Notebook, nombrándolo como deseemos (ej. "La esencia del cliente 1"). Conectamos el notebook a Google Drive.

Descargamos el dataset desde las URLs proporcionadas, creamos un directorio en Google Drive y subimos el dataset allí. Con esto, estamos listos para avanzar a la siguiente etapa.

### 2. Obtención de los datos 
* Cargamos los archivos almacenados en Google Drive utilizando la biblioteca pandas.

* Traducimos el dataset del inglés al español para una mejor comprensión. Utilizamos diccionarios de traducción proporcionados en un archivo de Python.

* Exportamos el dataset traducido en formato .csv y lo almacenamos en nuestro directorio de Google Drive para su uso en la siguiente parte del desafío.


### 3. Exploración de los datos 

La exploración visual de datos permite identificar características importantes como valores atípicos, distribuciones, correlaciones y agrupaciones, que pueden no ser evidentes al examinar solo los números.

* Utilizamos Matplotlib y Seaborn para generar diversos gráficos que nos ayuden a entender mejor los datos.

* Por ejemplo, un histograma puede mostrar la distribución de ingresos anuales de los clientes. Seleccionamos y visualizamos diferentes variables según lo consideremos pertinente.

* Registramos nuestras observaciones e hipótesis en una celda de texto del notebook a medida que generamos los gráficos.

Tip: Algunas variables de interés para el análisis visual incluyen Escolaridad, Ocupación, Miembro, Género, Estado Civil, Número de Hijos, Ingresos anuales, Categoría de alimentos y Tipo.

### 4. Preprocesamiento y Obtención de Features

Codificamos las variables categóricas para que el modelo de clusterización las reconozca, utilizando métodos como one-hot-encoder, get_dummies o asignación de valores numéricos basados en jerarquías (ej. primaria = 1, secundaria = 2, universidad = 3).

Reemplazamos las cadenas de texto en el dataset con los valores numéricos asignados.

Tip: No es necesario codificar todas las columnas categóricas, solo aquellas relevantes para la clusterización.

Seleccionamos las variables más relevantes para el análisis, con el objetivo de agrupar a los clientes en diversos clústeres para entender sus características y brindarles mejor servicio.

Con al menos 6 y máximo 12 atributos seleccionados, estandarizamos los datos (que ahora son todos numéricos) utilizando StandardScaler, para asegurar que todas las variables estén en la misma escala y el modelo aprenda correctamente de todos los atributos. Almacenamos los valores estandarizados en una variable llamada X_std.

Al finalizar, obtenemos un numpy array listo para avanzar a la próxima fase.

### 5. Clusterización y validación

**Clusterización**

Utilizamos el algoritmo KMeans para la clusterización, aunque se pueden usar otros como Mean Shift o DBSCAN. El objetivo es encontrar el mejor número de clústeres.

**Validación:**

Número de clústeres: Instanciamos de 3 a 10 clústeres utilizando X_std. Evaluamos con métricas como Silhouette (mínimo 0.50), Davies-Bouldin (máximo 0.75) y Calinski-Harabasz (lo más alto posible) para determinar la mejor configuración.

* **Estructura:** Generamos una baseline con números aleatorios (random_data) y repetimos el paso 2 para compararlo con X_std. Nos aseguramos de que X_std tenga un desempeño significativamente superior al de random_data.

* **Estabilidad:** Evaluamos la estabilidad segmentando X_std en 3 o 5 partes iguales utilizando array_split() de numpy. Repetimos los pasos de validación para cada segmento y aseguramos que las variaciones en las métricas no sean mayores a ±5% entre los sets, garantizando homogeneidad en los clústeres.

Si no se logran los resultados sugeridos, revisamos y ajustamos las variables, y repetimos los pasos anteriores.

**Instanciación de la mejor configuración de clústeres:**

Instanciamos el algoritmo de clusterización con la mejor configuración encontrada y creamos un nuevo atributo en el dataset datos_raw llamado 'cluster' para almacenar las etiquetas de los clústeres. No ejecutamos KMeans nuevamente para mantener la consistencia de los clústeres.

Realizamos gráficos de dispersión comparando variables y añadiendo una tercera dimensión con los clústeres en el parámetro hue. Describimos las observaciones, por ejemplo: "En el clúster 0, de color rojo, se agrupan los clientes que gastan más dinero en productos no comestibles". Repetimos hasta obtener descripciones detalladas de cada clúster.

### 6. Descripción de los clusters

Generamos una celda de texto con el resultado consolidado del análisis de los clústeres. Este análisis incluye las características principales de cada clúster basadas en las observaciones realizadas en los gráficos de dispersión.

## Información que me ayudo

 - [K-Means](https://antonio-richaud.com/blog/archivo/publicaciones/12-k-means.html)
 - [PCA](https://antonio-richaud.com/blog/archivo/publicaciones/29-pca.html)



---

**Conecta conmigo en alguna de mis redes sociales 🤓**

Si deseas seguir mi progreso o conectar conmigo, puedes hacerlo a través de mis redes sociales o visitar mi portafolio web:

[![LinkedIn](https://img.shields.io/badge/-LINKEDIN-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/antonio-richaud/)
[![X](https://img.shields.io/badge/-(Twitter)-000000?style=for-the-badge&logo=X&logoColor=white)](https://twitter.com/Antonio_Richaud)
[![Youtube](https://img.shields.io/badge/-YOUTUBE-D14836?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@AntonioRichaud/)
[![TIKTOK](https://img.shields.io/badge/-TIKTOK-000000?style=for-the-badge&logo=tiktok&logoColor=white)](https://www.tiktok.com/@antonio_richaud)
[![Antonio-Richaud.com](https://img.shields.io/badge/-ANTONIORICHAUD.COM-8E2DE2?style=for-the-badge&logo=react&logoColor=white)](https://antonio-richaud.com/)

---