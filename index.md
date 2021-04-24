## Portfolio

---

### Predicción tasa cumplimiento y clusters despachos

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](capstone_desafiolatam.html)

Proyecto final de la carrera de Data Science de la [Academia Desafío Latam](https://desafiolatam.com/data-science/).
El acceso de los datos fue a través de archivos planos que fueron ingestados en el bucket de Google Storage para luego transformarlos en tablas en BigQuery y su posterior preprocesamiento y modelamiento AI Unificado de Google Cloud Platform con el fin de aprovechar la persistencia de la herramienta.

**Princpales insights**

1. Hay 4 segmentos de despachos: *Los business package, los enfocados, los patiperros y los pocos optimizados*,

2. Existen 15 tipos de despachos más propensos a no cumplir con la entrega y que conforman el 51% de la cartera y se dividen entre los pocos optimizados y patiperros.

3. El mejor modelo predictivo fue el DecisionTreeClassifier con un recall del 72%.

4. Los principales features son: distancia_envio_mts, valor_contratado, horas_desde_creacion_hasta_compromiso, horas_desde_creacion_hasta_salida_primera_milla


<img src="principales hallazgos.png?raw=true"/> 



---

### Identificación usuarios molestos y potenciales cierre de negocios usando PySpark y Spark MLlib

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](spark_aws.html)

En este proyecto se busca determinar en base al dataset de YELP, si un usuario registrado está insatisfecho y por otro lado, la probabilidad de que un negocio cierre aplicando algoritmos de clasificación, regresión y extracción y selección de features de la librería Spark MLlib.


<img src="Captura de Pantalla 2021-04-23 a la(s) 20.36.34.png?raw=true"/>



---
### Análisis de Sentimiento Twitter

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](sentiment_analysis.html)

En este proyecto se busca evaluar los sentimientos de los tweets del dataset de CrowdFlower usando WordNetLemmatize, usando Pipeline e hiperparámetros.

<img src="Captura de Pantalla 2021-04-23 a la(s) 21.48.59.png"/>


---

### Determinantes del ingreso

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](determinantes_ingreso.html)

El objetivo es desarrollar un modelo predictivo sobre la probabilidad de que un individuo presente salarios por sobre o bajo los 50.000 dólares anuales, en base a una serie de atributos sociodemográficos.

<img src="Captura de Pantalla 2021-04-23 a la(s) 22.04.34.png"/>







---

<p style="font-size:11px">Page template forked from <a href="https://github.com/evanca/quick-portfolio">evanca</a></p>
<!-- Remove above link if you don't want to attibute -->
