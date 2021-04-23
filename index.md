## Portfolio

---

### Category Name 1 

[Project 1 Title](/sample_page)
<img src="images/dummy_thumbnail.jpg?raw=true"/>


**Datos**<br>


Se entregan datos del mes de agosto, el cual corresponde a un mes tipo con demanda normal (sin eventos especiales de aumentos de demanda). La data entregada corresponde a registros con las  características de los pedidos ( dimesiones, categorías, tipos de clientes, etc.) y también sobre la trazabilidad del pedido durante el proceso y sus movimientos asociados.<br>


**Vector Objetivo**<br>


El vector objetivo corresponde a una variable binaria que determina si el pedido llego o no en la fecha comprometida, es 1 cuando la fecha de entrega real es menor o igual a la fecha de entrega comprometida, en otro caso no se cumple la entrega y es el resultado es 0.<br>


**Solución**<br>


La solución propuesta abarca en utilizar herramientas de google cloud para lograr dos objetivos:


1. Identificar los segmentos de servicios de entrega con peor tasa de cumplimiento


2. Obtener un modelo predictivo que permita determinar la probabilidad del resultado de una entrega lo antes posible.

#### **Módulos adicionales**

Se trabajan con diferentes módulos, en caso de falta alguno se pueden instalar con:


```python
#pandas
!pip install pandas
#numpy
!pip install numpy
#matplotlib
!pip install matplotlib
#seaborn
!pip install seaborn
#scipy
!pip install scipy
#networkx
!pip install networkx
#plotly
!pip install plotly
#yellowbrick
!pip install yellowbrick
```

#### **Importación de módulos y funciones propias**

Se importan librerias y funciones propias:


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.stats import mode
from scipy import stats
from scipy.stats import ttest_ind
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (15,10)
plt.style.use('seaborn-darkgrid')
import plotly.offline as py
import plotly.graph_objects as go
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score

from sklearn.metrics import classification_report,accuracy_score,roc_curve,auc,confusion_matrix,plot_confusion_matrix
from sklearn.metrics import roc_auc_score,precision_score,recall_score, make_scorer,f1_score

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

import pickle
from joblib import load, dump

from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
import plotly.figure_factory as ff
pd.set_option('display.float_format', '{:.2f}'.format) ### para sacarle el formato exponencial al dataframe
import plotly.graph_objects as go

### funciones propias
from aux import *
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-latest.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-latest.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>



#### **Ingesta de datos**

Se entregan dos csv con información sobre los pedidos:

 * `datos_med.csv`: presenta detalle de diferentes atributos de cada pedido
 
 * `datos_fact.csv`: presenta todos los movimientos que se hicieron en la vida del pedido.
 
Ambos csv se extraen desde la base de datos del cliente, para trabajarlos se subieron al ambiente de Google Cloud-Storage y por medio de BigQuery se genero una tabla que contenga los datos entregados con sus atributos.<br>



```python
%%bigquery df
select * from `charged-ground-301216.test_1_fast_project.datos_med`
```

Se revisa tamaño del dataframe y cantidad de atirbutos:


```python
df.shape
```




    (2568900, 34)



Son datos de 2.568.900 registros, correspondiente a los pedidos del mes de Agosto-20.

#### **Pre-procesamiento y limpieza inicial**

Se revisa presencia de datos nulos o perdidos con función auxiliar:


```python
missing_values_table(df)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing Values</th>
      <th>% Missing Values</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



No se encontraron registros con datos perdidos o nulos, se puede continuar con el procesamiento.

Se genera un nuevo atributo en la base que se calcula a partir las dimensiones de peso, volumen, alto y largo. Se llama peso equivalente, donde se evalua entre el peso físico versus el peso volumétrico, quedando el mayor valor como el seleccionado. El peso volumétrico se refiere  una convencion de que en un metro cúbico equivalen a 250 kgs ( largo x ancho x alto / 4000). <br>
En base a esto se considera además, la propia clasificación tarifaria de FAST que diferencia al peso equivalente en 4 categorías:
* Pequeño: desde 0 a 1.5 kilos.
* Mediano: mayor a 1.5 y hasta 3 kilos.
* Grande: mayor a 3 y hasta 4.5 kilos.
* Sobre-dimensionado: Mayor a 4.5 kilos.

Detalle de la transformación:


```python
%%bigquery df

select   CASE 
                WHEN ( ( (largo * ancho * alto) / 4000 ) > peso ) THEN CASE 
                                                                             WHEN ( ( (largo * ancho * alto) / 4000 ) <= 1.5 ) THEN 'peq' 
                                                                             WHEN ( ( (largo * ancho * alto) / 4000 ) > 1.5 and ( (largo * ancho * alto) / 4000 ) <= 3 ) THEN 'med'
                                                                             WHEN ( ( (largo * ancho * alto) / 4000 ) > 3 and ( (largo * ancho * alto) / 4000 ) <= 4.5 ) THEN 'gran'
                                                                             ELSE 'sob'
                                                                        END
                ELSE CASE 
                             WHEN ( peso <= 1.5 ) THEN 'peq'
                             WHEN ( peso > 1.5 and peso <= 3 ) THEN 'med'
                             WHEN ( peso > 3 and peso <= 4 ) THEN 'gran'
                             ELSE 'sob'
                      END
         END as dimension_pedido
        ,lower(replace(tipo_cliente,' ','_')) as tipo_cliente
        ,lower(replace(tipo_medicamento,' ','_')) as tipo_medicamento
        ,comuna_origen
        ,comuna_destino
        ,region_origen
        ,region_destino
        ,((tiene_nombre_destinatario * 2) + (tiene_correo_destinatario * 2) + (tiene_telefono_destinatario * 6)) as score_datos_contactabilidad_destintario
        ,((tiene_nombre_remitente * 2) + (tiene_correo_remitente * 2) + (tiene_telefono_remitente * 6))          as score_datos_contactabilidad_remitente
        ,lower(replace(velocidad_servicio,' ','_')) as velocidad_servicio
        ,lower(replace(tipo_envio,' ','_')) as tipo_envio
        ,lower(replace(tipo_induccion,' ','_')) as tipo_induccion
        ,case when sla_compromiso=1 then 'si' else 'no' end cumplimiento
        ,valor_contratado
        ,distancia_envio_mts
        ,satisfaccion_cliente
        ,dia_semana_admision
        ,dia_semana_entrega
        ,lower(replace(tipo_servicio,' ','_')) as tipo_servicio
        ,case when (admite_a_tiempo_para_p__ck_up = 1) THEN 'si' else 'no' end as admite_a_tiempo_para_pick_up
        ,horas_desde_creacion_hasta_compromiso
        ,case when horas_desde_creacion_hasta_compromiso<=48 then 'menor_48' else 'mayor_48' end tramo_hr_desp
        ,horas_desde_creacion_hasta_salida_primera_milla
from `charged-ground-301216.test_1_fast_project.datos_med`                               
```

#### **Sobre los datos**

Descripción de cada variable en el dataframe:


* `dimension_pedido`: (str) variable categórica que clasifica si el pedido es pequeño (peq), mediano (med), grande (gran) y sobredimensionado (sob).


* `tipo_cliente`:(str) variable categórica que define si el cliente es una persona natural (persona) o una empresa (empresa).


* `tipo_medicamento`:(str) variable categórica que define si el medicamento enviado es de alto valor o normal.


* `comuna_origen`: (str) variable que representa el código de la comuna desde donde se envia el medicamento.


* `comuna_destino`:(str) variable que representa el código de la comuna hacia donde se envia el medicamento.


* `region_origen`:(str) variable que representa el código de la región desde donde se envia el medicamento.


* `region_destino`:(str) variable que representa el código de la región hacia donde se envia el medicamento.


* `score_datos_contactabilidad_destintario`: (int) valor de escala de contactabilidad del destinatario, es un puntaje entregado por datos de datos personales, mail y teléfono de contactabilidad.


* `score_datos_contactabilidad_remitente`:(int) valor de escala de contactabilidad del remitente, es un puntaje entregado por datos de datos personales, mail y teléfono de contactabilidad.


* `velocidad_servicio`: (str) variable categórica que define la velocidad del servicio contratado va desde mismo dia hasta plus_1.


* `tipo_envio`:(str) variable categórica que define si el pedido fue enviado a la puerta o por ventanilla.


* `tipo_induccion`:(str) variable categórica que define si el drop es e sucursal o retirado al cliente.


* `cumplimiento`: (str) vector objetivo que explica si el pedido llego o no en la promesa de entrega realizada al inicio del proceso.


* `valor_contratado`: (float) valor asociado a la entrega del pedido.


* `distancia_envio_mts`:(float) valor asociado a la distancia para entregar el pedido.


* `tipo_servicio`:(str) variable categorica que define si el pedido es una entrega local o interregional.


* `admite_a_tiempo_para_pick_up`: (str) variable categórica que clasifica si el pedido fue admitido o no a tiempo para pickup.


* `horas_desde_creacion_hasta_compromiso`: (float) valor que refleja el número de horas totales que se tiene para entregar el pedido hasta el compromiso.


* `tramo_hr_desp`: (str) variable que define si el despacho se entregó en un tiempo menor o igual 48 horas, o bien superior a este.


* `dia_semana_admision`: (str) variable categórica que toma marcas lunes a domingo y define el día en que se crea el despacho en el sistema de FAST.


* `dia_semana_entrega`: (str) variable categórica que toma marcas lunes a domingo y define el día en que se entrega el despacho.


* `horas_desde_creacion_hasta_salida_primera_milla`:(float) valor que refleja el número de horas transcurridas desde la creación hasta la salida en primera milla.

Sobre las principales variables descriptivas:


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score_datos_contactabilidad_destintario</th>
      <th>score_datos_contactabilidad_remitente</th>
      <th>valor_contratado</th>
      <th>distancia_envio_mts</th>
      <th>satisfaccion_cliente</th>
      <th>horas_desde_creacion_hasta_compromiso</th>
      <th>horas_desde_creacion_hasta_salida_primera_milla</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2568900.00</td>
      <td>2568900.00</td>
      <td>2568900.00</td>
      <td>2568900.00</td>
      <td>2568900.00</td>
      <td>2568900.00</td>
      <td>2568900.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.72</td>
      <td>7.03</td>
      <td>11325.14</td>
      <td>323790.05</td>
      <td>3.60</td>
      <td>48.66</td>
      <td>22.33</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.03</td>
      <td>2.53</td>
      <td>17353.13</td>
      <td>440732.66</td>
      <td>0.50</td>
      <td>24.68</td>
      <td>35.36</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.73</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.00</td>
      <td>8.00</td>
      <td>6753.38</td>
      <td>16024.87</td>
      <td>3.00</td>
      <td>31.00</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.00</td>
      <td>8.00</td>
      <td>9235.65</td>
      <td>90529.70</td>
      <td>4.00</td>
      <td>37.00</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.00</td>
      <td>8.00</td>
      <td>11624.77</td>
      <td>464024.55</td>
      <td>4.00</td>
      <td>61.00</td>
      <td>28.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10.00</td>
      <td>10.00</td>
      <td>6624105.22</td>
      <td>3874988.21</td>
      <td>4.00</td>
      <td>158.00</td>
      <td>2863.00</td>
    </tr>
  </tbody>
</table>
</div>



#### **Segmentacion de conjuntos de validacion y modelamiento**

Sobre los conjuntos de la data a revisar, se trabajara con el 100% del conjunto de datos entregado para la modelación y para el entranmiento de modelos se dividirá en dos grupos de Test/Training con un % de 70 y 30.<br>
Para la validación del modelo se contrastará los resultados de los modelos con un nuevo dataset correspondiente a otro mes o periodo.

### Analisis descriptivo y pre seleccion de atributos

#### **Vector Objetivo**

El vector objetivo corresponde a una variable binaria que determina si el pedido llego o no en la fecha comprometida, es 1 cuando la fecha de entrega real es menor o igual a la fecha de entrega comprometida, en otro caso no se cumple la entrega y es el resultado es 0. Sobre su comportamiento:


```python
eda_plots(pd.DataFrame(df['cumplimiento']),1)
```


![png](output_31_0.png)


* Se nota una distribución bastante pareja para nuestro vector objetivo, donde el cerca del 60% de los pedidos cumplen y el 40% no, de modo que existen varias oportunidades de mejora.

#### **Atributos**

Revisamos la distribucion de los atributos:


```python
atributos=list(df.columns)
atributos.remove('cumplimiento')
eda_plots(df.loc[:,atributos],4)
```


![png](output_35_0.png)


* Revisando los datos podemos ver que las variables continuas presentan un sesgo positivo, con un gráficos con tendencia a la izquierda, esto se podria corregir con algún tipo de normalización que evaluaremos más adelante.
* Sobre las variables categóricas, en general se ve un desbalanceo de clases, por ejemplo la gran mayoría de los pedidos son del tipo pequeño con clientes empresa y de alto valor. <br>

A continuación se revisa más a fondo el desbalanceo de estos atributos, al revisarlos con el vector objetivo.

#### **Dist. atributos con respecto al vector objetivo**

Se revisa cada atributo categórico y cómo es el cumplimiento del compromiso para cada segmento, <br>

**Cumplimiento según tamaño**


```python
g = sns.catplot(x="cumplimiento", col="dimension_pedido", col_wrap=4,data=df,kind="count", height=2.5, aspect=1.8)
```


![png](output_39_0.png)


* Dado la especialidad de nuestra empresa, esta contenida dentro de la industria courrier, en la que las grandes dimensiones son multadas con tarifas mayores.
* Dado el desbalance a las dimensiones pequeñas se obserba una pequeña alza de 3 puntos porcentuales, con respecto al cumplimiento global 60%, asociado a la facilidad del transporte y la priorización que naturalmente se realiza en la operación hacia los menores volúmenes. (Transportar más en el mismo espacio) <br>

**Cumplimiento según tipo cliente**


```python
g = sns.catplot(x="cumplimiento", col="tipo_cliente", col_wrap=4,data=df,kind="count", height=2.5, aspect=1.8)
```


![png](output_41_0.png)


* Se observa un desbalance marcado hacia los clientes empresa que aportan un mayor volumen de medicamentos que los clientes esporádicos (Contado).
* Se obserba que la relación entre tipo de empresa y cumplimiento es levemente la misma 60.38% empresa y 61.44% para contado, esto puede tratarse de tipo de recolección que se realiza en la primera milla, más sujeta de las condiciones operacionales de cada cliente en el crédito y más normalizadas en los retiros a oficinas comerciales.<br>

**Cumplimiento según tipo medicamento**


```python
g = sns.catplot(x="cumplimiento", col="tipo_medicamento", col_wrap=4,data=df,kind="count", height=2.5, aspect=1.8)
```


![png](output_43_0.png)


* Se observa un desbalance marcado hacia los clientes con bajo valor.
* Se obserba que la relación entre tipo de medicamento y el cumplimiento es mnarcadamente mayor para los de mayor valor 60.15% y 74.41% para los de mayor valor. Esto se da ya que los medicamentos de mayor valor tienen un proceso de recolección, procesamiento o clasificación diferenciado en la primera y ultima milla, es decir, tratamiento dedicado por un area especializada de valorados. <br>

**Cumplimiento según velocidad servicio**


```python
g = sns.catplot(x="cumplimiento", col="velocidad_servicio", col_wrap=4,data=df,kind="count", height=2.5, aspect=1.8)
```


![png](output_45_0.png)


* Se observa un desbalance marcado hacia la velocidad de servicio standar y que muestra un comportamiento equivalente al global 60%
* La velocidad de entrega SAME DAY (2.4%), muestra un comportamiento extremadamente bajo 44%, dado que los plazos pueden ser en algunos casos horas.<br>

**Cumplimiento según tipo envio**


```python
g = sns.catplot(x="cumplimiento", col="tipo_envio", col_wrap=4,data=df,kind="count", height=2.5, aspect=1.8)
```


![png](output_47_0.png)


* A pesar del que la muestra esta sesgada hacia la entrega en domicilio v/s retiro en sucursal, ambos muestran un comprotamiento similar en terminos de cumplimiento 60.38% y 60.34%, dado que ambas distribuciones se realizan con la misma flota, y bajo el mismo sistema de optimización. (Excepto Medicamentos Valorados)

**Cumplimiento según tipo de inducción**


```python
g = sns.catplot(x="cumplimiento", col="tipo_induccion", col_wrap=4,data=df,kind="count", height=2.5, aspect=1.8)
```


![png](output_49_0.png)


* A pesar del que la muestra esta sesgada hacia el PickUP al cliente, que refuerza el desbalance hacia las ventas a clientes empresa, el retiro al cliente empresa muestra un 60.43% v/s un 60.81% de drop en sucursales, en parte a la misma justicación para el "Análisis de cumplimiento para el tipo de envío (Ultima Milla)", dado     que ambas distribuciones se realizan con la misma flota, y bajo el mismo sistema de optimización. (Excepto Medicamentos Valorados)<br>

**Cumplimiento según tipo de servicio**


```python
g = sns.catplot(x="cumplimiento", col="tipo_servicio", col_wrap=4,data=df,kind="count", height=2.5, aspect=1.8)
```


![png](output_51_0.png)


* A pesar del que la muestra no muestra un desbalance, los indicadores de cumplimiento son bastante deferentes, 52.25% para el servicio Regional que representa 59% del global y un cumplimiento del 72.43% para los servicios locales, entendiendo que el transporte Regional, con rutas de mayor kilometrajes presentan una mayor riesgo al cumplimiento.
* Es interesante que dos segmentos que son relativamente similares tienen comportamientos tan distintos en terminos de cumplimiento, por lo que un aumento en el rendimiento de las rutas troncales podria afectar en un eventual doble digito de aumento al cumplimiento.<br>

**Cumplimiento según tiempo de pickup**


```python
g = sns.catplot(x="cumplimiento", col="admite_a_tiempo_para_pick_up", col_wrap=4,data=df,kind="count", height=2.5, aspect=1.8)
```


![png](output_53_0.png)


* Con la intención de mostrar un indicador relacionado a al tiempo, sin afectar la oportunidad de la explotación del modelo se generó este indicador que representa si el medicamento fue admitido antes que el conductor pasara a la sucursal se haya realizado, a pesar de que su incumplimiento implica un incumplimiento automático, esto es mitigado con controles en los proceso de clasificación que reconocen estos casos y son derivados a la flota de valorados lo que explica su % de Incumplimiento 60.25% admite a tiempo para pickup, contra un 70.29% cuando es admitido fuera del plazo del pickup.<br>


**Cumplimiento según día de creación del despacho**


```python
g = sns.catplot(x="cumplimiento", col="dia_semana_admision", col_wrap=4,data=df,kind="count", height=2.5, aspect=1.8)
```


![png](output_55_0.png)


* Se observa que en términos absolutos y relativos que los despachos creados los lunes son los que menor tasa de entrega, seguido del día martes.

**Cumplimiento según día de entrega del despacho**


```python
g = sns.catplot(x="cumplimiento", col="dia_semana_entrega", col_wrap=4,data=df,kind="count", height=2.5, aspect=1.8)
```


![png](output_58_0.png)


* Los días de entrega miércoles y jueves son aquellos que tienen una mayor tasa de entrega fallida.


**Cumplimiento según tiempo de entrega**


```python
g = sns.catplot(x="cumplimiento", col="tramo_hr_desp", col_wrap=4,data=df,kind="count", height=2.5, aspect=1.8)
```


![png](output_60_0.png)


### Correlaciones

Sobre las variables continuas, se revisa una matriz de correlaciones:


```python
corr_matrix = df[['valor_contratado','distancia_envio_mts','horas_desde_creacion_hasta_compromiso','horas_desde_creacion_hasta_salida_primera_milla','satisfaccion_cliente']]
corrs = corr_matrix.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(4).values,
    showscale=True)
figure
```


<div>                            <div id="b767342c-8757-4b02-a3b1-97b0fdc32fc0" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("b767342c-8757-4b02-a3b1-97b0fdc32fc0")) {                    Plotly.newPlot(                        "b767342c-8757-4b02-a3b1-97b0fdc32fc0",                        [{"colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "reversescale": false, "showscale": true, "type": "heatmap", "x": ["valor_contratado", "distancia_envio_mts", "horas_desde_creacion_hasta_compromiso", "horas_desde_creacion_hasta_salida_primera_milla", "satisfaccion_cliente"], "y": ["valor_contratado", "distancia_envio_mts", "horas_desde_creacion_hasta_compromiso", "horas_desde_creacion_hasta_salida_primera_milla", "satisfaccion_cliente"], "z": [[1.0, 0.2851078195691064, 0.029864137652588715, -0.004599448464515889, -0.1025805720854128], [0.2851078195691064, 1.0, 0.10863574496658454, -0.030599915446633326, -0.27215890827578676], [0.029864137652588715, 0.10863574496658454, 1.0, 0.1349072710021357, 0.20432458005004497], [-0.004599448464515889, -0.030599915446633326, 0.1349072710021357, 1.0, -0.1265499613251182], [-0.1025805720854128, -0.27215890827578676, 0.20432458005004497, -0.1265499613251182, 1.0]]}],                        {"annotations": [{"font": {"color": "#000000"}, "showarrow": false, "text": "1.0", "x": "valor_contratado", "xref": "x", "y": "valor_contratado", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "0.2851", "x": "distancia_envio_mts", "xref": "x", "y": "valor_contratado", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "0.0299", "x": "horas_desde_creacion_hasta_compromiso", "xref": "x", "y": "valor_contratado", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "-0.0046", "x": "horas_desde_creacion_hasta_salida_primera_milla", "xref": "x", "y": "valor_contratado", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "-0.1026", "x": "satisfaccion_cliente", "xref": "x", "y": "valor_contratado", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "0.2851", "x": "valor_contratado", "xref": "x", "y": "distancia_envio_mts", "yref": "y"}, {"font": {"color": "#000000"}, "showarrow": false, "text": "1.0", "x": "distancia_envio_mts", "xref": "x", "y": "distancia_envio_mts", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "0.1086", "x": "horas_desde_creacion_hasta_compromiso", "xref": "x", "y": "distancia_envio_mts", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "-0.0306", "x": "horas_desde_creacion_hasta_salida_primera_milla", "xref": "x", "y": "distancia_envio_mts", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "-0.2722", "x": "satisfaccion_cliente", "xref": "x", "y": "distancia_envio_mts", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "0.0299", "x": "valor_contratado", "xref": "x", "y": "horas_desde_creacion_hasta_compromiso", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "0.1086", "x": "distancia_envio_mts", "xref": "x", "y": "horas_desde_creacion_hasta_compromiso", "yref": "y"}, {"font": {"color": "#000000"}, "showarrow": false, "text": "1.0", "x": "horas_desde_creacion_hasta_compromiso", "xref": "x", "y": "horas_desde_creacion_hasta_compromiso", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "0.1349", "x": "horas_desde_creacion_hasta_salida_primera_milla", "xref": "x", "y": "horas_desde_creacion_hasta_compromiso", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "0.2043", "x": "satisfaccion_cliente", "xref": "x", "y": "horas_desde_creacion_hasta_compromiso", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "-0.0046", "x": "valor_contratado", "xref": "x", "y": "horas_desde_creacion_hasta_salida_primera_milla", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "-0.0306", "x": "distancia_envio_mts", "xref": "x", "y": "horas_desde_creacion_hasta_salida_primera_milla", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "0.1349", "x": "horas_desde_creacion_hasta_compromiso", "xref": "x", "y": "horas_desde_creacion_hasta_salida_primera_milla", "yref": "y"}, {"font": {"color": "#000000"}, "showarrow": false, "text": "1.0", "x": "horas_desde_creacion_hasta_salida_primera_milla", "xref": "x", "y": "horas_desde_creacion_hasta_salida_primera_milla", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "-0.1265", "x": "satisfaccion_cliente", "xref": "x", "y": "horas_desde_creacion_hasta_salida_primera_milla", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "-0.1026", "x": "valor_contratado", "xref": "x", "y": "satisfaccion_cliente", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "-0.2722", "x": "distancia_envio_mts", "xref": "x", "y": "satisfaccion_cliente", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "0.2043", "x": "horas_desde_creacion_hasta_compromiso", "xref": "x", "y": "satisfaccion_cliente", "yref": "y"}, {"font": {"color": "#FFFFFF"}, "showarrow": false, "text": "-0.1265", "x": "horas_desde_creacion_hasta_salida_primera_milla", "xref": "x", "y": "satisfaccion_cliente", "yref": "y"}, {"font": {"color": "#000000"}, "showarrow": false, "text": "1.0", "x": "satisfaccion_cliente", "xref": "x", "y": "satisfaccion_cliente", "yref": "y"}], "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "autotypenumbers": "strict", "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"dtick": 1, "gridcolor": "rgb(0, 0, 0)", "side": "top", "ticks": ""}, "yaxis": {"dtick": 1, "ticks": "", "ticksuffix": "  "}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('b767342c-8757-4b02-a3b1-97b0fdc32fc0');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


* Los atributos **distancia_envio_mts** y **valor_contratado**, tienen una correlacion (+) de (0.2851). Este valor se justifica dado que el tarifario de FAST se ajusta a el alza en la medida que los valores y los tiempos de servicio aumenten.

* Los atributos **horas_desde_creacion_hasta_compromiso** y **valor_contratado", tienen una correlacion (+) de (0.0299). Este valor se justifica dado que el tarifario de FAST se ajusta a el alza en la medida que los valores y los tiempos de servicio aumenten.

* Los atributos **distancia_envio_mts** y **horas_desde_creacion_hasta_salida_primera_milla**, tienen una correlación (-) de (-0.0297). Este valor se justifica por la composición de la operación que prioriza el transporte regional o de mayor distancia y las procesa en el momento que llegan a los centros operacinales para después dedicarse a los pedidos Locales o de menor distancia, por lo que mayor distancia, menor será la demora en el cierre del ciclo de la primera milla.

Sobre las variables categorias, se arman dummies y luego se revisa matriz:


```python
df_3 = df.loc[:,['dimension_pedido','tipo_cliente','tipo_medicamento','velocidad_servicio','tipo_envio','tipo_induccion','tipo_servicio','admite_a_tiempo_para_pick_up']]
df_dum = pd.get_dummies(df_3, columns=['dimension_pedido','tipo_cliente','tipo_medicamento','velocidad_servicio','tipo_envio','tipo_induccion','tipo_servicio','admite_a_tiempo_para_pick_up'], drop_first=True)
plt.figure(figsize=(40,30))

corrs = df_dum.corr()

figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(4).values,
    showscale=True)

for i in range(len(figure.layout.annotations)):
    figure.layout.annotations[i].font.size = 8

figure.show()
```


<div>                            <div id="0b1bbe28-efd2-4b4f-a37d-6b166bc3f711" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("0b1bbe28-efd2-4b4f-a37d-6b166bc3f711")) {                    Plotly.newPlot(                        "0b1bbe28-efd2-4b4f-a37d-6b166bc3f711",                        [{"colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "reversescale": false, "showscale": true, "type": "heatmap", "x": ["dimension_pedido_med", "dimension_pedido_peq", "dimension_pedido_sob", "tipo_cliente_persona", "tipo_medicamento_normal", "velocidad_servicio_plus_2", "velocidad_servicio_plus_3", "velocidad_servicio_same_day", "tipo_envio_retiro_ventanilla", "tipo_induccion_retirado_a_cliente", "tipo_servicio_regional", "admite_a_tiempo_para_pick_up_si"], "y": ["dimension_pedido_med", "dimension_pedido_peq", "dimension_pedido_sob", "tipo_cliente_persona", "tipo_medicamento_normal", "velocidad_servicio_plus_2", "velocidad_servicio_plus_3", "velocidad_servicio_same_day", "tipo_envio_retiro_ventanilla", "tipo_induccion_retirado_a_cliente", "tipo_servicio_regional", "admite_a_tiempo_para_pick_up_si"], "z": [[1.0, -0.552907214556941, -0.21186306716336234, -0.01770652552543607, -0.04871133711512562, 0.005502872861730658, 0.005166182215155891, -0.02495986419623275, -0.0013196238696726917, 0.019051973345775068, 0.018398281802604166, 0.008206888372147265], [-0.552907214556941, 1.0, -0.5340912721327576, -0.006548897449039313, 0.09214689762898691, -0.014258218851366942, -0.012826315954965237, 0.036701935556977684, -0.01771946125163068, 0.004540420578338916, -0.027175514715190606, 0.0036026694890132515], [-0.21186306716336234, -0.5340912721327576, 1.0, 0.016187467077725025, -0.05184237784909868, 0.008470584377005635, 0.006865005776331517, -0.011012681789167195, 0.016809715425162304, -0.015355782095426042, 0.0118575822568065, -0.0061481876651798], [-0.01770652552543607, -0.006548897449039313, 0.016187467077725025, 1.0, -0.024968197281294405, 0.08723294501206104, 0.05465195626019519, 0.101331349900505, 0.3174095335007367, -0.9372598131600525, 0.13923577975282667, -0.4418248181792963], [-0.04871133711512562, 0.09214689762898691, -0.05184237784909868, -0.024968197281294405, 1.0, -0.0026523599352053714, 0.0011739570621011667, -0.015177684316736793, -0.0014137906383861526, 0.028174190214377046, -0.012283160233883018, 0.01386813008630837], [0.005502872861730658, -0.014258218851366942, 0.008470584377005635, 0.08723294501206104, -0.0026523599352053714, 1.0, -0.02946212078343245, -0.031899532753575606, 0.21060128810167617, -0.08407078537281092, 0.18369429040738636, -0.041260031872491026], [0.005166182215155891, -0.012826315954965237, 0.006865005776331517, 0.05465195626019519, 0.0011739570621011667, -0.02946212078343245, 1.0, -0.017480282427881445, 0.14463294480060127, -0.0518419553222385, 0.09899034917492065, -0.02063854879199356], [-0.02495986419623275, 0.036701935556977684, -0.011012681789167195, 0.101331349900505, -0.015177684316736793, -0.031899532753575606, -0.017480282427881445, 1.0, 0.010695913707029587, -0.13958032139115895, 0.06235671120929189, -0.03585864033760225], [-0.0013196238696726917, -0.01771946125163068, 0.016809715425162304, 0.3174095335007367, -0.0014137906383861526, 0.21060128810167617, 0.14463294480060127, 0.010695913707029587, 1.0, -0.3033155860790147, 0.2156540707249929, -0.14209129778121454], [0.019051973345775068, 0.004540420578338916, -0.015355782095426042, -0.9372598131600525, 0.028174190214377046, -0.08407078537281092, -0.0518419553222385, -0.13958032139115895, -0.3033155860790147, 1.0, -0.1422851696099906, 0.4722873451813926], [0.018398281802604166, -0.027175514715190606, 0.0118575822568065, 0.13923577975282667, -0.012283160233883018, 0.18369429040738636, 0.09899034917492065, 0.06235671120929189, 0.2156540707249929, -0.1422851696099906, 1.0, -0.06170678301826289], [0.008206888372147265, 0.0036026694890132515, -0.0061481876651798, -0.4418248181792963, 0.01386813008630837, -0.041260031872491026, -0.02063854879199356, -0.03585864033760225, -0.14209129778121454, 0.4722873451813926, -0.06170678301826289, 1.0]]}],                        {"annotations": [{"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "1.0", "x": "dimension_pedido_med", "xref": "x", "y": "dimension_pedido_med", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.5529", "x": "dimension_pedido_peq", "xref": "x", "y": "dimension_pedido_med", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.2119", "x": "dimension_pedido_sob", "xref": "x", "y": "dimension_pedido_med", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0177", "x": "tipo_cliente_persona", "xref": "x", "y": "dimension_pedido_med", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0487", "x": "tipo_medicamento_normal", "xref": "x", "y": "dimension_pedido_med", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0055", "x": "velocidad_servicio_plus_2", "xref": "x", "y": "dimension_pedido_med", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0052", "x": "velocidad_servicio_plus_3", "xref": "x", "y": "dimension_pedido_med", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.025", "x": "velocidad_servicio_same_day", "xref": "x", "y": "dimension_pedido_med", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0013", "x": "tipo_envio_retiro_ventanilla", "xref": "x", "y": "dimension_pedido_med", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0191", "x": "tipo_induccion_retirado_a_cliente", "xref": "x", "y": "dimension_pedido_med", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0184", "x": "tipo_servicio_regional", "xref": "x", "y": "dimension_pedido_med", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0082", "x": "admite_a_tiempo_para_pick_up_si", "xref": "x", "y": "dimension_pedido_med", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.5529", "x": "dimension_pedido_med", "xref": "x", "y": "dimension_pedido_peq", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "1.0", "x": "dimension_pedido_peq", "xref": "x", "y": "dimension_pedido_peq", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.5341", "x": "dimension_pedido_sob", "xref": "x", "y": "dimension_pedido_peq", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0065", "x": "tipo_cliente_persona", "xref": "x", "y": "dimension_pedido_peq", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.0921", "x": "tipo_medicamento_normal", "xref": "x", "y": "dimension_pedido_peq", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0143", "x": "velocidad_servicio_plus_2", "xref": "x", "y": "dimension_pedido_peq", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0128", "x": "velocidad_servicio_plus_3", "xref": "x", "y": "dimension_pedido_peq", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.0367", "x": "velocidad_servicio_same_day", "xref": "x", "y": "dimension_pedido_peq", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0177", "x": "tipo_envio_retiro_ventanilla", "xref": "x", "y": "dimension_pedido_peq", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0045", "x": "tipo_induccion_retirado_a_cliente", "xref": "x", "y": "dimension_pedido_peq", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0272", "x": "tipo_servicio_regional", "xref": "x", "y": "dimension_pedido_peq", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0036", "x": "admite_a_tiempo_para_pick_up_si", "xref": "x", "y": "dimension_pedido_peq", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.2119", "x": "dimension_pedido_med", "xref": "x", "y": "dimension_pedido_sob", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.5341", "x": "dimension_pedido_peq", "xref": "x", "y": "dimension_pedido_sob", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "1.0", "x": "dimension_pedido_sob", "xref": "x", "y": "dimension_pedido_sob", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0162", "x": "tipo_cliente_persona", "xref": "x", "y": "dimension_pedido_sob", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0518", "x": "tipo_medicamento_normal", "xref": "x", "y": "dimension_pedido_sob", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0085", "x": "velocidad_servicio_plus_2", "xref": "x", "y": "dimension_pedido_sob", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0069", "x": "velocidad_servicio_plus_3", "xref": "x", "y": "dimension_pedido_sob", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.011", "x": "velocidad_servicio_same_day", "xref": "x", "y": "dimension_pedido_sob", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0168", "x": "tipo_envio_retiro_ventanilla", "xref": "x", "y": "dimension_pedido_sob", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0154", "x": "tipo_induccion_retirado_a_cliente", "xref": "x", "y": "dimension_pedido_sob", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0119", "x": "tipo_servicio_regional", "xref": "x", "y": "dimension_pedido_sob", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0061", "x": "admite_a_tiempo_para_pick_up_si", "xref": "x", "y": "dimension_pedido_sob", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0177", "x": "dimension_pedido_med", "xref": "x", "y": "tipo_cliente_persona", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0065", "x": "dimension_pedido_peq", "xref": "x", "y": "tipo_cliente_persona", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0162", "x": "dimension_pedido_sob", "xref": "x", "y": "tipo_cliente_persona", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "1.0", "x": "tipo_cliente_persona", "xref": "x", "y": "tipo_cliente_persona", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.025", "x": "tipo_medicamento_normal", "xref": "x", "y": "tipo_cliente_persona", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.0872", "x": "velocidad_servicio_plus_2", "xref": "x", "y": "tipo_cliente_persona", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.0547", "x": "velocidad_servicio_plus_3", "xref": "x", "y": "tipo_cliente_persona", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.1013", "x": "velocidad_servicio_same_day", "xref": "x", "y": "tipo_cliente_persona", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.3174", "x": "tipo_envio_retiro_ventanilla", "xref": "x", "y": "tipo_cliente_persona", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.9373", "x": "tipo_induccion_retirado_a_cliente", "xref": "x", "y": "tipo_cliente_persona", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.1392", "x": "tipo_servicio_regional", "xref": "x", "y": "tipo_cliente_persona", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.4418", "x": "admite_a_tiempo_para_pick_up_si", "xref": "x", "y": "tipo_cliente_persona", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0487", "x": "dimension_pedido_med", "xref": "x", "y": "tipo_medicamento_normal", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.0921", "x": "dimension_pedido_peq", "xref": "x", "y": "tipo_medicamento_normal", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0518", "x": "dimension_pedido_sob", "xref": "x", "y": "tipo_medicamento_normal", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.025", "x": "tipo_cliente_persona", "xref": "x", "y": "tipo_medicamento_normal", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "1.0", "x": "tipo_medicamento_normal", "xref": "x", "y": "tipo_medicamento_normal", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0027", "x": "velocidad_servicio_plus_2", "xref": "x", "y": "tipo_medicamento_normal", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0012", "x": "velocidad_servicio_plus_3", "xref": "x", "y": "tipo_medicamento_normal", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0152", "x": "velocidad_servicio_same_day", "xref": "x", "y": "tipo_medicamento_normal", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0014", "x": "tipo_envio_retiro_ventanilla", "xref": "x", "y": "tipo_medicamento_normal", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0282", "x": "tipo_induccion_retirado_a_cliente", "xref": "x", "y": "tipo_medicamento_normal", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0123", "x": "tipo_servicio_regional", "xref": "x", "y": "tipo_medicamento_normal", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0139", "x": "admite_a_tiempo_para_pick_up_si", "xref": "x", "y": "tipo_medicamento_normal", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0055", "x": "dimension_pedido_med", "xref": "x", "y": "velocidad_servicio_plus_2", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0143", "x": "dimension_pedido_peq", "xref": "x", "y": "velocidad_servicio_plus_2", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0085", "x": "dimension_pedido_sob", "xref": "x", "y": "velocidad_servicio_plus_2", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.0872", "x": "tipo_cliente_persona", "xref": "x", "y": "velocidad_servicio_plus_2", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0027", "x": "tipo_medicamento_normal", "xref": "x", "y": "velocidad_servicio_plus_2", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "1.0", "x": "velocidad_servicio_plus_2", "xref": "x", "y": "velocidad_servicio_plus_2", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0295", "x": "velocidad_servicio_plus_3", "xref": "x", "y": "velocidad_servicio_plus_2", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0319", "x": "velocidad_servicio_same_day", "xref": "x", "y": "velocidad_servicio_plus_2", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.2106", "x": "tipo_envio_retiro_ventanilla", "xref": "x", "y": "velocidad_servicio_plus_2", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0841", "x": "tipo_induccion_retirado_a_cliente", "xref": "x", "y": "velocidad_servicio_plus_2", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.1837", "x": "tipo_servicio_regional", "xref": "x", "y": "velocidad_servicio_plus_2", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0413", "x": "admite_a_tiempo_para_pick_up_si", "xref": "x", "y": "velocidad_servicio_plus_2", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0052", "x": "dimension_pedido_med", "xref": "x", "y": "velocidad_servicio_plus_3", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0128", "x": "dimension_pedido_peq", "xref": "x", "y": "velocidad_servicio_plus_3", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0069", "x": "dimension_pedido_sob", "xref": "x", "y": "velocidad_servicio_plus_3", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.0547", "x": "tipo_cliente_persona", "xref": "x", "y": "velocidad_servicio_plus_3", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0012", "x": "tipo_medicamento_normal", "xref": "x", "y": "velocidad_servicio_plus_3", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0295", "x": "velocidad_servicio_plus_2", "xref": "x", "y": "velocidad_servicio_plus_3", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "1.0", "x": "velocidad_servicio_plus_3", "xref": "x", "y": "velocidad_servicio_plus_3", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0175", "x": "velocidad_servicio_same_day", "xref": "x", "y": "velocidad_servicio_plus_3", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.1446", "x": "tipo_envio_retiro_ventanilla", "xref": "x", "y": "velocidad_servicio_plus_3", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0518", "x": "tipo_induccion_retirado_a_cliente", "xref": "x", "y": "velocidad_servicio_plus_3", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.099", "x": "tipo_servicio_regional", "xref": "x", "y": "velocidad_servicio_plus_3", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0206", "x": "admite_a_tiempo_para_pick_up_si", "xref": "x", "y": "velocidad_servicio_plus_3", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.025", "x": "dimension_pedido_med", "xref": "x", "y": "velocidad_servicio_same_day", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.0367", "x": "dimension_pedido_peq", "xref": "x", "y": "velocidad_servicio_same_day", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.011", "x": "dimension_pedido_sob", "xref": "x", "y": "velocidad_servicio_same_day", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.1013", "x": "tipo_cliente_persona", "xref": "x", "y": "velocidad_servicio_same_day", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0152", "x": "tipo_medicamento_normal", "xref": "x", "y": "velocidad_servicio_same_day", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0319", "x": "velocidad_servicio_plus_2", "xref": "x", "y": "velocidad_servicio_same_day", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0175", "x": "velocidad_servicio_plus_3", "xref": "x", "y": "velocidad_servicio_same_day", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "1.0", "x": "velocidad_servicio_same_day", "xref": "x", "y": "velocidad_servicio_same_day", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0107", "x": "tipo_envio_retiro_ventanilla", "xref": "x", "y": "velocidad_servicio_same_day", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.1396", "x": "tipo_induccion_retirado_a_cliente", "xref": "x", "y": "velocidad_servicio_same_day", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.0624", "x": "tipo_servicio_regional", "xref": "x", "y": "velocidad_servicio_same_day", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0359", "x": "admite_a_tiempo_para_pick_up_si", "xref": "x", "y": "velocidad_servicio_same_day", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0013", "x": "dimension_pedido_med", "xref": "x", "y": "tipo_envio_retiro_ventanilla", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0177", "x": "dimension_pedido_peq", "xref": "x", "y": "tipo_envio_retiro_ventanilla", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0168", "x": "dimension_pedido_sob", "xref": "x", "y": "tipo_envio_retiro_ventanilla", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.3174", "x": "tipo_cliente_persona", "xref": "x", "y": "tipo_envio_retiro_ventanilla", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0014", "x": "tipo_medicamento_normal", "xref": "x", "y": "tipo_envio_retiro_ventanilla", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.2106", "x": "velocidad_servicio_plus_2", "xref": "x", "y": "tipo_envio_retiro_ventanilla", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.1446", "x": "velocidad_servicio_plus_3", "xref": "x", "y": "tipo_envio_retiro_ventanilla", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0107", "x": "velocidad_servicio_same_day", "xref": "x", "y": "tipo_envio_retiro_ventanilla", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "1.0", "x": "tipo_envio_retiro_ventanilla", "xref": "x", "y": "tipo_envio_retiro_ventanilla", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.3033", "x": "tipo_induccion_retirado_a_cliente", "xref": "x", "y": "tipo_envio_retiro_ventanilla", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.2157", "x": "tipo_servicio_regional", "xref": "x", "y": "tipo_envio_retiro_ventanilla", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.1421", "x": "admite_a_tiempo_para_pick_up_si", "xref": "x", "y": "tipo_envio_retiro_ventanilla", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0191", "x": "dimension_pedido_med", "xref": "x", "y": "tipo_induccion_retirado_a_cliente", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0045", "x": "dimension_pedido_peq", "xref": "x", "y": "tipo_induccion_retirado_a_cliente", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0154", "x": "dimension_pedido_sob", "xref": "x", "y": "tipo_induccion_retirado_a_cliente", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.9373", "x": "tipo_cliente_persona", "xref": "x", "y": "tipo_induccion_retirado_a_cliente", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0282", "x": "tipo_medicamento_normal", "xref": "x", "y": "tipo_induccion_retirado_a_cliente", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0841", "x": "velocidad_servicio_plus_2", "xref": "x", "y": "tipo_induccion_retirado_a_cliente", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0518", "x": "velocidad_servicio_plus_3", "xref": "x", "y": "tipo_induccion_retirado_a_cliente", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.1396", "x": "velocidad_servicio_same_day", "xref": "x", "y": "tipo_induccion_retirado_a_cliente", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.3033", "x": "tipo_envio_retiro_ventanilla", "xref": "x", "y": "tipo_induccion_retirado_a_cliente", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "1.0", "x": "tipo_induccion_retirado_a_cliente", "xref": "x", "y": "tipo_induccion_retirado_a_cliente", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.1423", "x": "tipo_servicio_regional", "xref": "x", "y": "tipo_induccion_retirado_a_cliente", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.4723", "x": "admite_a_tiempo_para_pick_up_si", "xref": "x", "y": "tipo_induccion_retirado_a_cliente", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0184", "x": "dimension_pedido_med", "xref": "x", "y": "tipo_servicio_regional", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0272", "x": "dimension_pedido_peq", "xref": "x", "y": "tipo_servicio_regional", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0119", "x": "dimension_pedido_sob", "xref": "x", "y": "tipo_servicio_regional", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.1392", "x": "tipo_cliente_persona", "xref": "x", "y": "tipo_servicio_regional", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0123", "x": "tipo_medicamento_normal", "xref": "x", "y": "tipo_servicio_regional", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.1837", "x": "velocidad_servicio_plus_2", "xref": "x", "y": "tipo_servicio_regional", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.099", "x": "velocidad_servicio_plus_3", "xref": "x", "y": "tipo_servicio_regional", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.0624", "x": "velocidad_servicio_same_day", "xref": "x", "y": "tipo_servicio_regional", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.2157", "x": "tipo_envio_retiro_ventanilla", "xref": "x", "y": "tipo_servicio_regional", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.1423", "x": "tipo_induccion_retirado_a_cliente", "xref": "x", "y": "tipo_servicio_regional", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "1.0", "x": "tipo_servicio_regional", "xref": "x", "y": "tipo_servicio_regional", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0617", "x": "admite_a_tiempo_para_pick_up_si", "xref": "x", "y": "tipo_servicio_regional", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0082", "x": "dimension_pedido_med", "xref": "x", "y": "admite_a_tiempo_para_pick_up_si", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0036", "x": "dimension_pedido_peq", "xref": "x", "y": "admite_a_tiempo_para_pick_up_si", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0061", "x": "dimension_pedido_sob", "xref": "x", "y": "admite_a_tiempo_para_pick_up_si", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.4418", "x": "tipo_cliente_persona", "xref": "x", "y": "admite_a_tiempo_para_pick_up_si", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "0.0139", "x": "tipo_medicamento_normal", "xref": "x", "y": "admite_a_tiempo_para_pick_up_si", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0413", "x": "velocidad_servicio_plus_2", "xref": "x", "y": "admite_a_tiempo_para_pick_up_si", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0206", "x": "velocidad_servicio_plus_3", "xref": "x", "y": "admite_a_tiempo_para_pick_up_si", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0359", "x": "velocidad_servicio_same_day", "xref": "x", "y": "admite_a_tiempo_para_pick_up_si", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.1421", "x": "tipo_envio_retiro_ventanilla", "xref": "x", "y": "admite_a_tiempo_para_pick_up_si", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "0.4723", "x": "tipo_induccion_retirado_a_cliente", "xref": "x", "y": "admite_a_tiempo_para_pick_up_si", "yref": "y"}, {"font": {"color": "#FFFFFF", "size": 8}, "showarrow": false, "text": "-0.0617", "x": "tipo_servicio_regional", "xref": "x", "y": "admite_a_tiempo_para_pick_up_si", "yref": "y"}, {"font": {"color": "#000000", "size": 8}, "showarrow": false, "text": "1.0", "x": "admite_a_tiempo_para_pick_up_si", "xref": "x", "y": "admite_a_tiempo_para_pick_up_si", "yref": "y"}], "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "autotypenumbers": "strict", "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"dtick": 1, "gridcolor": "rgb(0, 0, 0)", "side": "top", "ticks": ""}, "yaxis": {"dtick": 1, "ticks": "", "ticksuffix": "  "}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('0b1bbe28-efd2-4b4f-a37d-6b166bc3f711');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



    <Figure size 2880x2160 with 0 Axes>


* Los atributos **tipo_cliente (persona)** y **tipo_envio (Retiro en Ventanilla)**, tienen una correlación (+) de (0.3174). Este valor se justifica dado que el segmento contado, asociado a personas naturales prioriza el costo del envío por sobre la comodidad del retiro, priorizando el menor valor de una entrega de "Retiro en Sucursal", sobre el "Entrega a domicilio".

* Los atributos **tipo_cliente (persona)** y **admite_a_tiempo_para_pick_up (si), tienen una correlación (-) de (-0.4418). Este valor se justifica dado que el segmento persona esta asociado casi excluisivamente a este tipo de inducción, del que hacen uso los tipo de clientes Empresa, para la logística en devolución.

### Visualización de Outliers

Se realizan diagramas de cajas para revisar presencia de outliers:


```python
box_plots(corr_matrix,2)
```


![png](output_73_0.png)


Se puede ver una gran cantidad de valores fuera de las cajas, lo que muestra alta presencia de valores escapados, sin embargo, pueden estar muy relacionados con la escala de los valores, por ejemplo en el caso del valor contratado el mínimo es 1.73 y el máximo sobre pasa los 6 millones,todo esto nos indica que una normalización del tipo logarítmica nos ayudaría a regularizar los datos. Se probará esta transformación cuando se realicen los modelos de clusterización y clasificación.

### Test de medias

Para efectos de conocer el comportamiento de los atributos continuos según el vector objetivo, se procede a aplicar un Test de medias y así vislumbrar potenciales atributos con significancia estadística puedan ser usados en los posteriores análisis y modelos de este proyecto.


```python
mean_test(df,'valor_contratado')
```

    t = -226.73789542430694
    p = 2.0



```python
mean_test(df,'distancia_envio_mts')
```

    t = -649.4974124293791
    p = 2.0



```python
mean_test(df,'horas_desde_creacion_hasta_compromiso')
```

    t = 507.56059374033447
    p = 0.0



```python
mean_test(df,'horas_desde_creacion_hasta_salida_primera_milla')
```

    t = -252.14155420008012
    p = 2.0


* Se puede observar que efectivamente, los tiempos de entrega determinarían el que sea exitoso un despacho o no. Lógicamente esto estaría condicionado a los intentos de entrega que hace la compañía.

## Identificación segmentos de despachos


En este apartado el objetivo es encontrar los principales tipos de despachos que realiza FAST, y así determinar cuáles son los más propensos a que sean fallidos al término de su trayecto. Esto ayudaría a proponer acciones correctivas y enfocadas en el negocio.

En esta línea, desde una mirada exploratoria se definirán las variables son más ilustrativas para segmentar según una visualización de redes, aplicando teoría de grafos.

Finalmente, se implementará un modelo de clustering K-Means y el método de Elbow para definir cuántos grupos se pueden usar para la segmentación.


#### Visualización Grafo

Se genera una nueva categoria segun los cuartiles de la distancia en metros:
1. Tramo corto: primer cuartil. (hasta 16025 metros)
2. Tramo medio: segundo cuartil. (de 16026 a 90530 metros)
3. Tramo largo: tercer cuartil. (de 90531 a 464025 metros)
4. Tramo extra: cuarto cuartil. (sobre los 46026 metros)


```python
df['tipo_tramo'] = np.where(df['distancia_envio_mts'].between(0, 16025),'TRAMO CORTO',
                            np.where(df['distancia_envio_mts'].between(16026,90530),'TRAMO MEDIO',
                                     np.where(df['distancia_envio_mts'].between(90531, 464025),'TRAMO LARGO',
                                     'TRAMO EXTRA' ) ))
```

Revisamos la distribucion de la nueva cagtegoria y vemos que queda de manera bastante balanceada:


```python
sns.countplot(x="tipo_tramo", data=df);
```


![png](output_84_0.png)



```python
atr = dict(df['tipo_cliente'].value_counts())
atr.update(dict(df['tipo_envio'].value_counts()))
atr.update(dict(df['tipo_induccion'].value_counts()))
atr.update(dict(df['tipo_servicio'].value_counts()))
atr.update(dict(df['tipo_tramo'].value_counts()))
atr.update(dict(df['tramo_hr_desp'].value_counts()))
related = {}

for ix, it in df.iterrows():
    tc = it['tipo_cliente']
    te = it['tipo_envio']
    ti = it['tipo_induccion']
    ts = it['tipo_servicio']
    tt = it['tipo_tramo']
    th = it['tramo_hr_desp']
    #print(tc,te, ti,ts, tt)
    if tc not in related:
        related[tc] = {}
    if te not in related[tc]:
        d = {te: 1}
        related[tc].update(d)
        related[te] = {}
    else:
        d = {te: related[tc][te] + 1}
        related[tc].update(d)
    if ti not in related[te]:
        d = {ti:1}
        related[te].update(d)
        related[ti] = {}
    else:
        d = {ti: related[te][ti] + 1}
        related[te].update(d)
    if ts not in related[ti]:
        d = {ts:1}
        related[ti].update(d)
        related[ts] = {}
    else:
        d = {ts: related[ti][ts] + 1}
        related[ti].update(d)
    if tt not in related[ts]:
        d = {tt:1}
        related[ts].update(d)
        related[tt] = {}
    else:
        d = {tt: related[ts][tt] + 1}
        related[ts].update(d)
        
    if th not in related[tt]:
        d = {th:1}
        related[tt].update(d)
        related[th] = {}
    else:
        d = {th: related[tt][th] + 1}
        related[tt].update(d)
g = nx.Graph()
# Add node for each character
for at in atr.keys():
    if atr[at] > 0:
        g.add_node(at, size = atr[at])
for rel in related.keys():
    for co_rel in related[rel].keys():
        
        # Only add edge if the count is positive
        if related[rel][co_rel] > 0:
            g.add_edge(rel, co_rel, weight = related[rel][co_rel])
pos_ = nx.spring_layout(g,iterations=10)
# For each edge, make an edge_trace, append to list
edge_trace = []
for edge in g.edges():
    
    if g.edges()[edge]['weight'] > 0:
        char_1 = edge[0]
        char_2 = edge[1]

        x0, y0 = pos_[char_1]
        x1, y1 = pos_[char_2]

        text   = char_1 + '--' + char_2 + ': ' + str(g.edges()[edge]['weight'])
        
        trace  = make_edge([x0, x1, None], [y0, y1, None], text,
                           0.000004*g.edges()[edge]['weight'])

        edge_trace.append(trace)
node_trace = go.Scatter(x         = [],
                        y         = [],
                        text      = [],
                        textposition = "top center",
                        textfont_size = 10,
                        mode      = 'markers+text',
                        hoverinfo = 'none',
                        marker    = dict(color = [],
                                         size  = [],
                                         line  = None))
# For each node in midsummer, get the position and size and add to the node_trace
for node in g.nodes():
    x, y = pos_[node]
    
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    node_trace['marker']['color'] += tuple(['cornflowerblue'])
    node_trace['marker']['size'] += tuple([0.000007*g.nodes()[node]['size']])
    node_trace['text'] += tuple(['<b>' + node + '</b>'])
layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

#grafico
fig = go.Figure(layout = layout)

for trace in edge_trace:
    fig.add_trace(trace)

fig.add_trace(node_trace)

fig.update_layout(showlegend = False)

fig.update_xaxes(showticklabels = False)

fig.update_yaxes(showticklabels = False)

fig.show()
```


<div>                            <div id="a8ca8d24-0be0-42fb-93ee-5616af12313c" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("a8ca8d24-0be0-42fb-93ee-5616af12313c")) {                    Plotly.newPlot(                        "a8ca8d24-0be0-42fb-93ee-5616af12313c",                        [{"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 8.736616}, "mode": "lines", "text": ["empresa--a_la_puerta: 2184154"], "type": "scatter", "x": [-0.358744319171606, -0.17794378399931668, null], "y": [1.0, 0.6929877059296329, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 0.7395039999999999}, "mode": "lines", "text": ["empresa--retiro_ventanilla: 184876"], "type": "scatter", "x": [-0.358744319171606, -0.14351098665332565, null], "y": [1.0, 0.7408861980555871, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 0.44488}, "mode": "lines", "text": ["persona--a_la_puerta: 111220"], "type": "scatter", "x": [-0.30326887848379963, -0.17794378399931668, null], "y": [0.7734793804030796, 0.6929877059296329, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 0.35459999999999997}, "mode": "lines", "text": ["persona--retiro_ventanilla: 88650"], "type": "scatter", "x": [-0.30326887848379963, -0.14351098665332565, null], "y": [0.7734793804030796, 0.7408861980555871, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 8.648784}, "mode": "lines", "text": ["a_la_puerta--retirado_a_cliente: 2162196"], "type": "scatter", "x": [-0.17794378399931668, -0.04725739647050724, null], "y": [0.6929877059296329, 0.4171628342902587, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 0.532708}, "mode": "lines", "text": ["a_la_puerta--dejado_sucursal: 133177"], "type": "scatter", "x": [-0.17794378399931668, 0.023457006557661947, null], "y": [0.6929877059296329, 0.3136346272368848, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 0.36769199999999996}, "mode": "lines", "text": ["retiro_ventanilla--dejado_sucursal: 91923"], "type": "scatter", "x": [-0.14351098665332565, 0.023457006557661947, null], "y": [0.7408861980555871, 0.3136346272368848, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 0.7264079999999999}, "mode": "lines", "text": ["retiro_ventanilla--retirado_a_cliente: 181602"], "type": "scatter", "x": [-0.14351098665332565, -0.04725739647050724, null], "y": [0.7408861980555871, 0.4171628342902587, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 5.34798}, "mode": "lines", "text": ["retirado_a_cliente--regional: 1336995"], "type": "scatter", "x": [-0.04725739647050724, -0.06006888482318385, null], "y": [0.4171628342902587, -0.25635681648924485, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 4.027152}, "mode": "lines", "text": ["retirado_a_cliente--local: 1006788"], "type": "scatter", "x": [-0.04725739647050724, 0.3604388919161298, null], "y": [0.4171628342902587, -0.11156493055433592, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 0.7362799999999999}, "mode": "lines", "text": ["dejado_sucursal--regional: 184070"], "type": "scatter", "x": [0.023457006557661947, -0.06006888482318385, null], "y": [0.3136346272368848, -0.25635681648924485, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 0.164092}, "mode": "lines", "text": ["dejado_sucursal--local: 41023"], "type": "scatter", "x": [0.023457006557661947, 0.3604388919161298, null], "y": [0.3136346272368848, -0.11156493055433592, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 2.56834}, "mode": "lines", "text": ["regional--TRAMO EXTRA: 642085"], "type": "scatter", "x": [-0.06006888482318385, -0.11950342149826342, null], "y": [-0.25635681648924485, -0.6499178333067342, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 2.540704}, "mode": "lines", "text": ["regional--TRAMO LARGO: 635176"], "type": "scatter", "x": [-0.06006888482318385, -0.07712858663172475, null], "y": [-0.25635681648924485, -0.6263415288466287, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 0.97512}, "mode": "lines", "text": ["regional--TRAMO MEDIO: 243780"], "type": "scatter", "x": [-0.06006888482318385, 0.2681216946344595, null], "y": [-0.25635681648924485, -0.4105747955550026, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 8.4e-05}, "mode": "lines", "text": ["regional--TRAMO CORTO: 21"], "type": "scatter", "x": [-0.06006888482318385, 0.3996655237605549, null], "y": [-0.25635681648924485, -0.3977622289148215, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 2.614624}, "mode": "lines", "text": ["local--TRAMO CORTO: 653656"], "type": "scatter", "x": [0.3604388919161298, 0.3996655237605549, null], "y": [-0.11156493055433592, -0.3977622289148215, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 1.5483799999999999}, "mode": "lines", "text": ["local--TRAMO MEDIO: 387095"], "type": "scatter", "x": [0.3604388919161298, 0.2681216946344595, null], "y": [-0.11156493055433592, -0.4105747955550026, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 0.028235999999999997}, "mode": "lines", "text": ["local--TRAMO LARGO: 7059"], "type": "scatter", "x": [0.3604388919161298, -0.07712858663172475, null], "y": [-0.11156493055433592, -0.6263415288466287, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 1.693336}, "mode": "lines", "text": ["TRAMO CORTO--menor_48: 423334"], "type": "scatter", "x": [0.3996655237605549, 0.1516435244150345, null], "y": [-0.3977622289148215, -0.7530092683522245, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 0.903148}, "mode": "lines", "text": ["TRAMO CORTO--mayor_48: 225787"], "type": "scatter", "x": [0.3996655237605549, 0.08409961644788533, null], "y": [-0.3977622289148215, -0.7326233438964495, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 0.9507559999999999}, "mode": "lines", "text": ["TRAMO LARGO--mayor_48: 237689"], "type": "scatter", "x": [-0.07712858663172475, 0.08409961644788533, null], "y": [-0.6263415288466287, -0.7326233438964495, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 1.618044}, "mode": "lines", "text": ["TRAMO LARGO--menor_48: 404511"], "type": "scatter", "x": [-0.07712858663172475, 0.1516435244150345, null], "y": [-0.6263415288466287, -0.7530092683522245, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 1.194964}, "mode": "lines", "text": ["TRAMO EXTRA--mayor_48: 298741"], "type": "scatter", "x": [-0.11950342149826342, 0.08409961644788533, null], "y": [-0.6499178333067342, -0.7326233438964495, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 1.373376}, "mode": "lines", "text": ["TRAMO EXTRA--menor_48: 343344"], "type": "scatter", "x": [-0.11950342149826342, 0.1516435244150345, null], "y": [-0.6499178333067342, -0.7530092683522245, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 0.8551639999999999}, "mode": "lines", "text": ["TRAMO MEDIO--mayor_48: 213791"], "type": "scatter", "x": [0.2681216946344595, 0.08409961644788533, null], "y": [-0.4105747955550026, -0.7326233438964495, null]}, {"hoverinfo": "text", "line": {"color": "cornflowerblue", "width": 1.668324}, "mode": "lines", "text": ["TRAMO MEDIO--menor_48: 417081"], "type": "scatter", "x": [0.2681216946344595, 0.1516435244150345, null], "y": [-0.4105747955550026, -0.7530092683522245, null]}, {"hoverinfo": "none", "marker": {"color": ["cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue"], "size": [16.58321, 1.39909, 16.067618, 1.914682, 16.406593, 1.575707, 10.647594999999999, 7.334705, 4.57576, 4.495715, 4.494679, 4.416146, 11.144608999999999, 6.8376909999999995]}, "mode": "markers+text", "text": ["<b>empresa</b>", "<b>persona</b>", "<b>a_la_puerta</b>", "<b>retiro_ventanilla</b>", "<b>retirado_a_cliente</b>", "<b>dejado_sucursal</b>", "<b>regional</b>", "<b>local</b>", "<b>TRAMO CORTO</b>", "<b>TRAMO LARGO</b>", "<b>TRAMO EXTRA</b>", "<b>TRAMO MEDIO</b>", "<b>menor_48</b>", "<b>mayor_48</b>"], "textfont": {"size": 10}, "textposition": "top center", "type": "scatter", "x": [-0.358744319171606, -0.30326887848379963, -0.17794378399931668, -0.14351098665332565, -0.04725739647050724, 0.023457006557661947, -0.06006888482318385, 0.3604388919161298, 0.3996655237605549, -0.07712858663172475, -0.11950342149826342, 0.2681216946344595, 0.1516435244150345, 0.08409961644788533], "y": [1.0, 0.7734793804030796, 0.6929877059296329, 0.7408861980555871, 0.4171628342902587, 0.3136346272368848, -0.25635681648924485, -0.11156493055433592, -0.3977622289148215, -0.6263415288466287, -0.6499178333067342, -0.4105747955550026, -0.7530092683522245, -0.7326233438964495]}],                        {"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "showlegend": false, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "autotypenumbers": "strict", "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"showticklabels": false}, "yaxis": {"showticklabels": false}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('a8ca8d24-0be0-42fb-93ee-5616af12313c');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


* Existen 2 macrosegmentos de despachos, concentrándose principalmente en el grupo de clientes Empresa, con envíos regionales, distancias mayores a los 90 km y con un servicio de ir a buscar el producto donde el cliente mismo y repartirlo hasta a la puerta del cliente final, asimismo, tiende a entregar sus despachos en un plazo máximo de 48 hrs.


* Por otra parte, como el Macrosegmento 1 conforma sobre el 95% de la cartera de clientes, se puede encontrar un subgrupo que tiene repartos locales y que el producto se deje en una sucursal.


* Macrosegmento 2 corresponde a personas que realizan despachos locales y regionales con un tipo de inducción en una oficina de FAST y retiro en la ventanilla.

### K-Means


#### Estandarización de variables

Se crea un dataframe con los atributos de las variables para revisar presencia de clusters, en el caso de variables categóricas se transfroman con un label enconder y para las continuas una normalización logartímica.


```python
#label encoder
le = LabelEncoder()
#se crea dataframe con atributos del cluster
clustering = df.loc[:,['tipo_cliente','tipo_envio','tipo_induccion'
                    ,'tipo_servicio','tipo_tramo','distancia_envio_mts'
              ,'horas_desde_creacion_hasta_compromiso']].copy()
categorical_cols = ['tipo_cliente','tipo_envio','tipo_induccion'
                    ,'tipo_servicio','tipo_tramo']
float_cols = ['distancia_envio_mts','horas_desde_creacion_hasta_compromiso']
# para las variables categoricas se transforman con label enconder y para las continuas una normalizacion logaritmica
clustering[categorical_cols] = clustering[categorical_cols].apply(lambda col: le.fit_transform(col))
clustering[float_cols] = clustering[float_cols].apply(lambda col: np.log1p(col))
clustering.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tipo_cliente</th>
      <th>tipo_envio</th>
      <th>tipo_induccion</th>
      <th>tipo_servicio</th>
      <th>tipo_tramo</th>
      <th>distancia_envio_mts</th>
      <th>horas_desde_creacion_hasta_compromiso</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>13.31</td>
      <td>4.20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>12.09</td>
      <td>2.83</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>13.34</td>
      <td>3.30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>14.12</td>
      <td>4.53</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>13.73</td>
      <td>3.09</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>13.20</td>
      <td>4.29</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>14.32</td>
      <td>2.89</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>12.88</td>
      <td>2.89</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>10.13</td>
      <td>3.04</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>12.84</td>
      <td>2.83</td>
    </tr>
  </tbody>
</table>
</div>



#### Evaluación de número de clusters

Se genera un modelo de Kmeans y se evaluan sus resultados segun diferntes metodos:
##### Elbow Method


```python
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,10),timings=True)
visualizer.fit(clustering)
visualizer.poof(); 
```


![png](output_90_0.png)


El gráfico muestra que con 4 clusters se entrega el modelo más eficiente, segun la metrica de 'distortion' que se calcula comparando la suma de las distancias cuadradas para cada centro.


```python
visualizer = KElbowVisualizer(model, k=(2,10), metric='calinski_harabasz', timings=True)
visualizer.fit(clustering)    # Fit the data to the visualizer
visualizer.poof();
```


![png](output_92_0.png)


Utilizando la métrica de 'calinski harabasz' que se basa en el radio de dispersión de entre y para cada cluster, el número más eficiente para el modelo también son 4 grupos.

Dado los resultados mostrados anteriormente, se procede a aplicar el modelo KMeans con un número de 4 grupos, obteniendo los siguientes resultados:


```python
model = KMeans(4,random_state=465)
model.fit(clustering)
```




    KMeans(n_clusters=4, random_state=465)



Se agrega el atributo del cluster al que pertenece cada registro a nuestro dataframe para revisar su distribución,


```python
df['cluster'] = model.labels_
df['cluster'].value_counts('%')
```




    2   0.43
    1   0.31
    0   0.22
    3   0.03
    Name: cluster, dtype: float64



Se puede ver que la mayoría de los registros pertenecen al cluster 0 con un 43% de la data, luego al cluster 3 con un 31%, al cluster 1 con el 22% y finalmente al cluster 2 con el 3%.<br>
Se renombra cada cluster con las siguientes categorias:
* Cluster 2: 'Los Patiperros'
* Cluster 1: 'Los poco optimizados'
* Cluster 0: 'Business Package'
* Cluster 3: 'Los Enfocados'


```python
df["desc_cluster"] = df["cluster"].replace([0,2,3,1]
                                 ,["Business Package","Los Patiperros","Los Enfocados","Los Poco Optimizados"])  
```

#### Distribución de Variables continuas por Cluster

Se grafican en boxplots las atirbutos continuos para cada cluster y ver su comportamiento:


```python
cob_columns_cont = []
cob_columns_cate = []

for index, (columnas_df,serie) in enumerate(df.iteritems()):
    if pd.api.types.is_float_dtype(serie) is True: 
        cob_columns_cont.append(columnas_df)
    else: 
        if pd.api.types.is_integer_dtype(serie) is True: 
            cob_columns_cont.append(columnas_df)
        else:
            cob_columns_cate.append(columnas_df)

cob_columns_cont.remove("cluster")   
cob_columns_cate.remove("desc_cluster")   
cob_columns_cate.remove("comuna_origen")
cob_columns_cate.remove("comuna_destino")   
plt.figure(figsize=(25,15))
for index,col in enumerate(cob_columns_cont):
    plt.subplot(3,3,index +1)
    sns.boxplot(x=df['desc_cluster'], y = df[col])
```


![png](output_102_0.png)


Conclusiones,
* Se puede apreciar que el cluster de los Patiperros, es el grupo con mayor variabilidad en la distancia recorrida abarcando las mayores distancias, por esta razón se le bautiza de esa forma. Lo siguen los poco optimizados, luego los business package y finalmente los enfocados.
* Los Patiperros presentan la mayor cantidad de outliers en el valor contratado, mostrando mayor variabilidad versus el resto de los clusters.
* Con respecto a las horas desde creación hasta compromiso o hasta la primera milla se ve una distribución bastante uniforme, independiente del tipo de cluster los valores seran parecidos. Esto nos indica que puede existir cierta política de compromiso que se aplica a todos los pedidos por igual, independiente de su naturaleza.

#### Distribución de Variables categóricas por Cluster


```python
plt.figure(figsize=(25,25))
for index,col in enumerate(cob_columns_cate):
    plt.subplot(6,3,index +1)
    sns.countplot(x=col, data = df, hue="desc_cluster")
```


![png](output_105_0.png)


Conclusiones,
* En el último gráfico se puede apreciar de mejor manera la distribución de los cluster según su distancia, algo abarcado en la sección anterior, donde los patiperros se componen por tramos extras y largos, y los poco optimizados son los largos y medios. Los clusters de business package y enfocados son los tramos cortos a nivel local.
* En el primer gráfico, se puede ver que independiente del tamaño del pedido, la distribución de los cluster es parecida para cada grupo.

## Predicción tasa de despachos fallidos

Como se revisó en la seccion de analisis descriptivo, las variables continuas presentaban una alta variabilidad en escala y presencia de outliers, por lo que se decide por aplicar una trasnformacion logaritmica antes de continuar con el desarrollo del modelo predictivo.


```python
### Logaritmo y graficos
continuas =['valor_contratado','distancia_envio_mts','horas_desde_creacion_hasta_compromiso','horas_desde_creacion_hasta_salida_primera_milla']
df['valor_contratado'] = np.log1p(df['valor_contratado'])
df['distancia_envio_mts'] = np.log1p(df['distancia_envio_mts'])
df['horas_desde_creacion_hasta_compromiso'] = np.log1p(df['horas_desde_creacion_hasta_compromiso'])
df['horas_desde_creacion_hasta_salida_primera_milla'] = np.log1p(df['horas_desde_creacion_hasta_salida_primera_milla'])
eda_plots(df[continuas],2)
```


![png](output_109_0.png)


Ahora se nota una distribución más parecida a una normal, por lo que se opta por trabajar los datos de esta manera. Se observan valores cero para el atributo distancia_envio_mts (3.2%), esto se da porque las distancias se calculan entre centros operacionales, y en algunos casos de tipos de envios "LOCALES", las distribución se realiza por el mismo centro operacional que la admite.
* Para solucionar este tema se asignara la media del primer quantil del atributo distancia_envio_mts.


```python
df["distancia_envio_mts"].describe()
df["distancia_envio_mts"].replace([0],df["distancia_envio_mts"].describe()[4],inplace = True)  
```

Se revisan resultados del cambio y se aprecia un comportamiento mucho más normalizado que antes,


```python
eda_plots(df[continuas],2)
```


![png](output_113_0.png)


### Preselección Atributos

* Se excluirán los atributos relacionados con las "Comunas", ya que son muchas clases y bastante desbalanceadas, lo que podria llevar a un overfit del modelo segun la comuna.
* Se excluyen datos como día de semana de entrega, satisfacción del clientes y tramos de hr, ya que son atributos que se conocen una vez entregado el pedido y no antes.
* Se excluyen variables creadas para análisis anteriores como tramo de entrega y clusters generados.


```python
df=df.drop(columns=['comuna_destino','comuna_origen','satisfaccion_cliente','dia_semana_entrega','tramo_hr_desp','dia_semana_admision','tipo_tramo','cluster','desc_cluster'])
df.columns
```




    Index(['dimension_pedido', 'tipo_cliente', 'tipo_medicamento', 'region_origen',
           'region_destino', 'score_datos_contactabilidad_destintario',
           'score_datos_contactabilidad_remitente', 'velocidad_servicio',
           'tipo_envio', 'tipo_induccion', 'cumplimiento', 'valor_contratado',
           'distancia_envio_mts', 'tipo_servicio', 'admite_a_tiempo_para_pick_up',
           'horas_desde_creacion_hasta_compromiso',
           'horas_desde_creacion_hasta_salida_primera_milla'],
          dtype='object')



### Binarización de variables categóricas

Se procede a binarizar las variables que son del tipo categórico:


```python
df_modelo = pd.get_dummies(df, columns=['dimension_pedido','region_origen',
                                          'region_destino','tipo_cliente',
                                          'tipo_medicamento','velocidad_servicio',
                                          'tipo_envio','tipo_induccion',
                                          'tipo_servicio','admite_a_tiempo_para_pick_up'], drop_first=True)
```

### Modelos candidatos

#### Preparación de variable objetivo.


```python
df_modelo["cumplimiento"].replace(["si","no"]
                                 ,[1,0],inplace = True)  
```

#### Preparación de Muestras de entrenamiento y testeo.


```python
columnas = df_modelo.columns
cols = list(columnas)
cols.remove("cumplimiento")
X_train, X_test,y_train,y_test = train_test_split(df_modelo.loc[:,cols],df_modelo["cumplimiento"],test_size=.33,random_state=202101)
```

#### Modelo AdaBoostClassifier 


```python
%%time
#Modelo Generico AdaBoostClassifier
Modelo_AdaBoost = AdaBoostClassifier()
Modelo_AdaBoost.fit(X_train, y_train)

y_hat_AdaBoost = Modelo_AdaBoost.predict(X_test)
```

    CPU times: user 2min 22s, sys: 15.3 s, total: 2min 38s
    Wall time: 2min 38s



```python
metrics_model(Modelo_AdaBoost,X_test,y_test)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.76</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.76</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.54</td>
      <td>0.89</td>
      <td>0.75</td>
      <td>0.72</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.63</td>
      <td>0.81</td>
      <td>0.75</td>
      <td>0.72</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>support</th>
      <td>333584.00</td>
      <td>514153.00</td>
      <td>0.75</td>
      <td>847737.00</td>
      <td>847737.00</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_127_1.png)



![png](output_127_2.png)



```python
#dump(Modelo_AdaBoost, "aboost_model.joblib")
```

A nivel general, el modelo de AdaBoostClassifier tiene buen nivel de prediccion, con un accuracy del 75,4%. Sin embargo, al revisar su desempeño por categoria vemos que presenta un bajo rendimiento para los pedidos 0, es decir, los que no cumplen la entrega segun su promesa, con un recall de 54% lo cual es bastante bajo.

### Modelo Decission Tree Classifier


```python
%%time
dtc_model = DecisionTreeClassifier()
dtc_model.fit(X_train,y_train)
y_hat_dtc = dtc_model.predict(X_test)
```

    CPU times: user 35.7 s, sys: 490 ms, total: 36.2 s
    Wall time: 36.2 s



```python
metrics_model(dtc_model,X_test,y_test)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.71</td>
      <td>0.82</td>
      <td>0.77</td>
      <td>0.76</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.72</td>
      <td>0.81</td>
      <td>0.77</td>
      <td>0.76</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.71</td>
      <td>0.81</td>
      <td>0.77</td>
      <td>0.76</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>support</th>
      <td>333584.00</td>
      <td>514153.00</td>
      <td>0.77</td>
      <td>847737.00</td>
      <td>847737.00</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_132_1.png)



![png](output_132_2.png)



```python
#dump(dtc_model, "dtc_model.joblib")
```

A nivel general, el modelo de DecisionTreeClassifier tiene buen nivel de predicción, con un accuracy del 77,2%. Al revisar su desempeño por categoria vemos que presenta buen rendimiento para los dos tipos de pedidos, tanto los que cumplen como los que no cumplen. Se tiene presente que este tipo de modelo (arbol) tiende a un overfit sobre la data de entrenamiento, por lo que luego se realizará validación con una muestra de otro periodo.

### Modelo Logistic Regression


```python
%%time
log_model = LogisticRegression()
log_model.fit(X_train,y_train)
y_hat_log = log_model.predict(X_test)
```

    CPU times: user 3min 30s, sys: 2min 21s, total: 5min 51s
    Wall time: 30.6 s



```python
metrics_model(log_model,X_test,y_test)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.72</td>
      <td>0.73</td>
      <td>0.73</td>
      <td>0.72</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.49</td>
      <td>0.88</td>
      <td>0.73</td>
      <td>0.68</td>
      <td>0.73</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.58</td>
      <td>0.79</td>
      <td>0.73</td>
      <td>0.69</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>support</th>
      <td>333584.00</td>
      <td>514153.00</td>
      <td>0.73</td>
      <td>847737.00</td>
      <td>847737.00</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_137_1.png)



![png](output_137_2.png)



```python
#dump(log_model, "log_model.joblib")
```

A nivel general, el modelo de Regresion Logistica tiene buen nivel de predicción, con un accuracy del 72,5%. Sin embargo, al revisar su desempeño por categoría vemos que presenta un bajo rendimiento para los pedidos 0, es decir, los que no cumplen la entrega según su promesa, con un recall de 49% lo cual es bastante bajo al no superar el 50%.

### Gradient Boosting


```python
%%time
gboost_model = GradientBoostingClassifier()
gboost_model.fit(X_train, y_train)

y_hat_GBoost = gboost_model.predict(X_test)
```

    CPU times: user 9min 43s, sys: 1.23 s, total: 9min 44s
    Wall time: 9min 44s



```python
metrics_model(gboost_model,X_test,y_test)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.85</td>
      <td>0.75</td>
      <td>0.77</td>
      <td>0.80</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.51</td>
      <td>0.94</td>
      <td>0.77</td>
      <td>0.73</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.64</td>
      <td>0.83</td>
      <td>0.77</td>
      <td>0.74</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>support</th>
      <td>333584.00</td>
      <td>514153.00</td>
      <td>0.77</td>
      <td>847737.00</td>
      <td>847737.00</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_142_1.png)



![png](output_142_2.png)



```python
#dump(gboost_model, "gboost_model.joblib")
```

A nivel general, el modelo de GradientBoostClassifier tiene buen nivel de prediccion, con un accuracy del 77,3%. Sin embargo, al revisar su desempeño por categoría vemos que presenta un bajo rendimiento para los pedidos 0, es decir, los que no cumplen la entrega segun su promesa, con un recall de 51% lo cual es bastante bajo.

### Comparación de modelos

A nivel general el accuracy de todos los modelos se mueve entre el 72% y el 77%, lo cual es un nivel aceptable de desempeño del modelo, siendo el mejor el Gradient Boosting:

1. GradientBoosting: 77,3%
2. DecisionTree: 77,2%
3. AdaBoost: 75,4%
4. Logistic Regression: 72,5%

El objetivo del modelo es predecir el cumplimiento de la promesa de entrega que se le notifica al cliente, en este sentido nos es más relevante poder identificar de manera correcta los pedidos que no cumplirian con el tiempo comprometido, es decir los de clasificacion 0, por este motivo al revisar el indicador de desempeño del recall para los clase 0:

1. DecisionTree: 72%
2. AdaBoost: 54%
3. GradienBoosting: 51%
4. Logistic Regression: 49%

Se puede ver que el modelo de DecisionTree presenta el mejor desempeño por bastante rango versus el resto de los modelos, con un 72% de recall para las categorias 0. Tambien se validaron los modelo con una muestra de datos de otro periodo, el accuraccy y metricas de desmepeño bajaron en aprox 5 puntos porcentuales, pero la conclusion es la misma, el modelo de Decision Tree entrega los resultados mas acertados. (Para revisar validacion revisar notebook adjunto: 'Validación_Modelos_FAST.ipynb')<br>

Si revisamos los atributos más importantes para el modelo de Decision Tree podemos ver que:


```python
best_attr = plot_importance(dtc_model,X_train.columns)
```


![png](output_147_0.png)


El atributo más importante para el modelo son la distancia en metros, luego las horas de creación hasta la salida de la primera milla y las horas de creación hasta la fecha comprometida, es decir, cuanto tiempo de plazo se tiene para la entrega.

### Conclusiones

Se traen las categorías obtenidas en con el modelo de clusterizacion y se seleccionan los grupos, mayores a 10.000 pedidos que tengan una baja probabilidad de cumplimiento según el modelo de predicción generado.


```python
%%bigquery clusters
select *
from `charged-ground-301216.test_1_fast_project.cluster_persona`
```


```python
columnas = df_modelo.columns
cols = list(columnas)
cols.remove("cumplimiento")
yhat_dtc=dtc_model.predict(df_modelo.loc[:,cols])
clusters['pred']=yhat_dtc
grupos=clusters.groupby(['cluster', 'tipo_cliente','tipo_envio', 'tipo_induccion', 'tipo_tramo']).pred.agg(['count', 'mean'])
grupos[(grupos['count']>10000) & (grupos['mean']<0.6)].sort_values(by='count', ascending=False).sort_values(by=['count','mean'], ascending=False)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th>count</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>cluster</th>
      <th>tipo_cliente</th>
      <th>tipo_envio</th>
      <th>tipo_induccion</th>
      <th>tipo_tramo</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th rowspan="2" valign="top">empresa</th>
      <th rowspan="2" valign="top">a_la_puerta</th>
      <th rowspan="2" valign="top">retirado_a_cliente</th>
      <th>TRAMO EXTRA</th>
      <td>491351</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>TRAMO LARGO</th>
      <td>348908</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>1</th>
      <th>empresa</th>
      <th>a_la_puerta</th>
      <th>retirado_a_cliente</th>
      <th>TRAMO LARGO</th>
      <td>127446</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">2</th>
      <th rowspan="2" valign="top">empresa</th>
      <th rowspan="2" valign="top">retiro_ventanilla</th>
      <th rowspan="2" valign="top">retirado_a_cliente</th>
      <th>TRAMO EXTRA</th>
      <td>77000</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>TRAMO LARGO</th>
      <td>54109</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">persona</th>
      <th>retiro_ventanilla</th>
      <th>dejado_sucursal</th>
      <th>TRAMO EXTRA</th>
      <td>33081</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">a_la_puerta</th>
      <th rowspan="2" valign="top">dejado_sucursal</th>
      <th>TRAMO EXTRA</th>
      <td>32844</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>TRAMO LARGO</th>
      <td>30780</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>retiro_ventanilla</th>
      <th>dejado_sucursal</th>
      <th>TRAMO LARGO</th>
      <td>27571</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">1</th>
      <th>empresa</th>
      <th>retiro_ventanilla</th>
      <th>retirado_a_cliente</th>
      <th>TRAMO LARGO</th>
      <td>18423</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">persona</th>
      <th>retiro_ventanilla</th>
      <th>dejado_sucursal</th>
      <th>TRAMO MEDIO</th>
      <td>14279</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>a_la_puerta</th>
      <th>dejado_sucursal</th>
      <th>TRAMO LARGO</th>
      <td>13803</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>retiro_ventanilla</th>
      <th>dejado_sucursal</th>
      <th>TRAMO LARGO</th>
      <td>12020</td>
      <td>0.58</td>
    </tr>
  </tbody>
</table>
</div>



Se ve que existen 15 grupos principales al evaluar los peores grupos con un Q de pedidos que son representativos para la muestra, y principalmente se ve que:
* Todos los grupos pertenece al cluster-2: 'Los Patiperros' y cluster-1:'Los poco optimizados'.
* Pertenece a tramos clasificados como Largos o extras, esto se explica ya que en le modelo de predicción el atributo más importante es la distancia recorrida.
* La gran mayoría de pedidos caen en el segmento empresa, pero aun así hay gran cantidad de pedidos persona.



---
[Project 2 Title](/pdf/sample_presentation.pdf)
<img src="images/dummy_thumbnail.jpg?raw=true"/>

---
[Project 3 Title](http://example.com/)
<img src="images/dummy_thumbnail.jpg?raw=true"/>

---

### Category Name 2

- [Project 1 Title](http://example.com/)
- [Project 2 Title](http://example.com/)
- [Project 3 Title](http://example.com/)
- [Project 4 Title](http://example.com/)
- [Project 5 Title](http://example.com/)

---




---
<p style="font-size:11px">Page template forked from <a href="https://github.com/evanca/quick-portfolio">evanca</a></p>
<!-- Remove above link if you don't want to attibute -->
