# Network anomaly detection

## Metodología original

![figures/final_base_flux.png](figures/final_base_flux.png)

## Metodología modificada para mejorar rendimiento

![figures/final_flux_wmods.png](figures/final_flux_wmods.png)

## Instalación
Además de clonar el respositorio, es necesario por un lado descargar el _dataset_, y por el otro instalar los paquetes requeridos para poder ejecutar los _scripts_. 
Con respecto al _dataset_, podemos usar la siguiente secuencia de comandos desde dentro del directorio raíz del repositorio:

### Descarga del dataset en sistemas Linux:
```bash
cd data
wget -O dataset_dist.csv "https://udcgal-my.sharepoint.com/:x:/g/personal/julio_jairo_estevez_pereira_udc_es/EQWdSJToODxJhYe9gaWyV0MBWI19AxHXivB1y4-DrP4Myg?e=9fzMs6&download=1"
````
### Descarga del dataset en sistemas Windows (Powershell):
```bash
cd data
wget "https://udcgal-my.sharepoint.com/:x:/g/personal/julio_jairo_estevez_pereira_udc_es/EQWdSJToODxJhYe9gaWyV0MBWI19AxHXivB1y4-DrP4Myg?e=9fzMs6&download=1" -OutFile dataset_dist.csv
````

En cuanto a la instalación de paquetes a partir del archivo requirements.txt, se puede llevar a cabo mediante la ejecución del siguiente comando desde la raíz del repositorio:
```bash
pip3 install -r requirements.txt 
````

## Ejecución
### Modificaciones previas para ejecución en Windows:
Si se quiere realizar la ejecución en Windows es necesario hacer una pequeña modificación en el script, localizado en `/src/python/rf_loading_outliers.py`. La razón es que se emplea una librería sólo disponible en Linux para obtener la cantidad de paralelismo adecuada (en Windows no se ha implementado paralelismo alguno, por errores relacionados con scikit). Las modificaciones que habría que hacer son las siguientes:

* Eliminar el import de la librería `from os import sched_getaffinity`:
```python
# from os import sched_getaffinity
````
* Cambiar el valor de la variable global PARALLELIZATION a 1:
```python
PARALLELIZATION = 1
````

### Modificaciones previas para ejecución en Linux:
No son necesarias modificaciones para la ejecución en Linux.

### Ejecución del script:
```bash
cd src/python
python rf_loading_outliers.py
````

Durante la ejecución del script, se irá tanto mostrando por pantalla como guardando en un archivo (bajo el directorio `/logs`) los logs de la ejecución. Al terminar, se creará un directorio con los resultados en archivos JSON dentro del directorio `/results`.

## Estado base y modificación de los parámetros
Los parámetros con los que está establecido ahora mismo el script son de prueba, y están pensados para ejecuciones rápidas que muestren la funcionalidad. A continuación se indican los parámetros fundamentales que se pueden modificar, su localización y valor por defecto:

* `GLOBAL_ITS`: controla las iteraciones del "Repeat" de la metodología que se puede observar al principio de este documento. Son repeticiones de la validación cruzada anidada cuyo objetivo es comprobar cómo afecta al experimento el uso de diferentes semillas. Por defecto está establecido a 2.
* `SPLITS_OUTER_CV`: controla las divisiones del conjunto de datos que se hacen en la parte del "External CV" en el diagrama de la metodología. Esta validación cruzada se hace para asegurar la estabilidad de los resultados del experimento cuando los datos en los conjuntos de entrenamiento y test son diferentes. Por defecto la validación cruzada externa divide los datos en 2.
* `SPLITS_INNER_CV`: controla las divisiones del conjunto de entrenamiento que se hacen en la parte del "GSCV" para la búsqueda del mejor conjunto de parámetros. Por defecto está establecido a 3.
* `PARALLELIZATION`: controla el grado de paralelismo que se permite en las funciones de scikit que aceptan argumentos relacionados (por ejemplo RandomForestClassifier). Por defecto está configurado para que en entornos Linux averigue el número adecuado de paralelismo a utilizar, pero se puede modificar a mano. En Windows sólo se ha ejecutado con el valor 1 (valores mayores dan problemas).
* `PARAMS_NUM_ESTIMATORS`: contiene una lista de los números de árboles que se probarán en el GSCV. Por defecto prueba 800, y 1000 árboles.
* `PARAMS_CRITERION`: contiene la función a utilizar para medir la calidad de las divisiones en los árboles de RandomForestClassifier. Por defecto está establecido a la función Gini.
* `PARAMS_MAX_FEAUTURES`: contiene el número de variables que se podrán considerar en RandomForestClassifier a la hora de tomar una decisión para dividir los árboles. Por defecto está establecido a 6.

Nota: además de los anterior parámetros, es relevante mencionar que para que el experimento se ejecute rápido, en este momento sólo se usan las 200 000 primeras filas del dataset (que tiene más de 1,8 millones). Para cambiar esto y trabajar con todos los datos se puede buscar la línea `X = pd.read_csv(DATASET_PATH).head(200000)` y cambiarla por `X = pd.read_csv(DATASET_PATH)`.
