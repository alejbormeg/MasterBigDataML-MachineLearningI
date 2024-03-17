# Tarea 1: Machine Learning

Autor: Alejandro Borrego Megías  
Correo: alejbormeg@gmail.com

# Índice
0. [Introducción](#introducción)
1. [Análisis Exploratorio y Limpieza de Datos](#análisis-exploratorio-y-limpieza-de-datos)
2. [Mejor Modelo de Regresión Logística](#mejor-modelo-de-regresión-logística)
   - [Justificación de Parametrizaciones](#justificación-de-parametrizaciones)
   - [Mejor Red Neuronal Basada en Regresión Logística](#mejor-red-neuronal-basada-en-regresión-logística)
3. [Selección de Variables y Mejor Red Neuronal en Términos de AUC](#selección-de-variables-y-mejor-red-neuronal-en-términos-de-auc)
   - [Justificación de Parametrizaciones](#justificación-de-parametrizaciones-1)
4. [Comparación de Modelos](#comparación-de-modelos)
5. [Mejor Búsqueda Paramétrica para Árboles de Decisión](#mejor-búsqueda-paramétrica-para-árboles-de-decisión)
   - [Representación Gráfica del Árbol Ganador](#representación-gráfica-del-árbol-ganador)
   - [Importancia de las Variables](#importancia-de-las-variables)
6. [Mejor Modelo de Bagging y Random Forest](#mejor-modelo-de-bagging-y-random-forest)
7. [Mejor Modelo de Gradiente Boosting y XGBoost](#mejor-modelo-de-gradiente-boosting-y-xgboost)
8. [Mejor Modelo de SVM con Diferentes Kernels](#mejor-modelo-de-svm-con-diferentes-kernels)
9. [Método de Ensamblado de Bagging](#método-de-ensamblado-de-bagging)
10. [Método de Stacking](#método-de-stacking)
11. [Conclusiones](#conclusiones)

## Introducción

En este informe, exploraremos el conjunto de datos proporcionado con el objetivo de clasificar en la categorías "presenta obseidad" (1) "no presenta obesidad" (0). Para este análisis utilizaremos un conjunto de datos que incluye variables como género, edad, altura, peso, historial familiar de sobrepeso, hábitos alimenticios y de actividad física, entre otros, para desarrollar modelos predictivos precisos.

Comenzaremos con un análisis exploratorio y limpieza de los datos para comprender mejor su estructura y calidad, abordando cualquier anomalía, valor faltante o atípico que pueda afectar la precisión de nuestros modelos. Este paso es fundamental para garantizar la integridad de nuestros análisis posteriores. Además las técnicas que se usarán en los siguientes apartados precisan de este análisis previo en su mayoría, por ello para no repetirnos lo realizaremos una sola vez.

Posteriormente, buscaremos el mejor modelo de regresión logística, analizando las variables más influyentes y, a partir de estas, desarrollaremos una red neuronal optimizada en términos de precisión. Este enfoque nos permitirá explorar la complejidad de las relaciones entre variables y su impacto en la clasificación de la obesidad.

Además, aplicaremos técnicas de selección de variables para identificar los predictores más significativos y, con ellos, entrenar un modelo de red neuronal enfocado en maximizar el Área Bajo la Curva (AUC) de la característica operativa del receptor (ROC), crucial para evaluar el rendimiento en tareas de clasificación.

Compararemos estos modelos para identificar el más adecuado para nuestra tarea. También realizaremos búsquedas paramétricas exhaustivas para optimizar modelos de árboles de decisión, bagging, random forest, gradiente boosting, XGBoost y máquinas de vectores de soporte (SVM) con diferentes kernels, buscando siempre la máxima precisión.

Exploraremos métodos de ensamblaje avanzados como Bagging con clasificadores base no tradicionales y Stacking, integrando múltiples algoritmos para mejorar la robustez y precisión de nuestras predicciones.

Este análisis nos permitirá no solo identificar el mejor modelo para clasificar la obesidad basándonos en características individuales, sino también proporcionar insights valiosos sobre la relación entre el estilo de vida, características personales y el riesgo de obesidad. La precisión y eficacia de nuestros modelos tienen el potencial de informar intervenciones dirigidas y políticas de salud pública para combatir la obesidad a nivel individual y comunitario.

## Análisis Exploratorio y Limpieza de Datos

(Descripción del análisis exploratorio de los datos, incluyendo estadísticas descriptivas, visualizaciones para comprender la distribución de los datos y la identificación de valores atípicos o faltantes. Detalle del proceso de limpieza de datos, incluyendo la imputación de valores faltantes, la eliminación de valores atípicos y cualquier transformación de variables realizada).

Los datos que usaremos se encuentran en el fichero *src\data\datos_practica_miss.csv*. Para leerlos usaremos la librería *pandas* de Python para cargarlos como un dataframe, tras esto usaremos la semilla *4579* (últimos dígitos de mi DNI) para tomar una muestra aleatoria de 1000 filas y realizamos una primera exploración de los datos con la función *info*:

```py
data = pd.read_csv("C:\\Users\\pablo\\OneDrive\\Documentos\\GitHub\\MasterBigDataML-MachineLearningI\\src\\data\\datos_practica_miss.csv")

data = data.sample(n=1000, random_state=4975)
data.info()
```

```bash
<class 'pandas.core.frame.DataFrame'>
Index: 1000 entries, 371 to 1382
Data columns (total 18 columns):
 #   Column                          Non-Null Count  Dtype  
---  ------                          --------------  -----  
 0   Unnamed: 0                      1000 non-null   int64  
 1   Gender                          983 non-null    object 
 2   Age                             977 non-null    float64
 3   Height                          975 non-null    float64
 4   Weight                          977 non-null    float64
 5   family_history_with_overweight  976 non-null    object 
 6   FAVC                            975 non-null    object 
 7   FCVC                            977 non-null    float64
 8   NCP                             973 non-null    float64
 9   CAEC                            975 non-null    object 
 10  SMOKE                           984 non-null    object 
 11  CH2O                            980 non-null    float64
 12  SCC                             981 non-null    object 
 13  FAF                             973 non-null    float64
 14  TUE                             981 non-null    float64
 15  CALC                            981 non-null    object 
 16  MTRANS                          980 non-null    object 
 17  NObeyesdad                      980 non-null    object 
dtypes: float64(8), int64(1), object(9)
```
Dentro del contexto del estudio sobre obesidad, cada variable recogida en los datos juega un papel importante en el análisis. A continuación, se explica el significado de cada una de estas variables:

1. Gender: Esta variable es de tipo objeto y representa el género del individuo, con 983 valores no nulos registrados. Los géneros típicamente se clasifican como masculino o femenino, pero el diseño del estudio podría incluir otras categorías según cómo se recolectaron los datos.

2. Age: Representa la edad de los individuos, con 977 valores no nulos. Se registra como un número de punto flotante, lo que permite incluir edades exactas hasta cierto nivel decimal (aunque usualmente la edad se maneja en números enteros).

3. Height: La altura de los individuos, registrada para 975 sujetos como un número de punto flotante. Esta medida es crucial para evaluar el Índice de Masa Corporal (IMC) junto con el peso.

4. Weight: El peso de los individuos, disponible para 977 sujetos, también como un número de punto flotante. Es uno de los factores clave para determinar el estado nutricional y el riesgo de obesidad.

5- family_history_with_overweight: Indica si el individuo tiene antecedentes familiares de sobrepeso, con 976 respuestas. Es una variable de tipo objeto, probablemente con respuestas como "sí" o "no", que ayuda a entender la predisposición genética o ambiental hacia la obesidad.

6. FAVC: "Frecuencia de Consumo de Alimentos Altos en Calorías" (High-Calorie Food Consumption Frequency), registrada para 975 individuos como objeto, indica si el individuo consume frecuentemente alimentos altos en calorías, un factor importante en el riesgo de obesidad.

7. FCVC: "Frecuencia de Consumo de Verduras", con 977 valores no nulos, mide la regularidad con la que el individuo consume verduras, contribuyendo a evaluar la calidad de la dieta.

8. NCP: "Número de Comidas Principales", con datos para 973 individuos, registra cuántas comidas principales hace la persona al día, importante para entender los patrones alimenticios.

9. CAEC: "Consumo de Alimentos Entre Comidas", con 975 valores no nulos, indica la frecuencia con la que la persona consume snacks o alimentos fuera de las comidas principales, lo cual puede afectar el balance calórico total.

10. SMOKE: Indica si el individuo fuma, con datos para 984 personas. Fumar puede influir en el metabolismo y en el apetito, afectando indirectamente el peso.

11. CH2O: La cantidad de agua consumida diariamente, registrada para 980 sujetos. Una adecuada hidratación puede influir en la digestión, el metabolismo y la sensación de saciedad.

12. SCC: "Monitoreo de Calorías Consumidas", con 981 valores, indica si el individuo lleva un control sobre las calorías que consume, lo cual puede ser indicativo de una mayor conciencia y gestión del peso.

13. FAF: "Frecuencia de Actividad Física", registrada para 973 individuos, refleja cuán a menudo la persona realiza ejercicio, crucial para el manejo del peso y la salud general.

14. TUE: "Tiempo Usado en Dispositivos Electrónicos", con datos para 981 individuos, puede indicar niveles de sedentarismo, afectando el balance energético y el riesgo de obesidad.

15. CALC: "Consumo de Alcohol", con 981 respuestas, refleja la frecuencia con la que el individuo consume bebidas alcohólicas, las cuales pueden contribuir a un exceso de calorías.

16. MTRANS: "Medio de Transporte", registrado para 980 individuos, indica el método principal de movilidad del individuo (por ejemplo, caminar, transporte público, automóvil), lo que puede influir en el nivel de actividad física diaria.

Cada una de estas variables ofrece una vista comprensiva sobre diferentes aspectos que pueden influir en el peso y la salud de una persona, permitiendo un análisis detallado sobre los factores de riesgo y las potenciales intervenciones para prevenir o manejar la obesidad.

A simple vista destaca la presencia de valores perdidos en la mayoría de columnas a excepción de la columna "Unnamed: 0", que eliminaremos por no conocer la relación de dicha columna con el problema.

Para conocer el significado de las siguientes categorías se ha realizado una búsqueda por internet llegando a las siguientes conclusiones:



### Valores atípicos
En un primer lugar veremos la presencia de valores atípicos en los datos y si se deben tratar como datos perdidos.

Realizamos un *Boxplot* por cada columna numérica:

![Numericas](./img/boxplot_numericas.png)

Vemos como hay valores atípicos en la columna de la edad (los cuales no podemos considerar como valores perdidos, pues debe de tratarse de la edad real de ciertas personas encuestadas, que en su mayoría son jóvenes de entre 20 y 27 años). Lo mismo haremos con los datos atípicos de las columnas de la altura, peso y NCP pues se tratan de valores dentro de la normalidad, aunque atípicos en este caso.


### Valores perdidos
Contamos el total de valores perdidos por columna:

```py
data.isna().sum()
```

```bash
Unnamed: 0                         0
Gender                            17
Age                               23
Height                            25
Weight                            23
family_history_with_overweight    24
FAVC                              25
FCVC                              23
NCP                               27
CAEC                              25
SMOKE                             16
CH2O                              20
SCC                               19
FAF                               27
TUE                               19
CALC                              19
MTRANS                            20
NObeyesdad                        20
```

Comprobamos también que en este caso no tenemos filas duplicadas.

Vemos como en ningún caso tenemos una cantidad extrema de valores perdidos como para considerar eliminar alguna columna del estudio.

Vamos a visualizar estos datos perdidos con ayuda de la librería *missingno* de python, para ver si la mayoría de valores perdidos caen juntos en la mismas filas, y considerar eliminarlas:

![Missingno](./img/missingno.png)

En primer lugar eliminamos los valores perdidos de la variable objetivo "NObeyesdad". Pues no tiene sentido imputarlos.

La distribución de valores eprdidos es muy aleatoria dentro de cada columna, por ello optamos por imputar los valores perdidos, sustituyendo las variables numéricas por la media o mediana (dependiente de si son o no simétricas) y las categóricas por la moda. Eliminamos también la primera columna por ser irrelevante para el estudio:

```py
data.drop('Unnamed: 0', axis=1, inplace=True)
data = data.dropna(subset=['NObeyesdad'])


for column in numerical_columns:
    if abs(skew(data[column].dropna())) > 1:
        data[column].fillna(data[column].median(), inplace=True)
    else:
        data[column].fillna(data[column].mean(), inplace=True)


for column in categorical_columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

data.isnull().sum()
```
Con esto comprobamos cómo hemos eliminado todos los valores perdidos: 

```bash
Gender                            0
Age                               0
Height                            0
Weight                            0
family_history_with_overweight    0
FAVC                              0
FCVC                              0
NCP                               0
CAEC                              0
SMOKE                             0
CH2O                              0
SCC                               0
FAF                               0
TUE                               0
CALC                              0
MTRANS                            0
NObeyesdad                        0
```

### Transformación en variable objetivo
La variable objetivo del problema es la columna "NObeyesdad", que como vemos presenta diferentes categorías:
* Obesity_Type_I
* Obesity_Type_II
* Obesity_Type_III
* Overweight_Level_I
* Overweight_Level_II
* Normal_Weight
* Insufficient_Weight

Dado que el problema que pretendemos resolver es un problema de clasificación **Binaria** debemos cambiar esta columna por valores numéricos 0 (si no presenta obesidad) y 1 (si presenta obesidad). A continuación mostramos la distribución de las diferentes etiquetas:

![](./img/categorias.png)

Como vemos todas ellas presentan una cantidad similar de valores, el mappeo que realizaremos en la columna es el siguiente:

```py
obesity_mapping = {
    'Obesity_Type_I': 1,
    'Obesity_Type_II': 1,
    'Obesity_Type_III': 1,
    'Overweight_Level_I': 0,
    'Overweight_Level_II': 0,
    'Normal_Weight': 0,
    'Insufficient_Weight': 0
}

data['NObeyesdad'] = data['NObeyesdad'].map(obesity_mapping)
data['NObeyesdad'].value_counts()
```

Obteniendo un total de 535 no obesos y 445 con obesidad. En futuras secciones trataremos de obtener una representación similar de valores positivos y negativos en las particiones que usemos para entrenamiento y test.

### Eliminación de variables
Finalmente, de cara a los modelos lineales estudiaremos la correlación entre las variables numéricas para comprobar si se debería eliminar alguna:

```py
matriz_corr = correlation.corr(method = 'pearson')
mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', linewidths=0.5, mask=mask)
sns.set(font_scale=1.2)
plt.title("Matriz de correlación")
plt.show()
```
![](./img/correlation_matrix.png)


## Mejor Modelo de Regresión Logística

Utilizando la librería `sklearn`, implementamos un modelo de Regresión Logística optimizado con un exhaustivo proceso de selección de hiperparámetros a través de `GridSearchCV`. Este modelo es ideal para problemas de clasificación binaria, donde nuestro objetivo es determinar la presencia o ausencia de obesidad en individuos.

### Preprocesamiento:

El paso crítico antes del entrenamiento fue el preprocesamiento de datos, esencial para adecuar nuestras variables a los modelos de aprendizaje automático. Empleamos `ColumnTransformer` para:

- **Variables Numéricas**: Escaladas con `MinMaxScaler` para normalizar cada característica en el rango [0, 1], facilitando el proceso de aprendizaje y optimización.
  
- **Variables Categóricas**: Convertidas a formato numérico mediante `OneHotEncoder`, permitiendo que los modelos matemáticos procesen eficientemente etiquetas categóricas.

### Definición del Modelo y Selección de Características:

Definimos `logistic_regression` como nuestro modelo base y preparamos un conjunto de datos con características seleccionadas usando `SequentialFeatureSelector` de `mlxtend`, optando por un enfoque stepwise que considera tanto la adición como la eliminación de características:

```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
logistic_regression = LogisticRegression(max_iter=5000)

# Configuramos la selección secuencial stepwise
sfs = SFS(estimator=logistic_regression, 
          k_features='best', 
          forward=True, 
          floating=True,  # Habilita la selección stepwise
          scoring='accuracy',
          cv=StratifiedKFold(5))

# Entrenamos el modelo con los datos preprocesados
sfs.fit(X_train_preprocessed, y_train)
```
Con este método realizamos la selección de variables para el entrenamiento del modelo.

### Búsqueda en Malla para Optimización de Hiperparámetros:

Con el objetivo de afinar el modelo, definimos un param_grid amplio para GridSearchCV:

```py
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l2', 'l1', 'elasticnet', 'none'],
    'class_weight': [None, 'balanced'],
    'max_iter': [100, 1000, 5000, 10000]
}
```

Dónde:
    
* **'C'**: Es el parámetro de inversión de la fuerza de regularización. Valores más bajos especifican una regularización más fuerte, lo que puede prevenir el sobreajuste reduciendo la complejidad del modelo, pero también puede causar un subajuste al no permitir que el modelo se ajuste suficientemente a los datos. El rango va desde 0.001 a 1000, lo que permite explorar desde una regularización muy fuerte a una muy débil.

* **'solver'**: Especifica el algoritmo a usar en el problema de optimización. Cada solucionador tiene sus propias características y es adecuado para tipos específicos de datos:

    * 'newton-cg', 'lbfgs', 'sag' y 'saga' son buenos para datasets grandes o cuando se requiere regularización L2, aunque los exploraremos igualmente.

    * 'liblinear' es una buena elección para datasets pequeños y soporta la regularización L1.

    * 'saga' también soporta la regularización "elasticnet" y es la única opción para este tipo de regularización.

* **'penalty'**: Se refiere al tipo de penalización o regularización a aplicar. La regularización ayuda a prevenir el sobreajuste y mejora la generalización del modelo. Las opciones son:
    * 'l2' es la penalización estándar que se aplica por defecto.
    * 'l1' es útil para generar modelos más simples o cuando se sospecha que algunas características pueden ser irrelevantes.
    * 'elasticnet' es una combinación de L1 y L2, proporcionando un equilibrio entre generar un modelo simple (L1) y mantener la regularización de modelos más complejos (L2).
    * 'none' indica que no se aplica ninguna regularización.

* **'class_weight'**: Este parámetro se usa para manejar desequilibrios en las clases. Al ajustar este parámetro, se puede influir en la importancia que el modelo da a cada clase durante el entrenamiento:
    * None significa que todas las clases tienen el mismo peso.
    * 'balanced' ajusta automáticamente los pesos inversamente proporcionales a las frecuencias de clase en los datos de entrada, lo que puede ser útil para datasets desequilibrados.

* **'max_iter'**: Especifica el número máximo de iteraciones tomadas para que los solucionadores converjan a una solución. Un número más alto de iteraciones permite más tiempo para que el modelo encuentre una solución óptima, pero también aumenta el tiempo de computación. Se exploran varios valores para asegurar que el solucionador converge adecuadamente para diferentes complejidades de modelos.

```py
# Reducimos los conjuntos a las características seleccionadas y aplicamos GridSearch
X_train_selected = X_train_preprocessed[:, sfs.k_feature_idx_]
X_test_selected = X_test_preprocessed[:, sfs.k_feature_idx_]
grid_search = GridSearchCV(estimator=logistic_regression, param_grid=param_grid, cv=StratifiedKFold(5), scoring='accuracy')
grid_search.fit(X_train_selected, y_train)
```

Los parámetros óptimos obtenidos fueron `{'C': 1, 'class_weight': None, 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'}` y la precisión alcanzada con el modelo ajustado fue de 0.98 aproximadamente, demostrando una excelente capacidad predictiva. Las características finales seleccionadas reflejan aspectos importantes tanto biológicos como relacionados con el estilo de vida que afectan la probabilidad de obesidad.

La matriz de confusión obtenida es la siguiente:
![](./img/confusion_matrix_logistic_reg.png)

### Mejor Red Neuronal Basada en Regresión Logística

Para construir la mejor red neuronal basada en las características seleccionadas por el modelo de regresión logística, utilizamos un enfoque sistemático para explorar una variedad de configuraciones de red neuronal. Este proceso se llevó a cabo mediante la implementación de *MLPClassifier* de Scikit-learn, junto con *GridSearchCV* para una búsqueda exhaustiva de hiperparámetros óptimos, garantizando así la selección de la configuración más adecuada para nuestro conjunto de datos.

La elección de características se basó en las seleccionadas por SequentialFeatureSelector, que identificó las variables más relevantes de acuerdo con su impacto en la precisión de la regresión logística. Estas características se utilizaron como entrada para entrenar la red neuronal, asegurando que el modelo se centrara solo en los predictores más significativos.

El param_grid_nn definido para GridSearchCV incluyó una gama de opciones para la arquitectura de la red y sus parámetros:

* **'hidden_layer_sizes'**: Exploramos varias configuraciones de capas ocultas, desde una sola capa de 50 o 100 neuronas hasta dos capas de 50 o 100 neuronas cada una. Esta variedad permite evaluar cómo la profundidad y la complejidad de la red afectan su rendimiento.

* **'activation'**: Se probó con funciones de activación 'tanh' y 'relu', para determinar cuál facilita mejor la convergencia y la capacidad de generalización del modelo.

* **'solver'**: Incluimos 'sgd' para el descenso de gradiente estocástico y 'adam', un optimizador basado en gradientes estocásticos más sofisticado, para identificar el algoritmo de optimización más efectivo.

* **'alpha'**: Se varió el término de regularización para prevenir el sobreajuste, evaluando desde una regularización fuerte a una más ligera.

* **'learning_rate'**: Se consideraron estrategias 'constant' y 'adaptive', esta última ajusta la tasa de aprendizaje a lo largo del entrenamiento para mejorar la eficiencia y la convergencia.

Tras el proceso de entrenamiento y validación mediante GridSearchCV, el modelo resultante alcanzó una precisión de 0.9694, un indicador de su excelente capacidad predictiva. Los mejores parámetros identificados para la red neuronal fueron:

* Activación: 'tanh', que sugiere que esta función de activación funciona bien con nuestro conjunto de datos, posiblemente ayudando a evitar problemas de gradientes que desaparecen en comparación con 'relu' en este contexto específico.

* Regularización alpha: 0.001, proporcionando un balance óptimo entre aprendizaje y prevención del sobreajuste.

* Tamaño de las capas ocultas: (50, 50), lo que indica que una red de dos capas con 50 neuronas cada una es suficiente para capturar la complejidad de nuestros datos sin incurrir en sobreajuste.

* Tasa de aprendizaje: 'constant', demostrando que mantener una tasa de aprendizaje constante a lo largo del entrenamiento es efectivo para este modelo.

* Solucionador: 'adam', confirmando que este optimizador es adecuado para nuestros datos, probablemente debido a su eficiencia en conjuntos de datos relativamente pequeños y su capacidad para manejar bien los mínimos locales.

Este enfoque metodológico y la selección cuidadosa de hiperparámetros han permitido desarrollar un modelo de red neuronal robusto y preciso, basado en los predictores clave identificados a través de la regresión logística, para predecir la obesidad en individuos.

La matriz de confusión obtenida es la siguiente:

![](./img/confusion_matrix_cnn_1.png)

## Selección de Variables y Mejor Red Neuronal en Términos de AUC

### Justificación de Parametrizaciones

(Descripción del proceso de selección de variables con k=4 y cómo se encontró el mejor modelo de red neuronal en términos de AUC, incluyendo las parametrizaciones probadas y los parámetros escogidos).

## Comparación de Modelos

(Comparación de los modelos candidatos a ganadores de las secciones 1) y 2), evaluando sus rendimientos y características distintivas).

## Mejor Búsqueda Paramétrica para Árboles de Decisión

(Descripción de la búsqueda paramétrica realizada para encontrar el mejor árbol de decisión, incluyendo los cuatro métodos de validación de la bondad de la clasificación utilizados).

### Representación Gráfica del Árbol Ganador

(Representación gráfica del árbol ganador, incluyendo sus reglas en formato texto).

### Importancia de las Variables

(Gráfica que muestra la importancia de las variables en el árbol ganador).

## Mejor Modelo de Bagging y Random Forest

(Descripción de la búsqueda paramétrica para encontrar el mejor modelo de Bagging y Random Forest según Accuracy, incluyendo justificaciones de las parametrizaciones y los parámetros escogidos).

## Mejor Modelo de Gradiente Boosting y XGBoost

(Descripción de la búsqueda paramétrica para encontrar el mejor modelo de Gradiente Boosting y XGBoost según Accuracy, incluyendo justificaciones de las parametrizaciones y los parámetros escogidos).

## Mejor Modelo de SVM con Diferentes Kernels

(Descripción de la búsqueda paramétrica para determinar el mejor modelo de SVM con al menos dos kernels diferentes, incluyendo justificaciones de las parametrizaciones y los parámetros escogidos).

## Método de Ensamblado de Bagging

(Descripción del método de ensamblado de Bagging utilizado, con un clasificador base que no sea un árbol, incluyendo justificaciones de las parametrizaciones y los parámetros escogidos).

## Método de Stacking

(Descripción del método de Stacking escogido, incluyendo los algoritmos de entrada y el modelo utilizado como ensamblaje, junto con las justificaciones de las parametrizaciones y los parámetros escogidos).

## Conclusiones

(Resumen de los hallazgos más importantes, recomendaciones y posibles pasos a seguir en investigaciones futuras).
