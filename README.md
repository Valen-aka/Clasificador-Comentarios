# 📊 Clasificador de Comentarios
## 📝 Objetivo
### Analizar y clasificar automáticamente las quejas de los clientes de la empresa Comcast utilizando técnicas de procesamiento de lenguaje natural (NLP) y aprendizaje supervisado.
## 🧠 Alumnos participantes
### Huamán Cortez, Anabella Karina
### Montenegro López, Valentina Étoile
## 🗂️ Dataset
### El conjunto de datos seleccionado para este proyecto es el Comcast Telecom Complaints Dataset, el cual proviene de la plataforma Kaggle. Está compuesto por 2,224 registros distribuidos en 11 variables, que contienen información sobre quejas realizadas por clientes de una empresa de telecomunicaciones en Estados Unidos. 
### Para adaptar el dataset a los objetivos del presente proyecto, se realizó una modificación al conjunto de datos original, añadiendo una nueva variable denominada Category. Con esta incorporación, el dataset pasó de 11 a 12 variables.
## 📌 Conclusiones
### El proyecto implementó un sistema de clasificación de comentarios utilizando técnicas de procesamiento de lenguaje natural (limpieza, tokenización y TF-IDF) junto con modelos supervisados como Naive Bayes y Regresión Logística. Los resultados muestran un buen desempeño general, destacando la Regresión Logística con un accuracy de 0.95, mientras que Naive Bayes alcanzó 0.87. Aunque el modelo clasifica bien las categorías más frecuentes, presenta dificultades en clases con pocos datos, lo que evidencia la necesidad de mejorar el balance del dataset. Como trabajo futuro se considera ampliar datos minoritarios, aplicar técnicas de balanceo y explorar modelos más avanzados o embeddings semánticos para optimizar la precisión del sistema.
