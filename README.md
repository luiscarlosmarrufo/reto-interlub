# Reto Interlub — Sistema de Recomendación de Grasas Lubricantes

Este repositorio contiene el desarrollo del proyecto realizado para Interlub. El objetivo fue analizar un conjunto de grasas lubricantes y construir un sistema de recomendación que permitiera identificar productos similares según sus propiedades técnicas y descripciones textuales.

Nota importante: todo el contenido relevante se encuentra en la carpeta `notebooks/FINAL/`, incluyendo:

- `Recomendador_grasas_completo.py`: código completo del recomendador implementado en Streamlit.
- `Regresion.ipynb`: notebook con el análisis y el modelo de regresión lineal múltiple.

---

## Descripción general del recomendador

El sistema permite buscar grasas mediante dos enfoques principales:

### 1. Búsqueda por similitud
El usuario puede seleccionar una grasa existente o definir una grasa personalizada.  
El modelo calcula similitud utilizando:

- Representación textual TF–IDF.
- Variables numéricas técnico–fisicoquímicas estandarizadas.
- Una métrica de similitud basada en cosine similarity.
- Un modo híbrido que combina similitud en texto y en variables numéricas.

Este enfoque permite identificar las grasas más parecidas dentro del catálogo, ya sea por características técnicas o por descripción.

### 2. Búsqueda por filtros técnicos
El usuario puede filtrar productos por parámetros como:

- Grado NLGI  
- Temperatura mínima y máxima de operación  
- Carga Timken mínima  
- Registro NSF  

Este modo está orientado a usuarios con requisitos operativos específicos.

---

## Modelo de regresión lineal

Se desarrolló un modelo de regresión lineal múltiple para analizar relaciones entre variables como viscosidad, penetración, punto de gota, carga Timken y temperaturas de servicio.  

Aunque el modelo mostró patrones interesantes, su precisión no fue suficiente para integrarlo en el recomendador final.  
Su propósito principal fue apoyar el análisis exploratorio y entender mejor la estructura del conjunto de datos.