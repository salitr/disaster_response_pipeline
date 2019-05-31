
# Disaster Response Pipeline
Building a web app that classifies disaster messages. The datasets used in bulding the Machine Learning Model are provided from Figure Eight.

## Libraries
The following libraties should be imported; pandas, numpy, os, matplotlib json, multiple sklearn models, plotly, sys, re, pickle, warnings, sqlalchemy, NLTK, subprocess, termcolor, joblib, and flask

## Motivation
 The dataset contains real messages that were sent during disaster events. The repository include a web app where an emergency worker can input a new message and get classification results in several categories, and hence, the message will be sent to an appropriate disaster relief agency. The web app displaies a data visualizations of the overall training data.

## File Descriptions
1. **data** Folder includes 4 files:
  * *disaster_messages.csv*: Dataset includes all the messages and genres.
  * *disaster_categories.csv*: Dataset includes all the categories.
  * *process_data.py*: Code used to transforme and clean the data and then to create a SQLite database.
  * *DisasterResponse.db*: SQLite database contains the transformed and cleaned data.


2. **models** Folder includes 2 files:
  * *train_classifier.py*: Machine Learning Model to train and export a classifier as a pickle file.
  * *classifier.pkl*: Final model as a pickle file.


3. **app** Folder includes 1 files:
  * *run.py*: Flask file to run the web app.

## Instructions
1. **process_data.py**:
  * *To process and run the ETL pipeline that transforms and cleans the data and the creates a SQLite database*:
  ```
  python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
  ```


2. **train_classifier.py**:
  * *To train and run the ML pipeline that train and export a classifier as a pickle file*:
  ```
  python train_classifier.py ../data/DisasterResponse.db classifier.pkl
  ```

3. **run.py**:
  * *To run the web application that allows for adding new messages and then getting classification results in several categories*:
  ```
  python run.py
  ```
  * Go to http://0.0.0.0:3001/

![Screenshot 1](https://github.com/salitr/disaster_response_pipeline/blob/master/Screen%20Shots/Screen%20Shot%201.png)
![Screenshot 2](https://github.com/salitr/disaster_response_pipeline/blob/master/Screen%20Shots/Screen%20Shot%202.png)
![Screenshot 3](https://github.com/salitr/disaster_response_pipeline/blob/master/Screen%20Shots/Screen%20Shot%203.png)
![Screenshot 4](https://github.com/salitr/disaster_response_pipeline/blob/master/Screen%20Shots/Screen%20Shot%204.png)


## Licensing and Acknowledgement
  * The datasets used were provided by Figure Eight.
  * Code templates were provided by Udacity as a part of the Udacity Data Scientist Nanodegree.
  * Plotly provides examples of codes for different types of plots of which some have been adapted, edited, and used as needed.
