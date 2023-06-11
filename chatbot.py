import pandas as pd
import numpy as np
import tensorflow as tf
import pyttsx3
import csv
import warnings
import re
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore', category=DeprecationWarning)

#symptoms = pd.read_csv('dataprocessed/symptoms.csv')
description = pd.read_csv('dataprocessed/description.csv')
precaution = pd.read_csv('dataprocessed/precaution.csv')
severity = pd.read_csv('dataprocessed/severity.csv')

description_list = dict()
precaution_list = dict()
severity_list = dict()

def getDescription():
    global description_list
    for index,row in description.iterrows():
        _desc = {row[0]:[row['Disease'],row['Description']]}
        description_list.update(_desc)

def getPrecaution():
    global precaution_list
    for index,row in precaution.iterrows():
        _pre = {row[0]: [row['Precaution_1'], row['Precaution_2'], row['Precaution_3'], row['Precaution_4']]}
        precaution_list.update(_pre)

def getSeverity():
    global severity_list
    for index,row in severity.iterrows():
        _sev = {row[0]:[row['Symptom'],row['weight']]}
        severity_list.update(_sev)

def getInfo():
    print('-----------------------------------HeRe Self Diagnosis-----------------------------------')
    print('\nYour Name? \t\t\t\t',end='->')
    name = input("")
    print("Hello, ",name)

def predict(pred_input):
    model = load_model('model/model.h5')
    preds_output = model.predict(pred_input)
    preds_output = np.argmax(preds_output, axis=1).item()
    return preds_output

def print_severity():
    print("{:<10} {:<10}".format('ID', 'Symptoms'))
    for key, value in severity_list.items():
        symptoms, weight = value
        id = key
        print("{:<10} {:<10}".format(id, symptoms))

def ask_symptoms():
    symptoms_input=[0 for i in range(17)]
    print_severity()
    print('\nWhat symptoms are you experiencing?')
    for i in range(17):
        while True : 
            symptom_id = input('Please type symptom id! (valid id : 0-132) \t\t->')
            try:
                if 0 <= int(symptom_id) <= 132 :
                    print('Valid input!')
                    break
                else:
                    print('Input is not within the range of 0 to 132. Please try again.')
            except ValueError:
                print('Invalid input. Please enter an integer.')
        while True:
            next_input = input('Enter \'p\' for predict disease or \'d\' for input symptom again : ')
            if next_input == 'p':
                print('Valid input: \'p\'')
                break
            elif next_input == 'd':
                print('Valid input: \'d\'')
                break
            else:
                print('Invalid input. Please try again.')
        symptoms_input[i] = int(severity_list[int(symptom_id)][1])
        if next_input=='p':
            break
        else: 
            continue
    return symptoms_input
    


def chatbot():
    getSeverity()
    getPrecaution()
    getDescription()
    getInfo()
    symptoms_input = ask_symptoms()
    preds_input = np.array([symptoms_input])
    preds_output = predict(preds_input)
    print('Disease Prediction')
    print(description_list[preds_output][0])
    print('Description:')
    print(description_list[preds_output][1])
    print('Precautions:')
    precution_list=precaution_list[preds_output]
    for  i,j in enumerate(precution_list):
        print(i+1,")",j)


chatbot()






