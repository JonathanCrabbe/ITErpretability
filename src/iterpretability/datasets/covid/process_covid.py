import numpy as np
import pandas as pd



def process_covid():
    covid_dataset = pd.read_csv('src/iterpretability/datasets/covid/covid_data_brazil.csv')
    columns_to_keep = ['Sex_male', 'Age',
       'Fever', 'Cough', 'Sore_throat', 'Shortness_of_breath',
       'Respiratory_discomfort', 'SPO2', 'Dihareea', 'Vomitting',
       'Cardiovascular', 'Asthma', 'Diabetis', 'Pulmonary', 'Immunosuppresion',
       'Obesity', 'Liver', 'Neurologic', 'Renal', 'Branca', 'Preta', 'Amarela',
       'Parda', 'Indigena']

    covid_dataset = covid_dataset[columns_to_keep]

    covid_dataset['Age'] = (covid_dataset['Age'] - np.min(covid_dataset['Age'])) / (np.max(covid_dataset['Age']) - np.min(covid_dataset['Age']))

    return covid_dataset.to_numpy()

