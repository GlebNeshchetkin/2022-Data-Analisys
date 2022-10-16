import pandas as pd
import numpy as np
import config as cfg
from config import *
import time


def leads_healthy_lifestyle(data: pd.DataFrame) -> None:
    data['leads_healthy_lifestyle'] = 0
    for i in range(len(data['Статус Курения'])):
        if data['Статус Курения'][i] == 'Никогда не курил(а)' and data['Алкоголь'][i] == 'никогда не употреблял':
            print(i)
            data['leads_healthy_lifestyle'][i] = 0
        else:
            data['leads_healthy_lifestyle'][i] = 1
            
def early_riser(data: pd.DataFrame) -> None:
    data['early_riser'] = 0
    for i in range(len(data['Время пробуждения'])):
        if pd.to_datetime((data['Время пробуждения'][i])).hour <=6:
            data['early_riser'][i] = 0
        else:
            data['early_riser'][i] = 1
            
def serious_diseases(data: pd.DataFrame) -> None:
    data['serious_diseases'] = 0
    for i in range(len(data['Сахарный диабет'])):
        if data['Сахарный диабет'][i] == 1 or data['Гепатит'][i] == 1 or data['Онкология'][i] == 1 or data['Хроническое заболевание легких'][i] == 1 or data['Бронжиальная астма'][i] ==1 or data['Туберкулез легких '][i] == 1 or data['ВИЧ/СПИД'][i] == 1:
            data['serious_diseases'][i] = 0
        else:
            data['serious_diseases'][i] = 1

def featurize_data(df: pd.DataFrame) -> pd.DataFrame:
    leads_healthy_lifestyle(df)
    early_riser(df)
    serious_diseases(df)
    return df
        