import pandas as pd
import numpy as np
import config as cfg
from config import *


def drop_unnecesary_id(df: pd.DataFrame) -> pd.DataFrame:
    if 'ID_y' in df.columns:
        df = df.drop('ID_y', axis=1)
    return df


def fill_sex(df: pd.DataFrame) -> pd.DataFrame:
    most_freq = df[cfg.SEX_COL].value_counts().index[0]
    df[cfg.SEX_COL] = df[cfg.SEX_COL].fillna(most_freq)
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('category')

    ohe_int_cols = df[cfg.OHE_COLS].select_dtypes('number').columns
    df[ohe_int_cols] = df[ohe_int_cols].astype(np.int8)

    df[cfg.REAL_COLS] = df[cfg.REAL_COLS].astype(np.float32)
    return df


def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
    df = df.set_index(idx_col)
    return df
    
def replace_all_nans(df: pd.DataFrame) -> None:
    nan_values = df.isna().sum()
    for i in range(len(nan_values)):
        if nan_values[i]!=0:
            if nan_values.index[i] in REPLACE_MOST_FREQUENT:
                most_freq = df[nan_values.index[i]].value_counts().index[0]
                df[nan_values.index[i]] = df[nan_values.index[i]].fillna(most_freq)
                print(most_freq)
            if nan_values.index[i] in REPLACE_ZERO:
                df[nan_values.index[i]] = df[nan_values.index[i]].fillna(0)
                print(0)
            if nan_values.index[i] in ['Статус Курения']:
                df[nan_values.index[i]] = df[nan_values.index[i]].fillna('Никогда не курил(а)')
                print('Никогда не курил(а)')
            if nan_values.index[i] in ['Алкоголь']:
                df[nan_values.index[i]] = df[nan_values.index[i]].fillna('никогда не употреблял')
                print('никогда не употреблял')
            if nan_values.index[i] in ['Возраст алког']:
                most_freq = df[nan_values.index[i]].value_counts().index[0]
                for j in range(len(df[nan_values.index[i]])):
                    if(df['Алкоголь'][j] == 'никогда не употреблял'):
                        df[nan_values.index[i]][j] = 0
                    else:
                        df[nan_values.index[i]][j] = most_freq
            if nan_values.index[i] in ['Возраст курения']:
                most_freq = df[nan_values.index[i]].value_counts().index[0]
                for j in range(len(df[nan_values.index[i]])):
                    if(df['Статус Курения'][j] == 'Никогда не курил(а)'):
                        df[nan_values.index[i]][j] = 0
                    else:
                        df[nan_values.index[i]][j] = most_freq
            if nan_values.index[i] in ['Сигарет в день']:
                most_freq = df[nan_values.index[i]].value_counts().index[0]
                for j in range(len(df[nan_values.index[i]])):
                    if(df['Возраст курения'][j] == 0):
                        df[nan_values.index[i]][j] = 0
                    else:
                        df[nan_values.index[i]][j] = most_freq
            if nan_values.index[i] in ['Частота пасс кур']:
                most_freq = df[nan_values.index[i]].value_counts().index[0]
                for j in range(len(df[nan_values.index[i]])):
                    if(df['Пассивное курение'][j] == 0):
                        df[nan_values.index[i]][j] = '0 раз'
                    else:
                        df[nan_values.index[i]][j] = most_freq
                        

def add_ordinal_smoke_frequency(data: pd.DataFrame) -> None:
    data['Частота пасс кур cat'] = (
        data['Частота пасс кур'].replace({
            '0 раз': 0,
            '1-2 раза в неделю': 1,
            '3-6 раз в неделю': 2,
            'не менее 1 раза в день': 3,
            '2-3 раза в день': 4,
            '4 и более раз в день': 5
        })
    )
    
def add_alcohol_ordinal(data: pd.DataFrame) -> None:
    data['Алкоголь cat'] = (
        data['Алкоголь'].replace({
            'никогда не употреблял': 0,
            'ранее употреблял': 1,
            'употребляю в настоящее время': 2
        })
    )

def add_ordinal_education_feature(data: pd.DataFrame) -> None:
    # образование как порядковый признак
    data['Образование cat'] = data['Образование'].str.slice(0, 1).astype(np.int8)


def add_ordinal_smoke_status(data: pd.DataFrame) -> None:
    data['Статус Курения cat'] = (
        data['Статус Курения'].replace({
            'Никогда не курил(а)': 0,
            'Никогда не курил': 0,
            'Бросил(а)': 1,
            'Курит': 2
        }))

def add_prof_ordinal(data: pd.DataFrame) -> None:
    data['Профессия cat'] = (
        data['Профессия'].replace({
            'ведение домашнего хозяйства': 0,
            'вооруженные силы': 1,
            'дипломированные специалисты': 2,
            'квалифицированные работники сельского хозяйства и рыболовного': 3,
            'низкоквалифицированные работники': 4,
            'операторы и монтажники установок и машинного оборудования': 5,
            'представители   законодат.   органов   власти,  высокопостав. долж.лица и менеджеры': 6,
            'работники,  занятые в сфере обслуживания, торговые работники магазинов и рынков': 7,
            'ремесленники и представители других отраслей промышленности': 8,
            'служащие': 9,
            'техники и младшие специалисты': 10
        })
    )

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, cfg.ID_COL)
    df = drop_unnecesary_id(df)
    df = fill_sex(df)
    df = cast_types(df)
    df["Частота пасс кур"] = df["Частота пасс кур"].cat.add_categories('0 раз')
    replace_all_nans(df)
    add_ordinal_smoke_frequency(df)
    add_alcohol_ordinal(df)
    add_ordinal_education_feature(df)
    add_ordinal_smoke_status(df)
    add_prof_ordinal(df)
    return df



def preprocess_target(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.TARGET_COLS] = df[cfg.TARGET_COLS].astype(np.int8)
    return df


def extract_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, target = df.drop(cfg.TARGET_COLS, axis=1), df[TARGET_COLS]
    return df, target