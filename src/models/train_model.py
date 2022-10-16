
import sys
sys.path.append('src/')
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from utils import save_as_pickle
from config import REAL_COLS
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import lightgbm as ltb
from category_encoders.count import CountEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from catboost import CatBoostClassifier
import pickle
from sklearn.metrics import precision_score
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GridSearchCV

@click.command()
@click.argument('input_filepath_train', type=click.Path(exists=True))
@click.argument('input_filepath_target', type=click.Path())
@click.argument('output_filepath_model', type=click.Path())
@click.argument('evaluate_test_output_data', type=click.Path())
@click.argument('evaluate_test_output_target', type=click.Path())
def main(input_filepath_train, input_filepath_target, output_filepath_model, evaluate_test_output_data, evaluate_test_output_target):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('train model')
    
    train = pd.read_pickle(input_filepath_train)
    target = pd.read_pickle(input_filepath_target)
    
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.4, random_state=0)
    
    cat_features__=[i for i in X_train.columns if X_train.dtypes[i]=='category']
    cat_features_=[i for i in range(len(X_train.columns)) if X_train.dtypes[X_train.columns[i]]=='category']

    model_catboost = CatBoostClassifier(
        iterations=1000, 
        loss_function='MultiLogloss', 
        eval_metric='MultiLogloss',
        learning_rate=0.03,
        bootstrap_type='Bayesian',
        boost_from_average=False,
        leaf_estimation_iterations=1,
        leaf_estimation_method='Gradient',
        ctr_leaf_count_limit = 1,
        store_all_simple_ctr = True,
        silent = True
    )
    
    train_pool = Pool(X_train, 
                  y_train,
                  cat_features_)
    
    model_catboost.fit(train_pool)
    pickle.dump( model_catboost, open(output_filepath_model, 'wb'))
    save_as_pickle(X_test, evaluate_test_output_data)
    save_as_pickle(y_test, evaluate_test_output_target)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()