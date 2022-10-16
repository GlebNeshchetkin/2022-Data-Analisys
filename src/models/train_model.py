
import sys
sys.path.append('../../src/')
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
    logger.info('making final data set from raw data')
    
    train = pd.read_pickle(input_filepath_train)
    target = pd.read_pickle(input_filepath_target)
    
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.4, random_state=0)
    
    real_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore',sparse=False))])
    
    cat_features__=[i for i in X_train.columns if X_train.dtypes[i]=='category']
    cat_features_=[i for i in range(len(X_train.columns)) if X_train.dtypes[X_train.columns[i]]=='category']
    
    preprocess_pipe = ColumnTransformer(transformers=[
        ('real_cols', real_pipe, REAL_COLS),
        ('cat_cols', cat_pipe, cat_features__),
        ('cat_bost_cols', CountEncoder(), cat_features__), ##cat_features
    ])
    
    model_ltb = ltb.LGBMClassifier(
        boosting_type ='dart',
        num_leaves = 20,
        learning_rate = 0.01,
        n_estimators = 500
    )
    
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

    model_pipe = Pipeline([('preprocess', preprocess_pipe),('model', model_catboost)])
    pipline_catboost = MultiOutputClassifier(model_pipe, n_jobs=4)
    pipline_catboost.fit(X_train, y_train)
    preds_class = pipline_catboost.predict(X_test)
    print(preds_class)
    y_test_array = y_test.to_numpy()
    print(precision_score(y_test_array, preds_class, average='micro'))
    pickle.dump(pipline_catboost, open(output_filepath_model, 'wb'))
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