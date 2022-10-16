
import sys
sys.path.append('src/')
sys.path.append('src/data')
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from utils import save_as_pickle
import pandas as pd
import pickle
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import lightgbm as ltb
from category_encoders.count import CountEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from config import REAL_COLS

@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('input_filepath_train', type=click.Path())
@click.argument('input_filepath_target', type=click.Path())
@click.argument('output_path', type=click.Path())
def main(model_filepath, input_filepath_train, input_filepath_target, output_path):
    logger = logging.getLogger(__name__)
    logger.info('pipeline for lightgbm model')
    
    train = pd.read_pickle(input_filepath_train)
    target = pd.read_pickle(input_filepath_target)
    
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.4, random_state=0)
    
    cat_features__=[i for i in X_train.columns if X_train.dtypes[i]=='category']
    
    model_ltb = ltb.LGBMClassifier(
        boosting_type ='dart',
        num_leaves = 20,
        learning_rate = 0.01,
        n_estimators = 1000
    )
    
    real_pipe = Pipeline([('scaler', StandardScaler())])

    cat_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore',sparse=False))])

    preprocess_pipe = ColumnTransformer(transformers=[
        ('real_cols', real_pipe, REAL_COLS),
        ('cat_cols', cat_pipe, cat_features__),
        ('cat_bost_cols', CountEncoder(), cat_features__), ##cat_features
    ])

    model_pipe = Pipeline([('preprocess', preprocess_pipe),('model', model_ltb)])
    pipline_ltb = MultiOutputClassifier(model_pipe, n_jobs=4)
    pipline_ltb.fit(X_train, y_train)
    
    preds_class = pipline_ltb.predict(X_test)
    y_test_array = y_test.to_numpy()
    
    precision = precision_score(y_test_array, preds_class, average='micro')
    recall = recall_score(y_test_array, preds_class, average='micro')
    accuracy = accuracy_score(y_test_array, preds_class)
    rocauc = roc_auc_score(y_test_array, preds_class, average='micro')
    
    metrics_dictionary = {
        "Model Name": "model_ltb (pipeline)",
        "Precision Score": precision,
        "Recall Score": recall,
        "Accuracy Score": accuracy,
        "ROC AUC Score": rocauc
    }
    
    json_object = json.dumps(metrics_dictionary, indent=4)
 
    # Writing to sample.json
    with open(output_path, "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

