
import sys
sys.path.append('src/')
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from utils import save_as_pickle
import pandas as pd
import pickle
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score
import json

@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('train_data_filepath', type=click.Path())
@click.argument('target_data_filepath', type=click.Path())
@click.argument('output_path', type=click.Path())
def main(model_filepath, train_data_filepath, target_data_filepath, output_path):

    logger = logging.getLogger(__name__)
    logger.info('model evaluation')
    
    X_test = pd.read_pickle(train_data_filepath)
    y_test = pd.read_pickle(target_data_filepath)
    
    pickled_model = pickle.load(open(model_filepath, 'rb'))
    preds_class = pickled_model.predict(X_test)
    
    y_test_array = y_test.to_numpy()
    
    precision = precision_score(y_test_array, preds_class, average='micro')
    recall = recall_score(y_test_array, preds_class, average='micro')
    accuracy = accuracy_score(y_test_array, preds_class)
    rocauc = roc_auc_score(y_test_array, preds_class, average='micro')
    
    metrics_dictionary = {
        "Model Name": "CatBoostClassifier",
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

