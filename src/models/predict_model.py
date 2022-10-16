
import sys
sys.path.append('src/')
sys.path.append('src/data')
sys.path.append('src/features')
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from utils import save_as_pickle
import pandas as pd
import pickle
from preprocess import preprocess_data
from featurization import featurize_data
import json
from config import TARGET_COLS

@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('test_data_filepath', type=click.Path())
@click.argument('output_path_data', type=click.Path())
@click.argument('output_path_prediction', type=click.Path())
def main(model_filepath, test_data_filepath, output_path_data, output_path_prediction):
    logger = logging.getLogger(__name__)
    logger.info('making prediction for test data')
    
    X_test = pd.read_csv(test_data_filepath)
    X_test = preprocess_data(X_test)
    X_test = featurize_data(X_test)
    pickled_model = pickle.load(open(model_filepath, 'rb'))
    preds_class = pickled_model.predict(X_test)
    df = pd.DataFrame(preds_class, columns = TARGET_COLS)
    
    save_as_pickle(df, output_path_prediction)
    save_as_pickle(X_test, output_path_data)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

