input_dir = data_dir + "preproc/"
model_dir = data_dir + f"models/{data_type}/"
shap_dir = data_dir + "shap/"
gene_dir = data_dir + "gene_lists/"

import shap
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from keras import backend
from keras.models import model_from_json

from src.constants import map_dict, inv_map
from src.common import run_inference, prepare_x_y 
from src.shap_utils import filter_shap, get_rank_df

def main(
    data_dir
):

    """Main function
    
    Args:
        model_json:            JSON file with model architecture
        model_weights:         H5 file with trained weights 
        gpus_count:            number of GPUs to use    
        reference_data_path:   path to reference data file
        test_data_path:        path to test data file
        output_dir:            output directory to save SHAP scores file
        frac:                  fraction of test set used
    
    Returns:
        None
    """

    ref_data = pd.read_pickle(
        input_dir + f'gtex_filtered_tmm_intersect_{data_type}.pkl'
    )
    
    test_data = pd.read_pickle(
        input_dir + 'gtex_filtered_tmm_intersect_test.pkl'
    )

    # Load model beatifully
    model_json_path = model_dir+f"{data_type}_model_topology.json"
    model = model_from_json(
        open(model_json_path, "r").read()
    )

    # load weights into new model
    model_weights_path = model_dir+f"{data_type}_model_weights.hdf5"
    model.load_weights(model_weights_path)

    y_pred = run_inference(test_data, model)


    X_ref, _ = prepare_x_y(ref_data, "tissue")
    X_test, _ = prepare_x_y(test_data, "tissue")

    explainer = shap.GradientExplainer(model, X_ref)

    out_list = []
    num_samples = np.shape(X_test)[0]
    for sample in tqdm(range(3)):
        # shap
        shap_values = explainer.shap_values(X_test[sample : sample + 1])
        out_list.append(shap_values)
    shap_arr = np.squeeze(np.array(out_list))

    shap_df = filter_shap(test_data, shap_arr, y_pred)

    backend.clear_session()

    output_file = shap_dir + f"shap_scores_{data_type}_df.pkl"

    output_file = shap_dir + "shap_tmm_smote_df.pkl"
    with open(output_file, "rb") as f:
        shap_df = pickle.load(f)

    rank_df = get_rank_df(shap_df)


    gene_list = []
    for index, row in tqdm(rank_df.iterrows()):
        gene_list.extend(list(row.values))
        val = len(np.unique(gene_list))/len(gene_list)
        if val <= 0.5:
            print(index)
            break


    shap_genes = np.unique(rank_df[:index].values.flatten())

    output_file = shap_dir + f"shap_scores_{data_type}.pkl"

    pickle.dump(shap_df, open(str(output_file), "wb"))

    output_file = shap_dir + f"{data_type}_ranks.pkl"

    pickle.dump(rank_df, open(str(output_file), "wb"))

    output_file = gene_dir + f"shap_genes.pkl"

    pickle.dump(shap_genes, open(str(output_file), "wb"))