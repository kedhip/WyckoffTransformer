import pickle
import logging
import gzip
from data import read_all_MP_csv
from tokenization import get_tokens, MASK_SITE

MP_20_cache_data = "cache/mp_20/data.pkl.gz"
MP_20_cache_tensors = "cache/mp_20/tensors.pkl.gz"
MP_20_path = "cdvae/data/mp_20"

def load_all_data(allow_retokenize:bool=False):
    try:
        with gzip.open(MP_20_cache_data, "rb") as f:
            datasets_pd, max_len = pickle.load(f)
    except Exception as e:
        logging.info("Error reading data cache:")
        logging.info(e)
        logging.info("Reading from csv")
        datasets_pd, max_len = read_all_MP_csv()
        with gzip.open(MP_20_cache_data, "wb") as f:
            pickle.dump((datasets_pd, max_len), f)
    try:
        with gzip.open(MP_20_cache_tensors, "rb") as f:
            tensors, site_to_ids, element_to_ids, spacegroup_to_ids = pickle.load(f)
    except Exception as e:
        if not allow_retokenize:
            raise e
        logging.warning("Error reading tensor cache! The token order will change!")
        logging.info(e)
        logging.info("Generating new tensors")
        tensors, site_to_ids, element_to_ids, spacegroup_to_ids = get_tokens(datasets_pd)
        with gzip.open(MP_20_cache_tensors, "wb") as f:
            pickle.dump((tensors, site_to_ids, element_to_ids, spacegroup_to_ids), f)
    return datasets_pd, tensors, site_to_ids, element_to_ids, spacegroup_to_ids, max_len