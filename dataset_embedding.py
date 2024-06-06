from sentence_transformers import SentenceTransformer
from RandomizeKaggleDB import read_json_DB
import pickle
from tqdm.auto import tqdm

id_to_abstract_path = 'arXivIDs to Abstract_Subset_physics.json'
ARXIVID_TO_ABSTRACT = read_json_DB(filepath=id_to_abstract_path)

def get_abstract_from_arxiv_id(arxiv_id: str) -> str:
    """
    Returns the abstract from the Kaggle DB given an arxiv_id
    """
    return ARXIVID_TO_ABSTRACT[arxiv_id]

def transform_dataset(dataset: list, transformer: str='sentence-transformers/all-MiniLM-L6-v2') -> list | None:
    """
    Takes the dataset in the form:
    dataset:    |Article ArXiV ID | Reference ArXiV ID | context |

    and a transformer which is the name of the transformer model to be used.
    default for transformer: 'sentence-transformers/all-MiniLM-L6-v2'

    Returns the vector embedded dataset in the form:
    transformed_dataset: | transformed context | transformed abstract |
    """
    try:
        model = SentenceTransformer(transformer)
        transformed_dataset = []
        for i in tqdm(range(len(dataset)), desc="Transforming dataset"):
            context = dataset[i][2] # Context
            arxiv_id = dataset[i][1] # arXiv-ID
            abstract = get_abstract_from_arxiv_id(arxiv_id)
            transformed_context, transformed_abstract = model.encode([context, abstract])
            transformed_dataset.append([transformed_context, transformed_abstract])
    except Exception as e:
        print(f"Failed to transform the dataset with error: {e}")
        return None
    return transformed_dataset

def download_dataset(filepath: str) -> list | None:
    try:
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"Failed to download the dataset with error: {e}")
        return None
    return dataset

def upload_transformed_dataset(transformed_dataset: str, filepath: str) -> str:
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(transformed_dataset, f)
    except Exception as e:
        print(f"Failed to upload the transformed dataset with error: {e}")
        return 'Failed'
    return 'Success'

if __name__ == '__main__':
    # Load the dataset
    dataset = download_dataset('dataset.pkl')
    if not dataset:
        exit()
    transformed_dataset = transform_dataset(dataset)
    if not transformed_dataset:
        exit()
    upload_transformed_dataset(transformed_dataset, 'transformed_dataset.pkl')
    