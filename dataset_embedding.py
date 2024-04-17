from sentence_transformers import SentenceTransformer
import pandas as pd

def get_abstract_from_arxiv_id(arxiv_id):
    """
    Returns the abstract from the Kaggle DB given an arxiv_id
    """
    return 'abstract' + ' ' + arxiv_id # Lav færdig

def transform_dataset(dataset, transformer = 'sentence-transformers/all-MiniLM-L6-v2'):
    """
    Takes the dataset in the form:
    dataset:    |Article ArXiV ID | Reference ArXiV ID | context |

    and a transformer which is the name of the transformer model to be used.
    default for transformer: 'sentence-transformers/all-MiniLM-L6-v2'

    Returns the vector embedded dataset in the form:
    transformed_dataset: | transformed context | transformed abstract |
    """
    model = SentenceTransformer(transformer)
    transformed_dataset = []
    for i in range(len(dataset)):
        context = dataset.iloc[i]['context']
        arxiv_id = dataset.iloc[i]['ArXiV ID']
        abstract = get_abstract_from_arxiv_id(arxiv_id)
        transformed_context, transformed_abstract = model.encode([context, abstract])
        transformed_dataset.append([transformed_context, transformed_abstract])
    return pd.DataFrame(transformed_dataset, columns=['context', 'abstract'])