from datasets import load_from_disk
import pandas as pd

def load_raw_data(path="data/raw/research_paper2023"):
    ds = load_from_disk(path)
    df = ds['train'].to_pandas()
    return df