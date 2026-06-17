from datasets import load_from_disk

def load_raw_data(path="data/raw/research_paper2023"):
    ds = load_from_disk(path)
    return ds["train"]