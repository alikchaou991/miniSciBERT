from datasets import load_dataset

ds = load_dataset("Falah/research_paper2023")

# Save it to your data/raw folder
ds.save_to_disk("data/raw/research_paper2023")

print("Dataset downloaded successfully!")
print(ds)