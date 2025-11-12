from datasets import load_dataset

# Load training split - ignore verification to skip unsupervised split check
# The data is already downloaded, so this will work
train_data = load_dataset("imdb", split='train', ignore_verifications=True)

print(f"Train size: {len(train_data)}")
print(f"\nFirst example:")
print(train_data[0])
