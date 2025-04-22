import requests
import os

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Téléchargé : {local_filename}")

# Créer un dossier pour les annotations
os.makedirs("annotations", exist_ok=True)

# URLs des annotations Kinetics-700-2020
urls = {
    "train": "https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv",
    "val": "https://s3.amazonaws.com/kinetics/700_2020/annotations/val.csv",
    "test": "https://s3.amazonaws.com/kinetics/700_2020/annotations/test.csv"
}

# Télécharger les fichiers
for split, url in urls.items():
    local_filename = os.path.join("annotations", f"k700_2020_{split}.csv")
    download_file(url, local_filename)