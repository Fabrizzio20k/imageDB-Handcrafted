
from pathlib import Path
import pandas as pd
import requests
import time
from tqdm import tqdm

caltech_path = Path("Data/caltech-101/101_ObjectCategories")
queries_path = Path("Data/queries")
queries_path.mkdir(exist_ok=True, parents=True)

categories = sorted([d.name for d in caltech_path.iterdir() if d.is_dir()])

queries_data = []
counter = 1

headers = {'User-Agent': 'Mozilla/5.0'}

for category in tqdm(categories, desc="Descargando"):
    try:
        search = category.replace('_', ' ')
        wiki_api = f"https://en.wikipedia.org/w/api.php?action=query&titles={search}&prop=pageimages&format=json&pithumbsize=400"
        
        response = requests.get(wiki_api, headers=headers, timeout=10)
        data = response.json()
        
        pages = data.get('query', {}).get('pages', {})
        for page_id, page_data in pages.items():
            if 'thumbnail' in page_data:
                img_url = page_data['thumbnail']['source']
                
                img_response = requests.get(img_url, headers=headers, timeout=10)
                
                if img_response.status_code == 200:
                    filename = f"querie_{counter:02d}.jpg"
                    filepath = queries_path / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(img_response.content)
                    
                    queries_data.append({'filename': filename, 'class_label': category})
                    counter += 1
                    tqdm.write(f"✓ {category}")
                    break
        
        time.sleep(0.5)
    except Exception as e:
        tqdm.write(f"✗ {category}")

pd.DataFrame(queries_data).to_csv(queries_path / "queries.csv", index=False)
print(f"\nTotal: {len(queries_data)}/{len(categories)}")