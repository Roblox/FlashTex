import json
import glob
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import track
from PIL import Image
import numpy as np
from tqdm import tqdm

base_dir = "your/rendered/data/subsetA/" #TODO: change the data directory
render_folders = glob.glob(os.path.join(base_dir, "*"))
print(len(render_folders))

def process_folder(render_folder):
    metadata_path = os.path.join(render_folder, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_render = json.load(f)

    if 'annotations' not in metadata_render:
        return None
    anno = metadata_render['annotations']
    text_prompt = anno['name'] + ', ' + ', '.join([tag['name'] for tag in anno['tags']])
    if '3d' not in text_prompt:
        text_prompt += ', 3d assets'

    results = []
    for img_i in range(12):
        rgb_path = os.path.join(render_folder, f"{str(img_i).zfill(3)}_rgb.png")
        cond_path = os.path.join(render_folder, f"{str(img_i).zfill(3)}_cond.png")

        cond1_path = os.path.join(render_folder, f"{str(img_i).zfill(3)}_m0r1.png")
        cond2_path = os.path.join(render_folder, f"{str(img_i).zfill(3)}_m5r5.png")
        cond3_path = os.path.join(render_folder, f"{str(img_i).zfill(3)}_m1r0.png")

        if (not os.path.exists(rgb_path)) or (not os.path.exists(cond1_path)) or (not os.path.exists(cond2_path)) or (not os.path.exists(cond3_path)):
            print(render_folder, img_i)
            continue

        results.append({
            "text": text_prompt,
            "image": os.path.relpath(rgb_path, base_dir),
            "conditioning_image": os.path.relpath(cond_path, base_dir),
        })
        
        if os.path.exists(cond_path):
            continue
    
    return results

# Function to process folders and update the progress bar
def process_folders_with_progress(folders):
    results = []
    with ThreadPoolExecutor() as executor:
        # Create a list of future tasks
        futures = [executor.submit(process_folder, folder) for folder in folders]

        # Process the futures and update the progress bar
        for future in tqdm(as_completed(futures), total=len(folders)):
            result = future.result()
            if result:
                results.extend(result)
    return results

# Call the processing function with progress tracking
metadata_controlnet = process_folders_with_progress(render_folders)

df = pd.DataFrame(metadata_controlnet)
metadata_output_path = os.path.join(base_dir, "train.jsonl")
with open(metadata_output_path, "w") as f:
    f.write(df.to_json(orient='records', lines=True, force_ascii=False))


