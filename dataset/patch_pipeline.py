"""
patch_pipeline.py
Generates georeferenced 112x112 RGB patches from Sentinel-2 GeoTIFF tiles
and matches each patch to a poverty score label from the Metal Roof Index grid.

Usage:
    Run in Google Colab with Google Drive mounted.
    Update tile_paths, labels_path, and output_dir as needed.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
import os
from PIL import Image
from shapely.geometry import Point, shape
import json
import gc

# ── CONFIG ────────────────────────────────────────────────
tile_paths = [
    '/content/drive/MyDrive/BRACU_DATASET/bracu_full_image4-0.tif',
    '/content/drive/MyDrive/BRACU_DATASET/bracu_full_image4-1.tif',
    '/content/drive/MyDrive/BRACU_DATASET/bracu_full_image4-2.tif',
    '/content/drive/MyDrive/BRACU_DATASET/bracu_full_image4-3.tif',
]
labels_path = '/content/drive/MyDrive/BRACU_DATASET/bracu_labels_roofs4.csv'
output_dir  = '/content/drive/MyDrive/BRACU_DATASET/34k/dataset_34k'
patch_size  = 112
TARGET      = 34000

# ── Load labels ───────────────────────────────────────────
labels_df = pd.read_csv(labels_path)

def parse_geometry(geo_str):
    try:    return shape(json.loads(geo_str))
    except: return None

labels_df['geometry'] = labels_df['.geo'].apply(parse_geometry)
labels_df = labels_df[labels_df['geometry'].notna()].reset_index(drop=True)
print(f"Valid grid cells: {len(labels_df)}")

# ── Find last saved patch ─────────────────────────────────
os.makedirs(output_dir, exist_ok=True)
existing_pngs = [f for f in os.listdir(output_dir) if f.endswith('.png')]
start_count   = max([int(f.replace('patch_','').replace('.png',''))
                     for f in existing_pngs]) + 1 if existing_pngs else 0
print(f"Resuming from patch_{start_count}  (target: {TARGET})")

# ── Load existing CSV ─────────────────────────────────────
csv_path = os.path.join(output_dir, 'dataset.csv')
dataset  = pd.read_csv(csv_path).to_dict('records') if os.path.exists(csv_path) else []
print(f"Existing CSV records: {len(dataset)}")

# ── GPS match helper ──────────────────────────────────────
def get_poverty_score(lon, lat):
    pt = Point(lon, lat)
    for _, row in labels_df.iterrows():
        if row['geometry'].contains(pt):
            return row['poverty_score']
    labels_df['dist'] = labels_df['geometry'].apply(
        lambda g: pt.distance(g.centroid))
    return labels_df.loc[labels_df['dist'].idxmin(), 'poverty_score']

# ── Resume patching ───────────────────────────────────────
count        = start_count
global_patch = 0

for tile_path in tile_paths:
    if count >= TARGET:
        break

    print(f"\nProcessing {os.path.basename(tile_path)}...")

    with rasterio.open(tile_path) as src:
        tile_img       = src.read().astype(np.float32)
        tile_transform = src.transform

    tile_img = np.nan_to_num(tile_img, nan=0.0, posinf=0.0, neginf=0.0)
    tile_img = np.moveaxis(tile_img, 0, -1)

    for i in range(tile_img.shape[2]):
        band = tile_img[:, :, i]
        bmin, bmax = np.percentile(band[band > 0], [2, 98]) if np.any(band > 0) else (0, 1)
        tile_img[:, :, i] = np.clip((band - bmin) / (bmax - bmin + 1e-8), 0, 1)

    tile_img = (tile_img * 255).astype(np.uint8)
    H, W, _  = tile_img.shape
    print(f"Tile size: {H} x {W}")

    for row in range(0, H - patch_size, patch_size):
        for col in range(0, W - patch_size, patch_size):
            patch = tile_img[row:row + patch_size, col:col + patch_size]

            if patch.mean() < 5:
                continue

            if global_patch < start_count:
                global_patch += 1
                continue

            center_lon, center_lat = xy(tile_transform,
                                        row + patch_size // 2,
                                        col + patch_size // 2)

            poverty_score = get_poverty_score(center_lon, center_lat)

            patch_rgb = patch[:, :, :3]
            png_name  = f'patch_{count}.png'
            Image.fromarray(patch_rgb).save(os.path.join(output_dir, png_name))

            dataset.append({
                'image':         png_name,
                'row':           row,
                'col':           col,
                'center_lat':    center_lat,
                'center_lon':    center_lon,
                'poverty_score': poverty_score
            })

            count        += 1
            global_patch += 1

            if count % 500 == 0:
                pd.DataFrame(dataset).to_csv(csv_path, index=False)
                print(f"  Patches saved: {count}")

            if count >= TARGET:
                break
        if count >= TARGET:
            break

    del tile_img
    gc.collect()
    print(f"Total patches after this tile: {count}")

# ── Final CSV save ────────────────────────────────────────
pd.DataFrame(dataset).to_csv(csv_path, index=False)
print(f"\nDone! Total patches: {count}")
