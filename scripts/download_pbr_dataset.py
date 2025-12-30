#!/usr/bin/env python3
"""
Download 150 PBR texture pairs from Poly Haven API for research dataset.

Requirements:
- 150 assets total (30 per category: Wood, Metal, Ground, Fabric, Abstract/Other)
- Download 'diff' (Albedo) and 'nor_gl' (OpenGL Normal) maps
- 1k resolution in jpg format
- Structured output: data/real_validation_set/<Category>/<AssetName>/
- Standardized filenames: albedo.jpg, normal.jpg
- ThreadPoolExecutor with max 4 workers
- Idempotent (skip if files exist)
"""

import os
import json
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict
import time
import sys

# Configuration
API_BASE_URL = "https://api.polyhaven.com/assets"
OUTPUT_BASE_DIR = Path("data/real_validation_set")
USER_AGENT = "ResearchDataAcquisition/1.0"
MAX_WORKERS = 4
ASSETS_PER_CATEGORY = 30
CATEGORIES = ["wood", "metal", "terrain", "fabric", "rock"]
TARGET_RESOLUTION = "1k"
TARGET_FORMAT = "jpg"

# Session with custom headers
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def fetch_assets_by_category(category: str, limit: int = 100) -> List[Dict]:
    """Fetch texture assets from Poly Haven API for a given category."""
    try:
        params = {
            "t": "textures",
        }
        response = session.get(API_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # API returns a dict with asset IDs as keys
        # Filter client-side by checking if category is in the asset's categories array
        if isinstance(data, dict):
            assets = []
            for asset_id, asset_data in data.items():
                asset_categories = asset_data.get("categories", [])
                # Check if our target category is in the asset's categories
                if category in asset_categories:
                    asset_data["id"] = asset_id
                    assets.append(asset_data)
                    if len(assets) >= limit:
                        break
            return assets
        return []
    except requests.RequestException as e:
        print(f"Error fetching assets for category '{category}': {e}")
        return []


def get_asset_download_urls(asset_id: str) -> Dict[str, Optional[str]]:
    """
    Fetch download URLs for albedo and normal maps from the files API.
    Returns dict with 'albedo' and 'normal' keys.
    """
    try:
        response = session.get(f"https://api.polyhaven.com/files/{asset_id}", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        urls = {"albedo": None, "normal": None}
        
        # Get albedo (Diffuse) URL
        if "Diffuse" in data and TARGET_RESOLUTION in data["Diffuse"]:
            if TARGET_FORMAT in data["Diffuse"][TARGET_RESOLUTION]:
                urls["albedo"] = data["Diffuse"][TARGET_RESOLUTION][TARGET_FORMAT].get("url")
        
        # Get normal (nor_gl for OpenGL) URL
        if "nor_gl" in data and TARGET_RESOLUTION in data["nor_gl"]:
            if TARGET_FORMAT in data["nor_gl"][TARGET_RESOLUTION]:
                urls["normal"] = data["nor_gl"][TARGET_RESOLUTION][TARGET_FORMAT].get("url")
        
        return urls
    except requests.RequestException as e:
        print(f"Error fetching URLs for asset '{asset_id}': {e}")
        return {"albedo": None, "normal": None}


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL to output_path."""
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        return True
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_asset_pair(
    asset_id: str, category: str, asset_name: str
) -> Dict[str, bool]:
    """Download both albedo and normal maps for an asset."""
    asset_dir = OUTPUT_BASE_DIR / category / asset_name
    asset_dir.mkdir(parents=True, exist_ok=True)

    albedo_path = asset_dir / "albedo.jpg"
    normal_path = asset_dir / "normal.jpg"

    # Check if files already exist (idempotency)
    if albedo_path.exists() and normal_path.exists():
        return {
            "asset_id": asset_id,
            "status": "skipped",
            "reason": "files_exist",
        }

    # Fetch download URLs
    urls = get_asset_download_urls(asset_id)
    
    if not urls["albedo"] or not urls["normal"]:
        return {
            "asset_id": asset_id,
            "status": "failed",
            "reason": "missing_urls",
        }

    # Download albedo
    albedo_success = download_file(urls["albedo"], albedo_path)

    # Download normal
    normal_success = download_file(urls["normal"], normal_path)

    if albedo_success and normal_success:
        return {
            "asset_id": asset_id,
            "status": "success",
            "category": category,
        }
    else:
        # Clean up partial downloads
        if albedo_path.exists() and not albedo_success:
            albedo_path.unlink()
        if normal_path.exists() and not normal_success:
            normal_path.unlink()
        return {
            "asset_id": asset_id,
            "status": "failed",
            "reason": f"albedo={albedo_success}, normal={normal_success}",
        }


def main():
    """Main execution function."""
    print("=" * 70)
    print("PBR Texture Dataset Downloader")
    print("=" * 70)
    print(f"Target: {len(CATEGORIES) * ASSETS_PER_CATEGORY} assets")
    print(f"Categories: {', '.join(CATEGORIES)}")
    print(f"Output: {OUTPUT_BASE_DIR}")
    print(f"Workers: {MAX_WORKERS}")
    print("=" * 70)

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    download_tasks = []
    results = {"success": 0, "failed": 0, "skipped": 0}
    used_asset_ids = set()  # Track used asset IDs to prevent duplicates

    # Collect all download tasks
    for category in CATEGORIES:
        print(f"\nFetching assets for category: {category}")
        assets = fetch_assets_by_category(category, limit=100)

        if not assets:
            print(f"  Warning: No assets found for category '{category}'")
            continue

        # Select ASSETS_PER_CATEGORY unique assets (not already used)
        selected_assets = []
        for asset in assets:
            asset_id = asset.get("id")
            if asset_id and asset_id not in used_asset_ids:
                selected_assets.append(asset)
                used_asset_ids.add(asset_id)
                if len(selected_assets) >= ASSETS_PER_CATEGORY:
                    break
        
        print(f"  Found {len(selected_assets)} unique assets, queuing for download")

        for asset in selected_assets:
            asset_id = asset.get("id")
            asset_name = asset.get("name", asset_id)

            if not asset_id:
                continue

            download_tasks.append((asset_id, category, asset_name))

    print(f"\nTotal tasks queued: {len(download_tasks)}")
    print("=" * 70)
    print("Starting downloads...\n")

    # Execute downloads with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_asset_pair, asset_id, category, name): (
                asset_id,
                category,
            )
            for asset_id, category, name in download_tasks
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            asset_id, category = futures[future]

            try:
                result = future.result()
                status = result.get("status")

                if status == "success":
                    results["success"] += 1
                    print(
                        f"[{completed}/{len(download_tasks)}] ✓ {asset_id} ({category})"
                    )
                elif status == "skipped":
                    results["skipped"] += 1
                    print(
                        f"[{completed}/{len(download_tasks)}] ⊘ {asset_id} ({category}) - already exists"
                    )
                else:
                    results["failed"] += 1
                    reason = result.get("reason", "unknown")
                    print(
                        f"[{completed}/{len(download_tasks)}] ✗ {asset_id} ({category}) - {reason}"
                    )
            except Exception as e:
                results["failed"] += 1
                print(f"[{completed}/{len(download_tasks)}] ✗ {asset_id} - {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Total: {results['success'] + results['failed'] + results['skipped']}")
    print("=" * 70)

    if results["success"] + results["skipped"] > 0:
        print(f"\nDataset saved to: {OUTPUT_BASE_DIR.absolute()}")
        return 0
    else:
        print("\nError: No assets were downloaded successfully.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
