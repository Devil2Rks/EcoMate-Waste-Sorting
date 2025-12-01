"""
Google Earth Engine Data Acquisition for Sundarbans Water Body Classification
Fetches Sentinel-2 Level-2A imagery with temporal sequences for research

Author: B.Tech Research Team
Target: Sundarbans (primary), Chilika Lake, Brahmaputra (validation)
"""

import ee
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime, timedelta
import time


class SentinelDataAcquisition:
    """
    Handles acquisition of Sentinel-2 Level-2A data from Google Earth Engine
    for water body classification research in Indian wetland regions
    """
    
    def __init__(self, drive_path: str = '/content/drive/MyDrive/WaterBodyResearch'):
        """
        Initialize data acquisition system
        
        Args:
            drive_path: Google Drive path for storing downloaded data
        """
        self.drive_path = drive_path
        self.ensure_directories()
        
        # Initialize Google Earth Engine
        try:
            ee.Initialize()
            print("Google Earth Engine initialized successfully")
        except Exception as e:
            print(f"GEE initialization failed: {e}")
            print("Please authenticate with: ee.Authenticate()")
    
    def ensure_directories(self):
        """Create necessary directories in Google Drive"""
        directories = [
            f"{self.drive_path}/data/raw/sundarbans",
            f"{self.drive_path}/data/raw/chilika", 
            f"{self.drive_path}/data/raw/brahmaputra",
            f"{self.drive_path}/data/processed",
            f"{self.drive_path}/data/annotations",
            f"{self.drive_path}/checkpoints",
            f"{self.drive_path}/results"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Ensured directory: {directory}")
    
    def get_study_regions(self) -> Dict[str, Dict]:
        """Define study region boundaries and parameters"""
        
        return {
            'sundarbans': {
                'geometry': ee.Geometry.Rectangle([88.0, 21.5, 89.5, 22.5]),
                'name': 'Sundarbans Mangrove Forest',
                'target_patches': 70,
                'temporal_frames': 3,
                'months': [11, 12, 1, 2, 3],  # Post-monsoon to pre-monsoon
                'priority': 'primary'
            },
            'chilika': {
                'geometry': ee.Geometry.Rectangle([85.5, 20.5, 85.7, 20.7]),
                'name': 'Chilika Lake',
                'target_patches': 30,
                'temporal_frames': 2,
                'months': [11, 12, 1, 2],  # Winter months
                'priority': 'validation'
            },
            'brahmaputra': {
                'geometry': ee.Geometry.Rectangle([89.5, 26.0, 92.0, 27.0]),
                'name': 'Brahmaputra Floodplains',
                'target_patches': 30,
                'temporal_frames': 2,
                'months': [10, 11, 3, 4],  # Pre and post flood
                'priority': 'validation'
            }
        }
    
    def create_sentinel2_collection(self, region_name: str, 
                                   start_date: str, end_date: str) -> ee.ImageCollection:
        """
        Create filtered Sentinel-2 Level-2A collection for specified region
        
        Args:
            region_name: Name of study region
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Filtered Earth Engine ImageCollection
        """
        
        regions = self.get_study_regions()
        region_info = regions[region_name]
        
        # Create Sentinel-2 Level-2A collection
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(region_info['geometry'])
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                     .select(['B2', 'B3', 'B4', 'B8', 'SCL']))  # Blue, Green, Red, NIR, Scene Classification
        
        print(f"Created collection for {region_info['name']}: {collection.size().getInfo()} images")
        
        return collection
    
    def compute_ndwi_and_indices(self, image: ee.Image) -> ee.Image:
        """
        Compute NDWI and other water indices for image
        
        Args:
            image: Sentinel-2 image
            
        Returns:
            Image with added spectral indices
        """
        
        # Compute NDWI (Normalized Difference Water Index)
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # Compute MNDWI (Modified NDWI using SWIR)
        # Note: Using B3 as proxy since we're focusing on RGB+NIR
        mndwi = image.normalizedDifference(['B3', 'B4']).rename('MNDWI')
        
        # Compute NDVI for vegetation context
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # Add all bands and indices
        return image.addBands([ndwi, mndwi, ndvi])
    
    def mask_clouds_and_shadows(self, image: ee.Image) -> ee.Image:
        """
        Mask clouds and shadows using SCL band
        
        Args:
            image: Sentinel-2 image with SCL band
            
        Returns:
            Masked image
        """
        
        scl = image.select('SCL')
        
        # SCL values: 3=cloud shadows, 8=cloud medium probability, 9=cloud high probability, 10=thin cirrus
        cloud_mask = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
        
        return image.updateMask(cloud_mask.Not())
    
    def generate_sampling_points(self, region_name: str, num_points: int) -> ee.FeatureCollection:
        """
        Generate stratified sampling points within region
        
        Args:
            region_name: Name of study region
            num_points: Number of sampling points to generate
            
        Returns:
            FeatureCollection of sampling points
        """
        
        regions = self.get_study_regions()
        region_geometry = regions[region_name]['geometry']
        
        # Create stratified sampling based on water probability
        # Use JRC Global Surface Water for stratification
        gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
        
        # Create water probability strata
        water_high = gsw.gt(75)  # High water probability
        water_medium = gsw.gt(25).And(gsw.lte(75))  # Medium water probability  
        water_low = gsw.lte(25)  # Low water probability
        
        # Generate points for each stratum
        points_high = water_high.sample(
            region=region_geometry,
            scale=500,
            numPixels=num_points // 3,
            geometries=True
        )
        
        points_medium = water_medium.sample(
            region=region_geometry,
            scale=500,
            numPixels=num_points // 3,
            geometries=True
        )
        
        points_low = water_low.sample(
            region=region_geometry,
            scale=500,
            numPixels=num_points // 3,
            geometries=True
        )
        
        # Combine all points
        all_points = points_high.merge(points_medium).merge(points_low)
        
        return all_points
    
    def extract_temporal_patches(self, region_name: str, 
                               year: int = 2023) -> List[Dict]:
        """
        Extract temporal patch sequences for specified region
        
        Args:
            region_name: Name of study region
            year: Year to extract data from
            
        Returns:
            List of patch dictionaries with temporal sequences
        """
        
        regions = self.get_study_regions()
        region_info = regions[region_name]
        
        # Generate sampling points
        sampling_points = self.generate_sampling_points(
            region_name, region_info['target_patches']
        )
        
        patches = []
        target_months = region_info['months']
        temporal_frames = region_info['temporal_frames']
        
        # For each sampling point, extract temporal sequence
        points_list = sampling_points.getInfo()['features']
        
        for i, point in enumerate(points_list[:region_info['target_patches']]):
            print(f"Processing patch {i+1}/{region_info['target_patches']} for {region_name}")
            
            point_geom = ee.Geometry.Point(point['geometry']['coordinates'])
            
            # Create 512x512 patch around point
            patch_region = point_geom.buffer(2560)  # 512 pixels * 10m resolution / 2
            
            temporal_images = []
            
            # Extract images for different months
            for month_idx in range(temporal_frames):
                target_month = target_months[month_idx % len(target_months)]
                
                start_date = f"{year}-{target_month:02d}-01"
                end_date = f"{year}-{target_month:02d}-28"
                
                # Get best image for this month
                collection = self.create_sentinel2_collection(region_name, start_date, end_date)
                
                # Filter to patch region and get median composite
                patch_collection = collection.filterBounds(patch_region)
                
                if patch_collection.size().getInfo() > 0:
                    # Get median composite to reduce noise
                    composite = patch_collection.median()
                    
                    # Apply cloud masking and compute indices
                    composite = self.mask_clouds_and_shadows(composite)
                    composite = self.compute_ndwi_and_indices(composite)
                    
                    # Clip to patch region
                    patch_image = composite.clip(patch_region)
                    
                    temporal_images.append({
                        'image': patch_image,
                        'month': target_month,
                        'date_range': f"{start_date}_to_{end_date}"
                    })
            
            if len(temporal_images) >= 2:  # Minimum temporal frames
                patches.append({
                    'patch_id': f"{region_name}_patch_{i:03d}",
                    'region': region_name,
                    'center_coords': point['geometry']['coordinates'],
                    'temporal_sequence': temporal_images,
                    'geometry': patch_region
                })
        
        print(f"Successfully created {len(patches)} temporal patches for {region_name}")
        return patches
    
    def download_patch_to_drive(self, patch_info: Dict, download_bands: List[str] = None):
        """
        Download patch temporal sequence to Google Drive
        
        Args:
            patch_info: Patch information dictionary
            download_bands: Bands to download (default: RGB + NIR + NDWI)
        """
        
        if download_bands is None:
            download_bands = ['B2', 'B3', 'B4', 'B8', 'NDWI']  # Blue, Green, Red, NIR, NDWI
        
        patch_id = patch_info['patch_id']
        region = patch_info['region']
        
        # Create patch directory
        patch_dir = f"{self.drive_path}/data/raw/{region}/{patch_id}"
        os.makedirs(patch_dir, exist_ok=True)
        
        # Download each temporal frame
        for frame_idx, temporal_frame in enumerate(patch_info['temporal_sequence']):
            image = temporal_frame['image']
            month = temporal_frame['month']
            
            # Select bands for download
            download_image = image.select(download_bands)
            
            # Export parameters
            export_params = {
                'image': download_image,
                'description': f"{patch_id}_frame_{frame_idx:02d}_month_{month:02d}",
                'folder': f"WaterBodyResearch/data/raw/{region}/{patch_id}",
                'scale': 10,  # 10m resolution
                'region': patch_info['geometry'],
                'maxPixels': 1e9,
                'fileFormat': 'GeoTIFF'
            }
            
            # Start export task
            task = ee.batch.Export.image.toDrive(**export_params)
            task.start()
            
            print(f"Started download: {patch_id}_frame_{frame_idx:02d}")
        
        # Save patch metadata
        metadata = {
            'patch_id': patch_id,
            'region': region,
            'center_coordinates': patch_info['center_coords'],
            'temporal_frames': len(patch_info['temporal_sequence']),
            'download_bands': download_bands,
            'download_date': datetime.now().isoformat()
        }
        
        metadata_path = f"{patch_dir}/metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def fetch_jrc_water_masks(self, region_name: str) -> ee.Image:
        """
        Fetch JRC Global Surface Water masks for region
        
        Args:
            region_name: Name of study region
            
        Returns:
            JRC water occurrence image
        """
        
        regions = self.get_study_regions()
        region_geometry = regions[region_name]['geometry']
        
        # Get JRC Global Surface Water
        jrc_water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
        
        # Select relevant bands
        water_occurrence = jrc_water.select('occurrence').clip(region_geometry)
        water_seasonality = jrc_water.select('seasonality').clip(region_geometry)
        
        return {
            'occurrence': water_occurrence,
            'seasonality': water_seasonality
        }
    
    def run_complete_data_acquisition(self):
        """
        Run complete data acquisition pipeline for all study regions
        """
        
        print("Starting complete data acquisition pipeline...")
        print("=" * 60)
        
        regions = self.get_study_regions()
        
        for region_name, region_info in regions.items():
            print(f"\nProcessing {region_info['name']}...")
            print(f"Target patches: {region_info['target_patches']}")
            print(f"Temporal frames: {region_info['temporal_frames']}")
            
            # Extract temporal patches
            patches = self.extract_temporal_patches(region_name, year=2023)
            
            # Download patches to Drive
            for patch in patches:
                self.download_patch_to_drive(patch)
                time.sleep(1)  # Avoid rate limiting
            
            print(f"Completed {region_name}: {len(patches)} patches queued for download")
        
        print("\n" + "=" * 60)
        print("Data acquisition pipeline completed!")
        print("Check Google Drive for downloaded files.")
        print("Note: Downloads may take 30-60 minutes to complete.")


def setup_gee_authentication():
    """
    Setup Google Earth Engine authentication for Colab
    """
    
    print("Setting up Google Earth Engine authentication...")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted successfully")
    except ImportError:
        print("⚠ Not running in Colab - Drive mounting skipped")
    
    # Authenticate Earth Engine
    try:
        import ee
        ee.Authenticate()
        ee.Initialize()
        print("✓ Google Earth Engine authenticated and initialized")
    except Exception as e:
        print(f"✗ GEE authentication failed: {e}")
        print("Please run: ee.Authenticate() manually")
    
    return True


def generate_sample_gee_query():
    """
    Generate sample GEE query for manual testing
    """
    
    sample_query = """
# Sample Google Earth Engine Query for Sundarbans Data
# Copy and paste this into GEE Code Editor for manual verification

// Define Sundarbans region
var sundarbans = ee.Geometry.Rectangle([88.0, 21.5, 89.5, 22.5]);

// Create Sentinel-2 collection
var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(sundarbans)
  .filterDate('2023-11-01', '2023-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .select(['B2', 'B3', 'B4', 'B8']);

// Compute NDWI
var addNDWI = function(image) {
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  return image.addBands(ndwi);
};

// Apply NDWI computation
var collectionWithNDWI = collection.map(addNDWI);

// Get median composite
var composite = collectionWithNDWI.median();

// Visualize
Map.centerObject(sundarbans, 10);
Map.addLayer(composite, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'RGB');
Map.addLayer(composite.select('NDWI'), {min: -1, max: 1, palette: ['red', 'yellow', 'blue']}, 'NDWI');

// Print collection info
print('Collection size:', collection.size());
print('Date range:', collection.aggregate_min('system:time_start'), 'to', collection.aggregate_max('system:time_start'));
"""
    
    return sample_query


# Example usage for Colab
if __name__ == "__main__":
    # Setup authentication
    setup_gee_authentication()
    
    # Create data acquisition system
    data_fetcher = SentinelDataAcquisition()
    
    # Print sample GEE query
    print("Sample GEE Query:")
    print("=" * 50)
    print(generate_sample_gee_query())
    
    # Run data acquisition (uncomment when ready)
    # data_fetcher.run_complete_data_acquisition()