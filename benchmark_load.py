import os
import os.path as osp
import pandas as pd
import warnings
from typing import Dict, List, Optional, Union
import hashlib
from tqdm import tqdm
import urllib.request
import json
import base64
from PIL import Image
import io

# Dataset configuration
DATASET_CONFIG = {
    'MME': {
        'url': 'https://opencompass.openxlab.space/utils/VLMEval/MME.tsv',
        'md5': 'b36b43c3f09801f5d368627fb92187c3',
        'type': 'Y/N'
    },
    'HallusionBench': {
        'url': 'https://opencompass.openxlab.space/utils/VLMEval/HallusionBench.tsv',
        'md5': '0c23ac0dc9ef46832d7a24504f2a0c7c',
        'type': 'Y/N'
    },
    'POPE': {
        'url': 'https://opencompass.openxlab.space/utils/VLMEval/POPE.tsv',
        'md5': 'c12f5acb142f2ef1f85a26ba2fbe41d5',
        'type': 'Y/N'
    },
    'AMBER': {
        'url': 'https://huggingface.co/datasets/yifanzhang114/AMBER_base64/resolve/main/AMBER.tsv',
        'md5': '970d94c0410916166e0a76ba75da7934',
        'type': 'Y/N'
    },
    'MMBench_DEV_EN': {
        'url': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN.tsv',
        'md5': 'b6caf1133a01c6bb705cf753bb527ed8',
        'type': 'MCQ'
    },
    'SEEDBench2_Plus': {
        'url': 'https://opencompass.openxlab.space/utils/benchmarks/SEEDBench/SEEDBench2_Plus.tsv',
        'md5': 'e32d3216dc4f452b0fe497a52015d1fd',
        'type': 'MCQ'
    },
    'ScienceQA_VAL': {
        'url': 'https://opencompass.openxlab.space/utils/benchmarks/ScienceQA/ScienceQA_VAL.tsv',
        'md5': '96320d05e142e585e7204e72affd29f3',
        'type': 'MCQ'
    },
    'MMMU_TEST': {
        'url': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_TEST.tsv',
        'md5': 'c19875d11a2d348d07e5eb4bdf33166d',
        'type': 'MCQ'
    },
    'OCRVQA_TESTCORE': {
        'url': 'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TESTCORE.tsv',
        'md5': 'c5239fe77db8bdc1f2ad8e55e0d1fe97',
        'type': 'VQA'
    },

    'MathVista_MINI': {
        'url': 'https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv',
        'md5': 'f199b98e178e5a2a20e7048f5dcb0464',
        'type': 'VQA'
    },
    'MathVision': {
        'url': 'https://opencompass.openxlab.space/utils/VLMEval/MathVision.tsv',
        'md5': '93f6de14f7916e598aa1b7165589831e',
        'type': 'VQA'
    },
    'MMDU': {
        'url': 'https://opencompass.openxlab.space/utils/VLMEval/MMDU.tsv',
        'md5': '848b635a88a078f49aebcc6e39792061',
        'type': 'MT'
    },
    'MIA-Bench': {
        'url': 'https://opencompass.openxlab.space/utils/VLMEval/Mia-Bench.tsv',
        'md5': '0b9de595f4dd40af18a69b94d89aba82',
        'type': 'VQA'
    }
}

class DownloadProgressBar(tqdm):
    """Download progress bar"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url: str, filename: str) -> str:
    """Download file"""
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
    except Exception as e:
        warnings.warn(f'Download failed: {e}')
        # Handle huggingface.co download failure
        if 'huggingface.co' in url:
            url_new = url.replace('huggingface.co', 'hf-mirror.com')
            try:
                download_file(url_new, filename)
                return filename
            except Exception as e:
                warnings.warn(f'Mirror download also failed: {e}')
                raise Exception(f'Unable to download {url}')
        else:
            raise Exception(f'Unable to download {url}')
    
    return filename

def md5(file_path: str) -> str:
    """Calculate file MD5"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def decode_base64_to_image_file(base64_str: str, output_path: str):
    """Decode base64 string to image file"""
    # Remove data:image/jpeg;base64, prefix
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    # Decode base64
    image_data = base64.b64decode(base64_str)
    
    # Save image
    with open(output_path, 'wb') as f:
        f.write(image_data)

def read_ok(file_path: str) -> bool:
    """Check if file is readable"""
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

def toliststr(x):
    """Convert input to string list"""
    if isinstance(x, str):
        return [x]
    elif isinstance(x, list):
        return [str(item) for item in x]
    else:
        return [str(x)]

class DatasetProcessor:
    """Dataset processor"""
    
    def __init__(self, data_root: str):
        """Initialize processor"""
        self.data_root = data_root
        
        os.makedirs(self.data_root, exist_ok=True)
        self.img_root = os.path.join(self.data_root, 'images')
        os.makedirs(self.img_root, exist_ok=True)
    
    def download_dataset(self, dataset_name: str) -> str:
        """Download single dataset"""
        if dataset_name not in DATASET_CONFIG:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        config = DATASET_CONFIG[dataset_name]
        url = config['url']
        file_md5 = config['md5']
        
        # Build file path
        file_name = url.split('/')[-1]
        data_path = os.path.join(self.data_root, file_name)
        
        # Check if file exists and MD5 is correct
        if os.path.exists(data_path):
            if md5(data_path) == file_md5:
                print(f"✓ Dataset {dataset_name} already exists and MD5 is correct")
                return data_path
            else:
                print(f"⚠ Dataset {dataset_name} exists but MD5 mismatch, re-downloading")
        
        # Download file
        print(f"Downloading dataset {dataset_name}...")
        download_file(url, data_path)
        
        # Verify MD5
        if md5(data_path) != file_md5:
            raise ValueError(f"Dataset {dataset_name} MD5 verification failed")
        
        print(f"✓ Dataset {dataset_name} downloaded successfully")
        return data_path
    
    def extract_images(self, dataset_name: str, data: pd.DataFrame) -> Dict[str, str]:
        """Extract images from dataset"""
        dataset_img_root = os.path.join(self.img_root, dataset_name)
        os.makedirs(dataset_img_root, exist_ok=True)
        
        image_paths = {}
        
        if 'image' in data.columns:
            print(f"Extracting images from {dataset_name}...")
            for idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Extract {dataset_name} images"):
                index = row['index']
                image_data = row['image']
                
                if pd.isna(image_data):
                    continue
                
                # Process image data
                if isinstance(image_data, str) and len(image_data) > 64:
                    # Assume base64 encoded image
                    image_path = os.path.join(dataset_img_root, f"{index}.jpg")
                    if not read_ok(image_path):
                        try:
                            decode_base64_to_image_file(image_data, image_path)
                        except Exception as e:
                            print(f"⚠ Failed to decode image (index {index}): {e}")
                            continue
                    image_paths[str(index)] = image_path
                elif isinstance(image_data, list):
                    # Handle multiple images case
                    for i, img in enumerate(image_data):
                        if isinstance(img, str) and len(img) > 64:
                            image_path = os.path.join(dataset_img_root, f"{index}_{i+1}.jpg")
                            if not read_ok(image_path):
                                try:
                                    decode_base64_to_image_file(img, image_path)
                                except Exception as e:
                                    print(f"⚠ Failed to decode image (index {index}_{i+1}): {e}")
                                    continue
                            image_paths[f"{index}_{i+1}"] = image_path
        
        print(f"✓ Extracted {len(image_paths)} images")
        return image_paths
    
    def process_dataset(self, dataset_name: str) -> Dict:
        """Process single dataset"""
        print(f"\n=== Processing dataset: {dataset_name} ===")
        
        # Download dataset
        data_path = self.download_dataset(dataset_name)
        
        # Load data
        data = pd.read_csv(data_path, sep='\t')
        print(f"✓ Loaded {len(data)} samples")
        
        # Extract images
        image_paths = self.extract_images(dataset_name, data)
        
        # Prepare results
        config = DATASET_CONFIG[dataset_name]
        result = {
            'dataset_name': dataset_name,
            'dataset_type': config['type'],
            'total_samples': len(data),
            'image_count': len(image_paths),
            'data': data,
            'image_paths': image_paths,
            'columns': list(data.columns)
        }
        
        # Add sample data
        sample_data = []
        for idx, row in data.head(3).iterrows():
            sample = {
                'index': row['index'],
                'question': row.get('question', 'N/A'),
                'answer': row.get('answer', 'N/A')
            }
            
            # Add options (if MCQ type)
            if config['type'] == 'MCQ':
                options = {}
                for col in ['A', 'B', 'C', 'D', 'E']:
                    if col in row and not pd.isna(row[col]):
                        options[col] = row[col]
                if options:
                    sample['options'] = options
            
            # Add image path
            if str(row['index']) in image_paths:
                sample['image_path'] = image_paths[str(row['index'])]
            
            sample_data.append(sample)
        
        result['sample_data'] = sample_data
        
        return result
    
    def process_datasets(self, dataset_names: List[str]) -> Dict[str, Dict]:
        """Process multiple datasets"""
        if not dataset_names:
            raise ValueError("Dataset name list cannot be empty")
        
        # Validate dataset names
        invalid_datasets = [name for name in dataset_names if name not in DATASET_CONFIG]
        if invalid_datasets:
            raise ValueError(f"Unsupported datasets: {invalid_datasets}")
        
        print(f"Starting to process {len(dataset_names)} datasets: {dataset_names}")
        
        results = {}
        
        for dataset_name in dataset_names:
            try:
                result = self.process_dataset(dataset_name)
                results[dataset_name] = result
                print(f"✓ Dataset {dataset_name} processing completed")
            except Exception as e:
                print(f"✗ Failed to process dataset {dataset_name}: {e}")
                results[dataset_name] = None
        
        return results

def process_vlmeval_datasets(dataset_names: List[str], data_root: str) -> Dict[str, Dict]:
    """
    Main function: Process VLMEval datasets
    
    Args:
        dataset_names: List of dataset names, supported datasets include:
            - Y/N type: MME, HallusionBench, POPE, AMBER
            - MCQ type: MMBench_DEV_EN, SEEDBench2_Plus, ScienceQA_VAL, MMMU_TEST
            - VQA type: OCRBench, MathVista_MINI, MathVision, MIA-Bench
            - MT type: MMDU
        data_root: Root directory for data storage
    
    Returns:
        Dictionary containing all dataset processing results
    """
    processor = DatasetProcessor(data_root)
    return processor.process_datasets(dataset_names)


if __name__ == "__main__":

    print("=== Example 1: Process single dataset ===")
    result = process_vlmeval_datasets(['MMBench_DEV_EN'], data_root='/media/raid/workspace/jiangkailin/data_and_ckpt/dataset/cache/vlmeval')
    
    for dataset_name, dataset_result in result.items():
        if dataset_result:
            print(f"\nDataset: {dataset_name}")
            print(f"Type: {dataset_result['dataset_type']}")
            print(f"Samples: {dataset_result['total_samples']}")
            print(f"Images: {dataset_result['image_count']}")
            print("First 3 samples:")
            for i, sample in enumerate(dataset_result['sample_data'], 1):
                print(f"  Sample{i}: index={sample['index']}, question={sample['question'][:50]}...")
