"""
Main script to run the complete MLLM Agent Challenge
Processes EVQA, Amber, and DocVQA datasets and generates predictions
"""

import json
import logging
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from pipeline import Pipeline
from load_datasets import EVQADataset, AmberDiscDataset, DocVQADataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load model once (shared across all samples)
logger.info("Loading Qwen2.5-VL-7B model...")
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(
    model_name,
    padding_side="left",
    trust_remote_code=True
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

logger.info("Model loaded successfully!")


def process_evqa_dataset(limit=None):
    """Process EVQA dataset"""
    logger.info("="*60)
    logger.info("Processing EVQA Dataset")
    logger.info("="*60)
    
    dataset = EVQADataset()
    results = []
    
    num_samples = min(limit, len(dataset)) if limit else len(dataset)
    logger.info(f"Processing {num_samples} EVQA samples...")
    
    for idx in tqdm(range(num_samples), desc="EVQA"):
        try:
            sample = dataset[idx]
            
            # Extract dataset_image_ids for RAG retrieval
            dataset_image_ids = sample.get('dataset_image_ids', '')
            
            # Create pipeline for this sample
            pipeline = Pipeline(
                user_query=sample['question'],
                image_path=None,  # We'll set image_bytes directly
                data_id=f"evqa_{dataset_image_ids}",
                ground_truth='|'.join(sample['ground_truths']),
                model=model,
                processor=processor
            )
            
            # Save image directly
            if 'image' in sample and sample['image'] is not None:
                pipeline.image_bytes = sample['image']
            
            # Store the dataset_image_ids for RAG retrieval
            pipeline.dataset_image_ids = dataset_image_ids
            
            result = pipeline.pipeline()
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing EVQA sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "data_id": sample.get('data_id', f'evqa_{idx}'),
                "prediction": f"ERROR: {str(e)}",
                "ground_truths": '|'.join(sample.get('ground_truths', ['']))
            })
    
    return results


def process_amber_dataset(limit=None):
    """Process Amber Discriminative dataset"""
    logger.info("="*60)
    logger.info("Processing Amber Discriminative Dataset")
    logger.info("="*60)
    
    dataset = AmberDiscDataset()
    results = []
    
    num_samples = min(limit, len(dataset)) if limit else len(dataset)
    logger.info(f"Processing {num_samples} Amber samples...")
    
    for idx in tqdm(range(num_samples), desc="Amber"):
        try:
            sample = dataset[idx]
            
            pipeline = Pipeline(
                user_query=sample['question'],
                image_path=sample.get('image_path'),
                data_id=sample['data_id'],
                ground_truth='',  # Amber doesn't have ground truth in test set
                model=model,
                processor=processor
            )
            
            if 'image' in sample and sample['image'] is not None:
                pipeline.image_bytes = sample['image']
            
            result = pipeline.pipeline()
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing Amber sample {idx}: {e}")
            results.append({
                "data_id": sample.get('data_id', f'amber_{idx}'),
                "prediction": f"ERROR: {str(e)}",
                "ground_truths": ''
            })
    
    return results


def process_docvqa_dataset(limit=None):
    """Process DocVQA dataset"""
    logger.info("="*60)
    logger.info("Processing DocVQA Dataset")
    logger.info("="*60)
    
    dataset = DocVQADataset()
    results = []
    
    num_samples = min(limit, len(dataset)) if limit else len(dataset)
    logger.info(f"Processing {num_samples} DocVQA samples...")
    
    for idx in tqdm(range(num_samples), desc="DocVQA"):
        try:
            sample = dataset[idx]
            
            pipeline = Pipeline(
                user_query=sample['question'],
                image_path=None,  # DocVQA provides PIL image directly
                data_id=sample['data_id'],
                ground_truth='|'.join(sample['ground_truths']) if isinstance(sample['ground_truths'], list) else sample['ground_truths'],
                model=model,
                processor=processor
            )
            
            if 'image' in sample and sample['image'] is not None:
                pipeline.image_bytes = sample['image']
            
            result = pipeline.pipeline()
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing DocVQA sample {idx}: {e}")
            results.append({
                "data_id": sample.get('data_id', f'docvqa_{idx}'),
                "prediction": f"ERROR: {str(e)}",
                "ground_truths": '|'.join(sample.get('ground_truths', ['']))
            })
    
    return results


def save_predictions(results, output_path):
    """Save predictions to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, indent=2, fp=f)
    logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MLLM Agent Challenge')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples per dataset (for testing)')
    parser.add_argument('--dataset', type=str, choices=['evqa', 'amber', 'docvqa', 'all'], default='all', help='Which dataset to process')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Output directory for predictions')
    
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    
    # Process selected datasets
    if args.dataset in ['evqa', 'all']:
        evqa_results = process_evqa_dataset(limit=args.limit)
        save_predictions(evqa_results, os.path.join(args.output_dir, 'evqa_predictions.json'))
        all_results.extend(evqa_results)
    
    if args.dataset in ['amber', 'all']:
        amber_results = process_amber_dataset(limit=args.limit)
        save_predictions(amber_results, os.path.join(args.output_dir, 'amber_disc_predictions.json'))
        all_results.extend(amber_results)
    
    if args.dataset in ['docvqa', 'all']:
        docvqa_results = process_docvqa_dataset(limit=args.limit)
        save_predictions(docvqa_results, os.path.join(args.output_dir, 'docvqa_predictions.json'))
        all_results.extend(docvqa_results)
    
    # Save combined results
    save_predictions(all_results, os.path.join(args.output_dir, 'all_predictions.json'))
    
    logger.info("="*60)
    logger.info("CHALLENGE COMPLETE!")
    logger.info("="*60)
    logger.info(f"Total samples processed: {len(all_results)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("\nTo evaluate, run:")
    logger.info(f"python utils/final_score.py \\")
    logger.info(f"    --input_path_evqa {os.path.join(args.output_dir, 'evqa_predictions.json')} \\")
    logger.info(f"    --input_path_docvqa {os.path.join(args.output_dir, 'docvqa_predictions.json')} \\")
    logger.info(f"    --input_path_amber_disc {os.path.join(args.output_dir, 'amber_disc_predictions.json')}")
