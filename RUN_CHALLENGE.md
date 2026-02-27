# Running the MLLM Agent Challenge

## Quick Start

### 1. Setup Environment
```bash
# On Leonardo login node
cd /leonardo_scratch/large/usertrain/a08trc0d/AI_winter_school_vision_challenge
source /leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge/pyenvs/MLLM_challenge/bin/activate
```

### 2. Request Interactive GPU Node
```bash
srun -A tra26_minwinsc -p boost_usr_prod -N 1 --gres=gpu:1 -t 01:00:00 --mem=64GB --pty bash
```

### 3. Activate Environment and Set Offline Mode
```bash
source /leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge/pyenvs/MLLM_challenge/bin/activate
cd /leonardo_scratch/large/usertrain/a08trc0d/AI_winter_school_vision_challenge

export HF_HUB_CACHE="/leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge/hf_models"
export HF_HOME="/leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge/hf_models"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

### 4. Run the Challenge

**Test with a few samples (recommended first):**
```bash
python run_challenge.py --limit 5 --dataset all
```

**Process specific dataset:**
```bash
python run_challenge.py --dataset evqa
python run_challenge.py --dataset amber
python run_challenge.py --dataset docvqa
```

**Process all datasets (full challenge):**
```bash
python run_challenge.py --dataset all
```

### 5. Evaluate Results
```bash
python utils/final_score.py \
    --input_path_evqa ./predictions/evqa_predictions.json \
    --input_path_docvqa ./predictions/docvqa_predictions.json \
    --input_path_amber_disc ./predictions/amber_disc_predictions.json
```

## Output Structure

Predictions are saved in the `./predictions/` directory:
- `evqa_predictions.json` - EVQA dataset predictions
- `amber_disc_predictions.json` - Amber Discriminative predictions
- `docvqa_predictions.json` - DocVQA predictions
- `all_predictions.json` - Combined predictions

Each prediction file contains:
```json
[
    {
        "data_id": "<dataset_name>_<idx>",
        "prediction": "model's answer",
        "ground_truths": "expected answer(s)"
    },
    ...
]
```

## Architecture

### Components:

1. **routing.py**: Router that classifies questions and selects appropriate tool
   - Uses Qwen2.5-VL with structured output (Pydantic validation)
   - Returns: `{"tools": "...", "chain_of_thought": "..."}`

2. **pipeline.py**: Main pipeline that executes the selected tool and generates final answer
   - Calls router
   - Executes tool (RAG_grounding, Extract_data_form_image, or Discriminate)
   - Generates final answer using LLM with tool results

3. **utils/tools/retriever.py**: RAG tool for EVQA
   - Retrieves Wikipedia articles from pre-computed index
   - Uses `dataset_image_ids` to lookup relevant documents

4. **utils/tools/ocr.py**: OCR tool for DocVQA
   - Extracts text from document images
   - Uses EasyOCR

5. **utils/tools/object_detector.py**: Detection tool for Amber
   - Detects objects in images
   - Returns presence/absence of requested objects

6. **run_challenge.py**: Main execution script
   - Loads all three datasets
   - Processes each sample through the pipeline
   - Saves predictions in required format

## Troubleshooting

**Module not found errors:**
- Make sure you activated the correct virtual environment
- Check that all packages are installed: `pip list`

**Network/download errors:**
- Ensure environment variables for offline mode are set
- Verify cached models exist in HF_HUB_CACHE

**Out of memory:**
- Request more memory: `--mem=128GB`
- Process in smaller batches with `--limit`

**Tool-specific errors:**
- Check individual tool imports work: `python -c "from utils.tools.retriever import Retriever"`
- Verify data paths in tools match your setup

## Tips for Best Scores

1. **Router Accuracy**: Ensure the router correctly classifies question types
2. **RAG Context**: Clean and format retrieved documents well
3. **Detection Thresholds**: Tune confidence thresholds for Amber
4. **OCR Quality**: Ensure OCR extraction is complete and accurate
5. **Final Answer Generation**: Provide good prompts for the LLM to ground answers on tool results
