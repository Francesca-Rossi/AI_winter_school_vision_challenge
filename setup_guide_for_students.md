# Guide for Students: Setting Up and Running the Challenge

Welcome to the Zero-Shot MLLM Agent Challenge! Follow this guide to quickly set up your environment, load the model, and perform inference.

Directory containing all the materials: `/leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge`

---

## Step 1: Environment Setup

1. **Environment Creation**:
   - Ensure you have Python 3.9+ installed.
   - Create and activate a virtual environment, then install dependencies from requirements.txt:
     ```bash
     module purge
     module load python
     module load cuda
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     ```

2. **Set Environment Variables**:
   - The script automatically sets the necessary environment variables for offline Hugging Face model usage:
     - `HF_HUB_CACHE`: Path to the cached models.
     - `HF_HOME`: Path to Hugging Face home directory.
     - `TRANSFORMERS_OFFLINE`: Enables offline mode for transformers.
     - `HF_HUB_OFFLINE`: Enables offline mode for Hugging Face Hub.

---

## Step 2: Load the Model

1. **Pre-cached Model**:
   - The Qwen2.5-VL-7B model is pre-cached in the `hf_models/` directory.
   - The script will load the model from this directory.

2. **Processor and Model**:
   - The processor formats inputs (text, images) for the model.
   - The model generates responses based on the inputs.

---

## Step 3: Perform Inference

1. **Run the Script**:
   - Open the `setup_and_inference.ipynb` file.
   - Replace the placeholders in the `image_path` and `question` variables:
     ```python
     image_path = "path_to_your_image.jpg"  # Replace with your image path
     question = "What is shown in this image?"  # Replace with your question
     ```

2. **View the Response**:
   - The script will print the model's response to the console.

---

## Notes

- **Offline Mode**:
  - Ensure the `hf_models/` directory contains the required model files.
  - The script will not download any files from the internet.

- **Image Requirements**:
  - Use a valid image file path for inference.

- **Extensibility**:
  - The script includes placeholders for integrating tools (e.g., object detection, OCR).

---

Good luck with the challenge!