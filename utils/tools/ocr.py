import easyocr
import numpy as np
from PIL import Image

DATASET_BASE_PATH = '/leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge'

ocr_reader = easyocr.Reader(['en'], model_storage_directory=f'{DATASET_BASE_PATH}/hf_models', download_enabled=False) 

def tool_ocr_extractor(pil_image: Image.Image) -> str:
    """
    Legge il testo da un'istanza PIL.Image e lo restituisce come stringa.
    """
    
    img_array = np.array(pil_image)
    
    results = ocr_reader.readtext(img_array)
    
    #Finish implementing the logic to extract and return the text from the OCR results
    #Each result is a tuple: (bounding_box, text, confidence)
    extracted_text = " ".join(text for _, text, _ in results)
    
    return extracted_text