import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import spacy
import os

env_vars= {
    "HF_HUB_CACHE": "./hf_models",
    "HF_HOME": "./hf_models",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    # "HF_DATASETS_OFFLINE": "1",
}

os.environ.update(env_vars)


class Detector:
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(self.device)


    @staticmethod
    def get_entity_from_query(query: str) -> str:
        pass #TODO: Implement a method to extract the entity from the query using for example spacy or any other NLP library

    @torch.inference_mode()
    def detect(self, image: Image.Image, query: str, box_threshold=0.4, text_threshold=0.3):
        entity: str = self.get_entity_from_query(query)
        inputs = self.processor(images=image, text=[entity], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )
        
        #TODO: Finish implementing the logic to process the results and return the detected bounding boxes and labels