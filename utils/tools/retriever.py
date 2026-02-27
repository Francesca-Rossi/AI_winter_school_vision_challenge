import json
import os
import pandas as pd

DATASET_BASE_PATH = '/leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge'

class Retriever:
    def __init__(self, top_k):
        self.top_k = top_k
        
        print("Loading KB...")
        wiki_KB_path = os.path.join(DATASET_BASE_PATH, './data/evqa/encyclopedic_kb_wiki.json')
        with open(wiki_KB_path, "r") as f:
            self.wikipedia = json.load(f)
        print("KB loaded.")

        self.google_lens_path = os.path.join(DATASET_BASE_PATH, './data/evqa/lens_entities.csv')
        self.google_lens_data = pd.read_csv(self.google_lens_path)

    def retrieve(self, dataset_image_id):
        # TODO 1: Retrieve wiki URLs based on dataset_image_id from Google Lens data
        matches = self.google_lens_data[self.google_lens_data['image_id'] == dataset_image_id]
        wiki_urls = matches['wiki_url'].dropna().tolist()[:self.top_k]
        
        # TODO 2: Extract text from the retrieved wiki URLs and concatenate them to form the context
        context_parts = []
        for url in wiki_urls:
            if url in self.wikipedia:
                entry = self.wikipedia[url]
                # Handles both plain string and dict with a 'text' field
                if isinstance(entry, str):
                    context_parts.append(entry)
                elif isinstance(entry, dict):
                    text = entry.get('text') or entry.get('content') or entry.get('summary', '')
                    if text:
                        context_parts.append(text)

        context = "\n\n".join(context_parts)

        return context

if __name__ == "__main__":
    retriever = Retriever(top_k=3)
    sample_image_id = "2715027"  # Replace with an actual image ID from the dataset
    context = retriever.retrieve(sample_image_id)
    print("Retrieved Context:")
    print(context)

    # def retrieve(self, dataset_image_id):
    #     wiki_urls = ... #TODO: implement retrieval of wiki urls based on dataset_image_id from google lens data
        
    #     context = "" #TODO: extract text from the retrieved wiki urls and concatenate them to form the context

    #     return context
        

