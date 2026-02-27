import json
import pandas as pd

class Retriever:
    def __init__(self, top_k):
        self.top_k = top_k
        
        print("Loading KB...")
        wiki_KB_path = './data/evqa/encyclopedic_kb_wiki.json'
        with open(wiki_KB_path, "r") as f:
            self.wikipedia = json.load(f)
        print("KB loaded.")

        self.google_lens_path = './data/evqa/lens_entities.csv'
        self.google_lens_data = pd.read_csv(self.google_lens_path)

    def retrieve(self, dataset_image_id):
        wiki_urls = ... #TODO: implement retrieval of wiki urls based on dataset_image_id from google lens data
        
        context = "" #TODO: extract text from the retrieved wiki urls and concatenate them to form the context

        return context
        

