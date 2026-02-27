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
        """
        Retrieve wiki URLs based on dataset_image_id from Google Lens data
        
        Args:
            dataset_image_id: The image ID to lookup (e.g., "2715027")
        
        Returns:
            Context text from retrieved Wikipedia articles
        """
        # Try both string and integer matching
        # First, try to find matches as string
        matches = self.google_lens_data[self.google_lens_data['dataset_image_id'].astype(str) == str(dataset_image_id)]
        
        # If no matches and dataset_image_id is numeric, try as integer
        if matches.empty:
            try:
                dataset_image_id_int = int(dataset_image_id)
                matches = self.google_lens_data[self.google_lens_data['dataset_image_id'] == dataset_image_id_int]
            except (ValueError, TypeError):
                pass
        
        if matches.empty:
            print(f"Warning: No matches found for image_id {dataset_image_id}")
            print(f"  Available image_ids (first 5): {self.google_lens_data['dataset_image_id'].head().tolist()}")
            print(f"  DataFrame shape: {self.google_lens_data.shape}")
            print(f"  Column dtype: {self.google_lens_data['dataset_image_id'].dtype}")
            return "No context available."
        
        print(f"Found {len(matches)} match(es) for image_id {dataset_image_id}")
        
        # Get the lens_wiki_urls column (it's a string representation of a list)
        wiki_urls_str = matches['lens_wiki_urls'].iloc[0]
        
        # Parse the string representation of the list into an actual list
        import ast
        try:
            wiki_urls = ast.literal_eval(wiki_urls_str)
        except:
            print(f"Warning: Could not parse wiki URLs: {wiki_urls_str}")
            wiki_urls = []
        
        # Limit to top_k URLs
        wiki_urls = wiki_urls[:self.top_k]
        
        print(f"Retrieved {len(wiki_urls)} URLs: {wiki_urls}")
        print(f"Total KB entries: {len(self.wikipedia)}")
        
        # Extract text from the retrieved wiki URLs
        context_parts = []
        for url in wiki_urls:
            if not url:  # Skip empty strings
                continue
                
            if url in self.wikipedia:
                entry = self.wikipedia[url]
                print(entry)  # Debug: print the entry for the URL
                # Handles both plain string and dict with a 'text' field
                if isinstance(entry, str):
                    context_parts.append(entry)
                    print(f"Found context for {url} (string, {len(entry)} chars)")
                elif isinstance(entry, dict):
                    texts =  entry.get('section_texts', [])
                    for text in texts:
                        context_parts.append(text)
                        print(f"Found context for {url} (dict, {len(text)} chars)")
            else:
                print(f"Warning: URL {url} not found in knowledge base")
                # Try to find similar keys
                similar_keys = [k for k in list(self.wikipedia.keys())[:5] if 'Smilax' in k or 'bona-nox' in k]
                if similar_keys:
                    print(f"  Similar keys in KB: {similar_keys[:3]}")

        context = "\n\n".join(context_parts)
        
        if not context:
            return "No relevant context found in knowledge base."

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
        

