import json
import os
import torch
import torch.utils.data
from PIL import Image
from datasets import load_dataset
import csv

DATASET_BASE_PATH = '/leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge'

class AmberDiscDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()


        #amber_disc.json -> contiene id, image, query
        with open(os.path.join(DATASET_BASE_PATH, 'data/amber_disc/amber_disc.json'), 'r') as f:
            self.data = json.load(f)

        self.image_base_path = os.path.join(DATASET_BASE_PATH, 'data/amber_disc/images/')
        
    def __getitem__(self, idx):
        sample = self.data[idx]

        image_path = os.path.join(self.image_base_path, sample['image'])
        sample['image_path'] = image_path
        sample['image'] = Image.open(sample['image_path']).convert('RGB')

        sample['data_id'] = f"AMBER_DISC_{sample['id']}"

        sample['question'] = sample['query']

        return sample
        

    def __len__(self):
        return len(self.data)

class DocVQADataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.data = load_dataset('lmms-lab/DocVQA', 'DocVQA', cache_dir=os.path.join(DATASET_BASE_PATH, './data/docvqa'))['validation']

    def __getitem__(self, idx):
        sample = self.data[idx]

        sample['ground_truths'] = sample['answers']

        sample['data_id'] = f"DOCVQA_{sample['questionId']}"

        return sample
    
    def __len__(self):
        return len(self.data)

class EVQADataset(torch.utils.data.Dataset):
    def __init__(self, ):
        super().__init__()

        self.data, self.header = self.load_csv_data(os.path.join(DATASET_BASE_PATH, './data/evqa/evqa_test.csv'))
        with open(os.path.join(DATASET_BASE_PATH, './data/evqa/evqa_test_images_paths.json'), 'r') as f:
            self.image_paths = json.load(f)
        

    def __getitem__(self, idx):
        sample = self.get_test_question(idx, self.data, self.header)
        image_path = self.image_paths[idx]

        sample['image'] = Image.open(image_path).convert('RGB')
        sample['question'] = sample['question']

        sample['ground_truths'] = sample['answer'].split('|')


        return sample
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod   
    def load_csv_data(test_file):
        test_list = []
        with open(test_file, "r") as f:
            reader = csv.reader(f)
            test_header = next(reader)
            for row in reader:
                try: 
                    # if (row[test_header.index("question_type")] == "automatic" or row[test_header.index("question_type")] == "templated" or row[test_header.index("question_type")] == "multi_answer" or row[test_header.index("question_type")] == "infoseek"): 
                    test_list.append(row)
                except:
                    # print row and line number
                    print(row, reader.line_num)
                    raise ValueError("Error in loading csv data")
        return test_list, test_header

    @staticmethod
    def get_test_question(preview_index, test_list, test_header):
        return {test_header[i]: test_list[preview_index][i] for i in range(len(test_header))}
