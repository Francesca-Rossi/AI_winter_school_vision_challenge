import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from routing import RoutingAgent

# Step 2: Load the Model and Processor
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

class Pipeline(RoutingAgent):
    def __init__(self, user_query: str, image_path: str = None):
        super().__init__(user_query, image_path, model, processor)
        self.documents=
    
    def pipeline(self):
        response = self.routing_agent.route(self.routing_agent.user_query)
        if response.tools == 'RAG_grounding':
            pass
        elif response.tools == 'Extract_data_form_image':
            pass
        elif response.tools == 'Discriminate':
            pass
        else:
            raise ValueError("Invalid tool selected")
        
