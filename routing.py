from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from PIL import Image
import json


class ResponseTools(BaseModel):
    tools: Literal['RAG_grounding','Extract_data_form_image','Discriminate'] = Field(description="The tool that the model has selected to answer the question")
    chain_of_thought: str = Field(description="The chain of thought for the model to answer the question")

SYSTEM_PROMPT = """
You are a multimodal RAG agent with access to various tools.

Available toolS: 
- RAG_grounding: A tool that retrieves relevant information from a knowledge base to help answer questions. This will retrive the information from .json and/or .csv files, froma Wikipedia search.
- Extract_data_form_image: A tool specialized in extracting specific data from dense scanned documents, invoices, and receipts.
- Discriminate: A tool that needs to identify the presence of a specific element in an image, its answer can be only boolean (true/false).

Your specific role is to determine which tool needs to be called to answer a given question, and to provide the chain of thought of the model needed in order to understand which is the tool which more fits the user request. You should analyze the question and decide which tool is most appropriate for retrieving the necessary information or extracting relevant data from images. Your response should include the name of the tool you recommend using, as well as a detailed explanation of your reasoning process in the form of a chain of thought.
Your response should be in the following JSON format:
{
    "tools": "the name of the tool that you have selected to answer the question",
    "chain_of_thought": "the chain of thought for the model to answer the question"
}
Do not include any other information in your response apart from the JSON format specified above.
""".strip()
FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "response": {
            "tools": "RAG_grounding",
            "chain_of_thought": "The question is asking for a specific piece of information, which is the capital of France. This type of question typically requires retrieving factual information from a knowledge base or database. Therefore, the most appropriate tool to use in this case would be RAG_grounding, as it is designed to retrieve relevant information from a knowledge base to help answer questions."
        }
    },
    {
        "question": "What is the total amount on this invoice?",
        "response": {
            "tools": "Extract_data_form_image",
            "chain_of_thought": "The question is asking for specific data that is likely to be found on an invoice, such as the total amount. This type of question typically requires extracting specific data from a dense scanned document. Therefore, the most appropriate tool to use in this case would be Extract_data_form_image, as it is specialized in extracting specific data from dense scanned documents, invoices, and receipts."
        }
    },
    {
        "question": "Is there a cat in this image?",
        "response": {
            "tools": "Discriminate",
            "chain_of_thought": "The question is asking for a binary answer (yes or no) regarding the presence of a specific element (a cat) in an image. This type of question typically requires analyzing the content of the image to determine if the specified element is present. Therefore, the most appropriate tool to use in this case would be Discriminate, as it is designed to identify the presence of specific elements in images and provide boolean answers."
        }
    }
]
 
class RoutingAgent:
    def __init__(self,user_query: str, image_path: str = None, model=None, processor=None):
        self.system_prompt = SYSTEM_PROMPT
        self.user_query = user_query
        self.image_path = image_path
        self.model = model
        self.processor = processor
        self.image_bytes=self.get_image_bytes()
    
    def get_image_bytes(self):
        if self.image_path:
            image = Image.open(self.image_path)
            return image

        return None

    def route(self, question: str):
        # Build user content based on whether image is provided
        if self.image_bytes is not None:
            user_content = [
                {"type": "image", "data": self.image_bytes},
                {"type": "text", "text": question}
            ]
        else:
            user_content = [
                {"type": "text", "text": question}
            ]
        
        messages=[
            {"role": "system", "content": self.system_prompt}, 
        ]
        messages+=FEW_SHOT_EXAMPLES
        messages.append({"role": "user", "content": user_content})

        text = self.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, response_format=ResponseTools.model_json_schema()
    )

        # Only pass images if they exist
        images = [self.image_bytes] if self.image_bytes is not None else None
        
        inputs = self.processor(
            text=[text],
            images=images,
            videos=None,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        inputs = inputs.to(self.model.device)

        # Generate the response
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            use_cache=True,
        )

        # Extract only the newly generated tokens (remove input tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Validate JSON using Pydantic to ensure correct structure
        try:
            response_json = json.loads(response_text)
            # Validate with Pydantic - this will raise an error if structure is wrong
            validated = ResponseTools(**response_json)
            # Return the validated dict
            return validated.model_dump()
        except json.JSONDecodeError as e:
            raise ValueError(f"Model returned invalid JSON: {e}\nRaw response: {response_text}")
        except Exception as e:
            raise ValueError(f"Model response doesn't match expected schema: {e}\nRaw response: {response_text}")

if __name__ == "__main__":
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

    routing_agent = RoutingAgent(user_query="Which is the capital of France?", model=model, processor=processor)
    response = routing_agent.route(routing_agent.user_query)
    
    # response is now a validated dict with guaranteed structure
    print("="*60)
    print("ROUTING RESULT (Validated by Pydantic)")
    print("="*60)
    print(json.dumps(response, indent=2))
    print("="*60)
    print(f"\nSelected Tool: {response['tools']}")
    print(f"Chain of Thought: {response['chain_of_thought']}")