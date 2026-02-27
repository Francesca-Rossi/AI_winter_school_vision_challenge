import torch
import json
import logging
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from routing import RoutingAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    def __init__(self, user_query: str, image_path: str = None, documents=None):
        super().__init__(user_query, image_path, model, processor)
        self.documents = documents or []
    
    def run_rag_grounding(self):
        """
        Execute RAG grounding tool to retrieve information from knowledge base
        
        Args:
            chain_of_thought: The reasoning for using this tool
        
        Returns:
            The answer from the RAG system
        """
        logger.info("[RAG_grounding] Executing RAG grounding tool")
        
    
    def run_extract_data_from_image(self):
        """
        Execute data extraction tool for documents, invoices, receipts
        
        Args:
            chain_of_thought: The reasoning for using this tool
        
        Returns:
            Extracted data from the image
        """
        logger.info("[Extract_data_form_image] Executing data extraction tool")
        # TODO: Implement OCR/data extraction logic here
        return "Data extraction result - Not yet implemented"
    
    def run_discriminate(self):
        """
        Execute discriminate tool to identify presence of elements in image
        
        Args:
            chain_of_thought: The reasoning for using this tool
        
        Returns:
            Boolean result (true/false) for element presence
        """
        logger.info("[Discriminate] Executing discriminate tool")
        # TODO: Implement discrimination/detection logic here
        return "Discrimination result - Not yet implemented"
    
    def pipeline(self):
        """
        Main pipeline that routes the query to the appropriate tool
        
        Returns:
            dict: Contains the selected tool, reasoning, and final answer
        """
        # Step 1: Get routing decision from RoutingAgent
        logger.info("="*60)
        logger.info("STEP 1: Routing the query...")
        logger.info("="*60)
        routing_response = self.route(self.user_query)
        
        # Step 2: Parse the response (already a dict from route method)
        selected_tool = routing_response['tools']
        chain_of_thought = routing_response['chain_of_thought']
        
        logger.info(f"Selected Tool: {selected_tool}")
        logger.info(f"Reasoning: {chain_of_thought}")
        
        # Step 3: Route to appropriate tool based on the response
        logger.info("="*60)
        logger.info("STEP 2: Executing the selected tool...")
        logger.info("="*60)
        
        if selected_tool == 'RAG_grounding':
            answer = self.run_rag_grounding(chain_of_thought)
        elif selected_tool == 'Extract_data_form_image':
            answer = self.run_extract_data_from_image(chain_of_thought)
        elif selected_tool == 'Discriminate':
            answer = self.run_discriminate(chain_of_thought)
        else:
            raise ValueError(f"Invalid tool selected: {selected_tool}")
        
        # Step 4: Return complete result
        result = {
            'question': self.user_query,
            'selected_tool': selected_tool,
            'chain_of_thought': chain_of_thought,
            'answer': answer
        }
        
        logger.info("="*60)
        logger.info("STEP 3: Pipeline complete!")
        logger.info("="*60)
        logger.info(f"Result: {json.dumps(result, indent=2)}")
        
        return result


if __name__ == "__main__":
    # Example usage
    pipeline = Pipeline(user_query="What is the capital of France?")
    result = pipeline.pipeline()
    
    logger.info("="*60)
    logger.info("FINAL RESULT")
    logger.info("="*60)
    logger.info(f"Question: {result['question']}")
    logger.info(f"Tool Used: {result['selected_tool']}")
    logger.info(f"Answer: {result['answer']}")
