import torch
import json
import logging
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from routing import RoutingAgent
from utils.tools.retriever import Retriever
from utils.tools.ocr import tool_ocr_extractor
from utils.tools.object_detector import Detector


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
    def __init__(self, user_query: str, image_path: str = None, data_id: str = None, ground_truth: str = None, documents=None, model=None, processor=None):
        # Use provided model/processor or load from global
        _model = model if model is not None else globals().get('model')
        _processor = processor if processor is not None else globals().get('processor')
        
        super().__init__(user_query, image_path, _model, _processor)
        self.documents = documents or []
        self.data_id = data_id
        self.ground_truth = ground_truth
    
    def run_rag_grounding(self, id_image):
        """
        Execute RAG grounding tool to retrieve information from knowledge base
        
        Args:
            id_image: The image ID for retrieval
        
        Returns:
            The context from the RAG system
        """
        retriever = Retriever(top_k=3)
        logger.info("[RAG_grounding] Executing RAG grounding tool")
        context = retriever.retrieve(id_image)
        return context
    
    def run_extract_data_from_image(self):
        """
        Execute data extraction tool for documents, invoices, receipts
        
        Returns:
            Extracted text from the image
        """
        logger.info("[Extract_data_form_image] Executing data extraction tool")
        image_text = tool_ocr_extractor(self.image_bytes)
        return image_text
    
    def run_discriminate(self):
        """
        Execute discriminate tool to identify presence of elements in image
        
        Returns:
            Detection results as text
        """
        logger.info("[Discriminate] Executing discriminate tool")
        detector = Detector()
        result = detector.detect(self.image_bytes, self.user_query)
        txt_result = ""
        for unit in result:
            txt_result += f"Object found: {unit['label']} with score {unit['score']:.2f}\n"
        if not txt_result:
            txt_result = "No objects detected."
        return txt_result
    
    def generate_final_answer(self, tool_result: str, selected_tool: str):
        """
        Use the LLM to generate a final answer based on the user query and tool result
        
        Args:
            tool_result: The result from the selected tool
            selected_tool: The name of the tool that was used
        
        Returns:
            The final answer from the LLM
        """
        logger.info("="*60)
        logger.info("STEP 3: Generating final answer with LLM...")
        logger.info("="*60)
        
        # Create a prompt for the LLM to answer based on the tool result
        answer_prompt = f"""Based on the following information, answer the user's question concisely and accurately.

User Question: {self.user_query}

Tool Used: {selected_tool}

Tool Result:
{tool_result}

Please provide a clear, direct answer to the user's question based on the tool result above."""

        # Build messages for LLM
        if self.image_bytes is not None:
            user_content = [
                {"type": "image", "data": self.image_bytes},
                {"type": "text", "text": answer_prompt}
            ]
        else:
            user_content = [
                {"type": "text", "text": answer_prompt}
            ]
        
        messages = [
            {"role": "user", "content": user_content}
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

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
            max_new_tokens=512,
            use_cache=True,
        )

        # Extract only the newly generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        final_answer = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        logger.info(f"Final Answer: {final_answer}")
        
        return final_answer.strip()
    
    def pipeline(self):
        """
        Main pipeline that routes the query to the appropriate tool
        
        Returns:
            dict: Contains the formatted result with data_id, prediction, and ground_truths
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
            # Extract image ID from data_id for EVQA
            # data_id format: "evqa_<dataset_image_ids>" or just the dataset_image_ids
            if self.data_id and 'evqa' in self.data_id.lower():
                # Extract the numeric part after "evqa_" or similar
                import re
                match = re.search(r'(\d+)$', self.data_id)
                id_image = match.group(1) if match else self.data_id
            else:
                id_image = self.data_id if self.data_id else self.image_path
            
            logger.info(f"Using image ID for RAG: {id_image}")
            tool_result = self.run_rag_grounding(id_image)
        elif selected_tool == 'Extract_data_form_image':
            tool_result = self.run_extract_data_from_image()
        elif selected_tool == 'Discriminate':
            tool_result = self.run_discriminate()
        else:
            raise ValueError(f"Invalid tool selected: {selected_tool}")
        
        logger.info(f"Tool Result: {tool_result}")
        
        # Step 4: Generate final answer using LLM based on tool result
        final_answer = self.generate_final_answer(tool_result, selected_tool)
        
        # Step 5: Format result according to specification
        result = {
            "data_id": self.data_id if self.data_id else "unknown_0",
            "prediction": final_answer,
            "ground_truths": self.ground_truth if self.ground_truth else ""
        }
        
        logger.info("="*60)
        logger.info("STEP 4: Pipeline complete!")
        logger.info("="*60)
        logger.info(f"Formatted Result: {json.dumps(result, indent=2)}")
        
        return result
    
    @staticmethod
    def process_batch(queries_data: list) -> list:
        """
        Process multiple queries in batch
        
        Args:
            queries_data: List of dicts, each containing:
                - query: str (required)
                - image_path: str (optional)
                - data_id: str (required)
                - ground_truth: str (optional)
        
        Returns:
            List of results in the format:
            [
                {"data_id": "<dataset_name>_<idx>", "prediction": "...", "ground_truths": "..."},
                ...
            ]
        """
        results = []
        
        for idx, query_data in enumerate(queries_data):
            logger.info(f"\nProcessing query {idx + 1}/{len(queries_data)}")
            
            try:
                pipeline = Pipeline(
                    user_query=query_data["query"],
                    image_path=query_data.get("image_path"),
                    data_id=query_data.get("data_id", f"unknown_{idx}"),
                    ground_truth=query_data.get("ground_truth", "")
                )
                result = pipeline.pipeline()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query {idx + 1}: {e}")
                # Add error result
                results.append({
                    "data_id": query_data.get("data_id", f"unknown_{idx}"),
                    "prediction": f"ERROR: {str(e)}",
                    "ground_truths": query_data.get("ground_truth", "")
                })
        
        return results


if __name__ == "__main__":
    import json as json_lib
    
    # Load dataset examples
    DATASET_BASE_PATH = '/leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge/data'
    
    # Load EVQA test data (for RAG_grounding examples)
    import pandas as pd
    evqa_test = pd.read_csv(f'{DATASET_BASE_PATH}/evqa/evqa_test.csv')
    evqa_images = json_lib.load(open(f'{DATASET_BASE_PATH}/evqa/evqa_test_images_paths.json'))
    
    # Example 1: RAG_grounding with EVQA dataset
    logger.info("="*60)
    logger.info("EXAMPLE 1: RAG_grounding with EVQA Dataset")
    logger.info("="*60)
    
    # Get first example from EVQA
    evqa_sample = evqa_test.iloc[0]
    evqa_image_id = evqa_sample['dataset_image_ids']
    evqa_question = evqa_sample['question']
    evqa_answer = evqa_sample['answer']
    evqa_image_path = evqa_images[0]  # First image path
    
    logger.info(f"Image ID: {evqa_image_id}")
    logger.info(f"Question: {evqa_question}")
    logger.info(f"Ground Truth: {evqa_answer}")
    logger.info(f"Image Path: {evqa_image_path}")
    
    pipeline_evqa = Pipeline(
        user_query=evqa_question,
        image_path=evqa_image_path,
        data_id=f"evqa_{evqa_image_id}",
        ground_truth=evqa_answer
    )
    result_evqa = pipeline_evqa.pipeline()
    
    # Example 2: Extract_data_form_image with DocVQA dataset
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 2: Extract_data_form_image with DocVQA")
    logger.info("="*60)
    
    # Load DocVQA dataset example
    from load_datasets import DocVQADataset
    docvqa_dataset = DocVQADataset()
    docvqa_sample = docvqa_dataset[0]  # Get first example
    
    docvqa_question = docvqa_sample['question']
    docvqa_image = docvqa_sample['image']
    docvqa_data_id = docvqa_sample['data_id']
    docvqa_ground_truths = docvqa_sample['ground_truths']
    
    logger.info(f"Data ID: {docvqa_data_id}")
    logger.info(f"Question: {docvqa_question}")
    logger.info(f"Ground Truths: {docvqa_ground_truths}")
    
    pipeline_docvqa = Pipeline(
        user_query=docvqa_question,
        image_path=docvqa_image,  # PIL Image object
        data_id=docvqa_data_id,
        ground_truth=docvqa_ground_truths
    )
    result_docvqa = pipeline_docvqa.pipeline()
    
    # Example 3: Discriminate with EVQA image
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 3: Discriminate with Image Detection")
    logger.info("="*60)
    
    discriminate_question = "Is there a plant in this image?"
    discriminate_image_path = evqa_images[1]  # Use second EVQA image
    
    pipeline_discriminate = Pipeline(
        user_query=discriminate_question,
        image_path=discriminate_image_path,
        data_id="evqa_discriminate_1",
        ground_truth="yes"
    )
    result_discriminate = pipeline_discriminate.pipeline()
    
    # Format all results
    results_list = [result_evqa, result_discriminate]
    
    print("\n" + "="*60)
    print("FINAL OUTPUT (JSON format):")
    print("="*60)
    print(json.dumps(results_list, indent=2))
    
    # Example 4: Batch processing with real dataset
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 4: Batch Processing Multiple EVQA Queries")
    logger.info("="*60)
    
    # Process first 3 EVQA examples
    batch_queries = []
    for idx in range(min(3, len(evqa_test))):
        sample = evqa_test.iloc[idx]
        batch_queries.append({
            "query": sample['question'],
            "image_path": evqa_images[idx],
            "data_id": f"evqa_{sample['dataset_image_ids']}",
            "ground_truth": sample['answer']
        })
    
    # Uncomment to run batch processing:
    # batch_results = Pipeline.process_batch(batch_queries)
    # print("\n" + "="*60)
    # print("BATCH PROCESSING OUTPUT (JSON format):")
    # print("="*60)
    # print(json.dumps(batch_results, indent=2))
    
    logger.info("Batch processing example prepared (uncomment to run)")

