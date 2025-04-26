import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import asyncio
from utils.judges import KodamaJudge
from typing import List, Dict

# Load model and tokenizer
model_name = "rpeddu/kodama-checkpoint-3"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Create a local model chat API class
class LocalModelAPI:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Combine messages into a single prompt
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"{role}: {content}\n"
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Add repetition penalty to discourage repeated text
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=kwargs.get("temperature", 0.1),
            repetition_penalty=1.2, 
            no_repeat_ngram_size=6, 
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the model's response after the prompt
        response = response[len(prompt):].strip()
        return response

# Initialize judge with local model
local_api = LocalModelAPI(model, tokenizer)
judge = KodamaJudge(model_name, local_api)

# Load dataset
dataset = load_dataset("ScalerLab/JudgeBench", split="gpt")

async def main():
    correct = 0
    total = len(dataset)
    
    for sample in dataset:
        question = sample["question"]
        answer_A = sample["response_A"]
        answer_B = sample["response_B"] 
        true_label = sample["label"]
        
        judgment = await judge.get_judgment(question, answer_A, answer_B)
        predicted_label = judgment["decision"]
        
        print(f"Question ID: {sample['pair_id']}")
        print(f"True label: {true_label}")
        print(f"Predicted: {predicted_label}")
        print("---")
        
        if predicted_label == true_label:
            correct += 1
            
    accuracy = (correct / total) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{total}")

if __name__ == "__main__":
    asyncio.run(main())
