import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest 
from peft import PeftConfig
import os


base_model_path = "/home/your_model_path"
lora_path = "/home/your_lora_path"

print("Loading model with vLLM...")

# Option 1: If your LoRA has already been merged into the base model
# llm = LLM(
#     model=base_model_path,
#     tensor_parallel_size=1,  
#     gpu_memory_utilization=0.8,  
#     max_model_len=4096,  
# )

# Option 2: Directly load the LoRA adapter
llm = LLM(
    model=base_model_path,
    enable_lora=True,
    max_loras=1,
    max_lora_rank=64,  
    tensor_parallel_size=2,  
    gpu_memory_utilization=0.8,
    max_model_len=4096,
)

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=300,
    stop=None,  
)

input_path = "../data/input_json/Amazon_edgeset.json"
output_path = "../data/output_json/Amazon_output.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Processing {len(data)} items...")

# Prepare prompts for batch inference
prompts = []
for item in data:
    instruction = item["instruction"]
    input_text = item["input"]
    prompt = f"{instruction}\n\n{input_text}\n\nResponse:"
    prompts.append(prompt)

# Batch inference
print("Starting batch inference...")
if os.path.exists(lora_path):
    outputs = llm.generate(
        prompts, 
        sampling_params,
        lora_request=LoRARequest("hedge_lora", 1, lora_path)  
    )
else:
    outputs = llm.generate(prompts, sampling_params)

# Process outputs
generated = []
print("Processing outputs...")
for i, output in enumerate(tqdm(outputs)):
    response = output.outputs[0].text.strip()
    
    item_out = {
        "id": data[i]["id"],
        "instruction": data[i]["instruction"], 
        "input": data[i]["input"],
        "output": response
    }
    generated.append(item_out)

# Save results
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(generated, f, indent=2, ensure_ascii=False)

print(f"Done! Saved {len(generated)} outputs to {output_path}")
print(f"Speed improvement: Processed {len(data)} items in batch mode!")