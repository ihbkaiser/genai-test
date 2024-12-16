from diffusers import AutoPipelineForText2Image
import torch
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs are available.")

import os
import yaml
with open('config.yml', 'r') as f:
    hf = yaml.safe_load(f)
hf_token = hf['hf'][0]['read_key']
os.environ['HF_TOKEN'] = hf_token
pipeline = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)
pipeline.load_lora_weights(".", weight_name="ihbv1.safetensors")
pipeline.enable_sequential_cpu_offload()
image = pipeline("[trigger] girl with blue eyes, black hair, standing against an orange background", guidance_scale=3.5, height=768).images[0]
image.save("test.png")
