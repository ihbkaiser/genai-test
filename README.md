# Generate Consistent Characters
This project demonstrates how to generate consistent characters using the Fluxv1-dev model. The experiments were conducted on a system equipped with 2x NVIDIA RTX 4090 GPUs, each with 24GB VRAM.
## Installation
1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/ihbkaiser/genai-test
cd genai-test
```
2. Install `ai-toolkit` repository:
```bash
git clone https://github.com/ostris/ai-toolkit
cd ai-toolkit
git submodule update --init --recursive
cd ..
```
3. Install the required dependencies:
```pip install -r requirements.txt```
### Training
1. Place the images of your character in the `input_img` directory.
2. Run the training script:```python train.py```
### Testing
1. Update the `test.py` file to specify the path to your fine-tuned weights. Modify the following line:
```python
pipeline.load_lora_weights(".", weight_name="ihbv1.safetensors")
```
2. Customize your prompt then run the testing script:```python test.py```
