import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Gemma2BInstructModel:

    def __init__(self):
        # Model name
        model_name = "google/gemma-2b-it"
        self.tokenizer, self.model = self.initialize_model(model_name)

    def initialize_model(self, model_name):
        # Tokenizer initialization
        tokenizer = AutoTokenizer.from_pretrained(model_name,truncation=True)
        model = AutoModelForCausalLM.from_pretrained(model_name,device_map='auto')
        return tokenizer, model

    def generate_answer(self, question, context=None):
        # Preparing the input prompt
        prompt = question if context is None else f"{context}\n\n{question}"

        chat = [{ "role": "user", "content": f"{prompt}" },]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=1500)
        response = self.tokenizer.decode(outputs[0])
        answer = response.split('<start_of_turn>')[-1][:-5]
        # Extracting and returning the generated text
        return answer