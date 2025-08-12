from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from persona.models.base import BaseLLM

class LLM(BaseLLM):
    def __init__(self, model_id: str, **kwargs):
            if model_id.startswith("hf/"):
                self.model_id = model_id[3:]  # strip "hf/"
            else:
                self.model_id = model_id  # local folder path

            print(f"Loading Hugging Face model: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )

    def get_name(self):
        return self.model_id

    def generate(self, user_prompt, system_prompt="", stop_condition=None, idx=None):
        input_text = f"{system_prompt}\n{user_prompt}" if system_prompt else user_prompt
        output = self.generator(input_text, max_new_tokens=256, do_sample=False)
        return output[0]["generated_text"]

    def extract_text(self, raw_response):
        return [raw_response]

    def add_usage(self, raw_response):
        pass  # No API usage tracking for local models

    def print_usage(self, n):
        print(f"Hugging Face local model used for {n} generations.")
