from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class InsuranceAssistant:
    def __init__(self):
        """Step 3: Augmentation & Step 4: Generation"""
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    
    def generate_response(self, question, context):
        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(**inputs, max_length=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)