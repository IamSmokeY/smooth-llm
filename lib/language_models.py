import torch
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM:

    """Forward pass through a LLM."""

    def __init__(
        self, 
        model_path, 
        tokenizer_path, 
        conv_template_name,
        device
    ):
        print(f"Initializing LLM with model_path={model_path}, tokenizer_path={tokenizer_path}, conv_template_name={conv_template_name}, device={device}")

        # Language model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True
        ).to(device).eval()
        print("Model loaded and moved to device")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.tokenizer.padding_side = 'left'
        if 'llama-2' in tokenizer_path:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Tokenizer loaded. Pad token: {self.tokenizer.pad_token}")

        # Fastchat conversation template
        self.conv_template = get_conversation_template(
            conv_template_name
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()
        print("Conversation template loaded")

    def __call__(self, batch, max_new_tokens=100):
        print(f"Calling LLM with batch size={len(batch)} and max_new_tokens={max_new_tokens}")

        # Pass current batch through the tokenizer
        batch_inputs = self.tokenizer(
            batch, 
            padding=True, 
            truncation=False, 
            return_tensors='pt'
        )
        print("Batch tokenized")
        
        batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
        batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)

        # Forward pass through the LLM
        try:
            print("Generating outputs from the model")
            outputs = self.model.generate(
                batch_input_ids, 
                attention_mask=batch_attention_mask, 
                max_new_tokens=max_new_tokens
            )
            print("Outputs generated")
        except RuntimeError as e:
            print(f"RuntimeError during model generation: {e}")
            return []

        # Decode the outputs produced by the LLM
        batch_outputs = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        print("Batch outputs decoded")
        
        gen_start_idx = [
            len(self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=True)) 
            for i in range(len(batch_input_ids))
        ]
        batch_outputs = [
            output[gen_start_idx[i]:] for i, output in enumerate(batch_outputs)
        ]
        print("Outputs processed and trimmed")

        return batch_outputs
