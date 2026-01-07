from datasets import load_from_disk
from transformers import AutoTokenizer

def load_mbpp_dataset():
    dataset = load_from_disk("./data/mbpp")
    return dataset

def create_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Add your custom special tokens
    special_tokens_dict = {
        'pad_token': '<PAD>',
        'bos_token': '<BOS>',  # beginning of sequence
        'eos_token': '<EOS>',
        'sep_token': '<SEP>',   # separator between prompt and code
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    
    tokenizer.save_pretrained("./saved_tokenizer")

def encode(text, code, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained("./saved_tokenizer")
    
    # input_ids: <BOS> text <SEP> code <EOS>
    prompt_tokens = tokenizer.encode(text, add_special_tokens=False)
    code_tokens = tokenizer.encode(code, add_special_tokens=False)
    input_ids = ([tokenizer.bos_token_id] +
        prompt_tokens + [tokenizer.sep_token_id] + code_tokens +
        [tokenizer.eos_token_id])
    
    # truncate
    if len(input_ids) > max_length:
        # Calculate how much space we have for code
        # bos (1) + prompt (len) + sep (1) + eos (1) = fixed overhead
        overhead = 1 + len(prompt_tokens) + 1 + 1
        available_for_code = max_length - overhead
        
        # Truncate code tokens
        code_tokens = code_tokens[:available_for_code]
        
        # Rebuild sequence
        input_ids = ([tokenizer.bos_token_id] +
            prompt_tokens + [tokenizer.sep_id] + code_tokens +
            [tokenizer.eos_id])
        
    # pad
    pad_id = tokenizer.convert_tokens_to_ids('<PAD>')
    padding_length = max_length - len(input_ids)
    input_ids = input_ids + [pad_id] * padding_length

    # attention_mask
    pad_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in input_ids]

    # labels: shifted ids
    labels = input_ids[1:] + [pad_id]
    sep_index = labels.index(tokenizer.sep_token_id)
    for i in range(len(labels)):
        if i <= sep_index or labels[i] == tokenizer.pad_token_id:
            labels[i] = -100  # ignore index for loss calculation

    return {
        'input_ids': input_ids,
        'attention_mask': pad_mask,
        'labels': labels
    }

def decode(input_ids):
    tokenizer = AutoTokenizer.from_pretrained("./saved_tokenizer")
    return tokenizer.decode(input_ids)