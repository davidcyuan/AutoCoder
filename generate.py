from transformers import GPT2LMHeadModel, AutoTokenizer
import torch

def generate(prompt):
    print("Loading model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained("dcyuan/autocoder")
    tokenizer = AutoTokenizer.from_pretrained("dcyuan/autocoder")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    input_text = f"{tokenizer.bos_token}{prompt}{tokenizer.sep_token}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=100,
            # temperature=temperature,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,  # or False for greedy
            top_p=0.95,  # nucleus sampling
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract just the code part (after the prompt)
    if tokenizer.sep_token in generated_text:
        code = generated_text.split(tokenizer.sep_token, 1)[1]
    else:
        code = generated_text
    
    return code.strip()