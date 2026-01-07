from transformers import GPT2LMHeadModel, AutoTokenizer
import torch

def load_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Resize token embeddings to accommodate new special tokens
    tokenizer = AutoTokenizer.from_pretrained("./saved_tokenizer")
    model.resize_token_embeddings(len(tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device