from torch.optim import AdamW
from tqdm import tqdm
import torch

def train(train_dataloader, val_dataloader, model, device, num_epochs=3, lr=1e-4):
    print("Starting training on ", device)

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):

        # training
        model.train()
        progress_bar = tqdm(train_dataloader)
        total_loss = 0
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass (loss computed automatically)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'train loss': loss.item()})
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

        # validiation
        model.eval()
        total_loss = 0
        progress_bar = tqdm(val_dataloader)
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass (loss computed automatically)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                total_loss += loss.item()
                progress_bar.set_postfix({'val_loss': loss.item()})
        avg_loss = total_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_loss:.4f}")