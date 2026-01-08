import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import os

# Custom Dataset Class
class FlickrDataset(Dataset):
    def __init__(self, df, img_dir, processor):
        self.df = df
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        raw_caption = str(self.df.iloc[idx, 1])
        
        # Path updated to look into the models/ModelFlick30k/Images directory
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        seo_formatted_text = f"SEO Alt Text: {raw_caption}. Keywords: {', '.join(raw_caption.split()[:5])}"
        encoding = self.processor(images=image, text=seo_formatted_text, truncation=True, max_length=80, return_tensors="pt")
        return {k: v.squeeze() for k, v in encoding.items()}

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': input_ids_padded
    }

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    # --- PATH UPDATES FOR NEW STRUCTURE ---
    # Scripts are in src/, so we go up one level (..) to reach data/ and models/
    data_path = os.path.join("..", "data", "captions.txt") 
    img_dir = os.path.join("..", "models", "ModelFlick30k", "Images")
    save_directory = os.path.join("..", "models", "neural_sight_v1")
    # ---------------------------------------
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found! Check your data folder.")
    else:
        df = pd.read_csv(data_path, sep=",", names=["image", "caption"], skiprows=1)
        df = df.groupby('image').head(5).reset_index(drop=True)
        
        dataset = FlickrDataset(df, img_dir, processor)
        train_loader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scaler = torch.amp.GradScaler('cuda')
        
        model.train()
        print(f"Starting training on {device}... Data: {data_path}")

        for epoch in range(1):
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if i % 10 == 0:
                    print(f"Step: {i}, Loss: {loss.item():.4f}")

        os.makedirs(save_directory, exist_ok=True)
        model.save_pretrained(save_directory)
        processor.save_pretrained(save_directory)
        print(f"Model saved to {save_directory}")