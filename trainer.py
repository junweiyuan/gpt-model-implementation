import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size=32,
        learning_rate=3e-4,
        weight_decay=0.01,
        max_epochs=10,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        warmup_steps=100,
        grad_clip=1.0
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.device = device
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device == 'cuda' else False
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if device == 'cuda' else False
            )
        else:
            self.val_loader = None
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        self.total_steps = len(self.train_loader) * max_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps - warmup_steps
        )
        
        self.current_step = 0
        
    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.optimizer.param_groups[0]['lr'] * (self.current_step / self.warmup_steps)
        return self.optimizer.param_groups[0]['lr']
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
            
            if self.current_step < self.warmup_steps:
                lr = self.optimizer.param_groups[0]['lr'] * (self.current_step / self.warmup_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            self.optimizer.zero_grad()
            
            logits, loss = self.model(x, y)
            
            loss.backward()
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            if self.current_step >= self.warmup_steps:
                self.scheduler.step()
            
            total_loss += loss.item()
            self.current_step += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.get_lr():.6f}'
            })
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        for x, y in tqdm(self.val_loader, desc='Validating'):
            x, y = x.to(self.device), y.to(self.device)
            logits, loss = self.model(x, y)
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.max_epochs):
            print(f'\nEpoch {epoch + 1}/{self.max_epochs}')
            
            train_loss = self.train_epoch()
            print(f'Train Loss: {train_loss:.4f}')
            
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f'Val Loss: {val_loss:.4f}')
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f'New best validation loss: {val_loss:.4f}')
            
        print('\nTraining completed!')
        return self.model
    
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_step': self.current_step,
        }, path)
        print(f'Checkpoint saved to {path}')
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_step = checkpoint['current_step']
        print(f'Checkpoint loaded from {path}')
