"""
Self-supervised pre-training for brain tumor classification.
Implements Masked Autoencoder (MAE) and contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math
from einops import rearrange


class PatchMasking(nn.Module):
    """Random patch masking for MAE pre-training."""
    
    def __init__(self, patch_size: int = 16, mask_ratio: float = 0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Create patches
        patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                           p1=self.patch_size, p2=self.patch_size)
        
        # Random mask
        num_masked = int(num_patches * self.mask_ratio)
        noise = torch.rand(B, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Masked and visible patches
        ids_keep = ids_shuffle[:, num_masked:]
        visible_patches = torch.gather(patches, 1, 
            ids_keep.unsqueeze(-1).expand(-1, -1, patches.size(-1)))
        
        # Binary mask (1 = masked)
        mask = torch.ones(B, num_patches, device=x.device)
        mask[:, :num_patches - num_masked] = 0
        mask = torch.gather(mask, 1, ids_restore)
        
        return visible_patches, mask, ids_restore


class MAEDecoder(nn.Module):
    """Lightweight decoder for MAE reconstruction."""
    
    def __init__(self, embed_dim: int = 512, patch_size: int = 16,
                 in_channels: int = 3, decoder_dim: int = 256, decoder_depth: int = 4):
        super().__init__()
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_dim, nhead=8, dim_feedforward=decoder_dim * 4,
                batch_first=True
            ) for _ in range(decoder_depth)
        ])
        
        self.decoder_pred = nn.Linear(decoder_dim, patch_size ** 2 * in_channels)
        self.patch_size = patch_size
        
        nn.init.normal_(self.mask_token, std=0.02)
    
    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)
        
        # Append mask tokens
        B, L, D = x.shape
        num_patches = ids_restore.size(1)
        mask_tokens = self.mask_token.expand(B, num_patches - L, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        
        # Decoder
        for block in self.decoder_blocks:
            x = block(x)
        
        # Predict patches
        return self.decoder_pred(x)


class MaskedAutoencoderTrainer:
    """
    Masked Autoencoder pre-training.
    Pre-trains on unlabeled brain MRIs by masking and reconstructing patches.
    """
    
    def __init__(self, encoder: nn.Module, patch_size: int = 16,
                 mask_ratio: float = 0.75, decoder_dim: int = 256):
        self.encoder = encoder
        self.patch_masking = PatchMasking(patch_size, mask_ratio)
        self.decoder = MAEDecoder(
            embed_dim=encoder.vit_encoder.embed_dim if hasattr(encoder, 'vit_encoder') else 512,
            patch_size=patch_size,
            decoder_dim=decoder_dim
        )
        self.patch_size = patch_size
    
    def get_parameters(self):
        """Get parameters for optimizer."""
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        return [p for p in params if p.requires_grad]
    
    def train_step(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single pre-training step."""
        # Mask patches
        visible_patches, mask, ids_restore = self.patch_masking(images)
        
        # Encode
        if hasattr(self.encoder, 'cnn'):
            cnn_features = self.encoder.cnn(images)
            patch_embed = self.encoder.patch_embed(cnn_features)
            encoded, _ = self.encoder.vit_encoder(patch_embed)
        else:
            encoded = self.encoder(images)
        
        # Decode
        pred = self.decoder(encoded, ids_restore)
        
        # Target patches
        target = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                          p1=self.patch_size, p2=self.patch_size)
        
        # Loss on masked patches only
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        
        return {'loss': loss, 'pred': pred, 'mask': mask}


class ContrastivePretrainer:
    """
    Contrastive learning pre-training (SimCLR-style).
    Learns representations by maximizing similarity of augmented views.
    """
    
    def __init__(self, encoder: nn.Module, projection_dim: int = 128,
                 temperature: float = 0.5):
        self.encoder = encoder
        self.temperature = temperature
        
        # Projection head
        if hasattr(encoder, 'fusion'):
            in_dim = encoder.fusion.output_dim
        else:
            in_dim = 512
        
        self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, projection_dim)
        )
    
    def get_parameters(self):
        params = list(self.encoder.parameters()) + list(self.projector.parameters())
        return [p for p in params if p.requires_grad]
    
    def train_step(self, images1: torch.Tensor, images2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Contrastive training step with two augmented views."""
        # Encode both views
        if hasattr(self.encoder, 'model'):
            out1 = self.encoder.model(images1, return_features=True)
            out2 = self.encoder.model(images2, return_features=True)
            z1 = self.projector(out1['fused_features'])
            z2 = self.projector(out2['fused_features'])
        else:
            z1 = self.projector(self.encoder(images1))
            z2 = self.projector(self.encoder(images2))
        
        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # NT-Xent Loss
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        
        B = z1.size(0)
        labels = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z.device)
        
        # Mask diagonal
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, -float('inf'))
        
        loss = F.cross_entropy(sim, labels)
        
        return {'loss': loss, 'similarity': sim.detach()}


def pretrain_ssl(model: nn.Module, dataloader, config: Dict,
                 device: torch.device) -> nn.Module:
    """Pre-train model using self-supervised learning."""
    method = config.get('ssl', {}).get('method', 'mae')
    epochs = config.get('ssl', {}).get('pretrain_epochs', 50)
    lr = config.get('ssl', {}).get('pretrain_lr', 1e-4)
    
    if method == 'mae':
        trainer = MaskedAutoencoderTrainer(model)
    else:
        trainer = ContrastivePretrainer(model)
    
    optimizer = torch.optim.AdamW(trainer.get_parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            images = batch['image'].to(device)
            
            if method == 'mae':
                output = trainer.train_step(images)
            else:
                # Create two augmented views (assumes augmentation in dataloader)
                output = trainer.train_step(images, images)
            
            loss = output['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        print(f"Pretrain Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model
