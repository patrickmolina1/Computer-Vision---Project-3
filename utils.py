import os
import pathlib
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, mobilenet_v2
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

def rgb_to_mask(label_rgb, colormap):
    h, w = label_rgb.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)  # Default to void (255)
    for idx, color in enumerate(colormap):
        match = np.all(label_rgb == np.array(color).reshape(1, 1, 3), axis=2)
        mask[match] = idx
    return mask

class VOCDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_ids, colormap, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_ids = image_ids
        self.colormap = colormap
        self.transform = transform

        # Add basic transforms for consistent size
        self.basic_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        self.label_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),

        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id + ('.jpg' or '.png' or '.jpeg'))  
        label_path = os.path.join(self.label_dir, image_id + '.png')

        # Load image and label

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        label = np.array(label)
        label = rgb_to_mask(label, self.colormap)
        label = Image.fromarray(label)  # Convert to PIL for transforms

        # Apply basic transforms first (resize to consistent size)

        image = self.basic_transform(image)
        label = self.label_transform(label)

        # Apply additional transformations if specified
        if self.transform:
            image, label = self.transform(image, label)

        # Convert to tensor and normalize image

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        label = torch.from_numpy(np.array(label)).long()

        return image, label


class AugmentTransform:
    def __call__(self, image, label):
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # Random rotation (-10 to 10 degrees)
        angle = random.uniform(-10, 10)
        image = TF.rotate(image, angle)
        label = TF.rotate(label, angle, fill=255)  # Fill void with 255

        # Resize and random crop
        resize_size = (256, 256)
        image = TF.resize(image, resize_size, TF.InterpolationMode.BILINEAR)
        label = TF.resize(label, resize_size, TF.InterpolationMode.NEAREST)

        # Random crop to 224x224
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(224, 224))
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        return image, label
    
# DoubleConv and AttentionGate classes remain the same
class DoubleConv(nn.Module):
    """(Convolution => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels):
        super().__init__()
        self.W_g = nn.Conv2d(gating_channels, in_channels, kernel_size=1)
        self.W_x = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.psi = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        g_conv = self.W_g(g)
        x_conv = self.W_x(x)
        combined = self.relu(g_conv + x_conv)
        attention = self.sigmoid(self.psi(combined))
        return x * attention

class CustomUNet(nn.Module):
    def __init__(self, n_classes=21, dropout_rate=0.3):
        super().__init__()
        # Encoder (ResNet-34)
        resnet = resnet34(pretrained=True)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        # Dropout layers
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Decoder with attention gates
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(256, 256)
        self.decoder1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(128, 128)
        self.decoder2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(64, 64)
        self.decoder3 = DoubleConv(128, 64)

        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.attn4 = AttentionGate(64, 64)
        self.decoder4 = DoubleConv(128, 64)

        # Final output
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)       # (64, 56, 56)
        e2 = self.encoder2(e1)      # (64, 56, 56)
        e2 = self.dropout(e2)
        e3 = self.encoder3(e2)      # (128, 28, 28)
        e3 = self.dropout(e3)
        e4 = self.encoder4(e3)      # (256, 14, 14)
        e4 = self.dropout(e4)
        e5 = self.encoder5(e4)      # (512, 7, 7)
        e5 = self.dropout(e5)

        # Decoder
        d1 = self.up1(e5)          # (256, 14, 14)
        a1 = self.attn1(d1, e4)    # (256, 14, 14)
        d1 = torch.cat([a1, e4], dim=1)  # (512, 14, 14)
        d1 = self.decoder1(d1)     # (256, 14, 14)
        d1 = self.dropout(d1)

        d2 = self.up2(d1)          # (128, 28, 28)
        a2 = self.attn2(d2, e3)    # (128, 28, 28)
        d2 = torch.cat([a2, e3], dim=1)  # (256, 28, 28)
        d2 = self.decoder2(d2)     # (128, 28, 28)
        d2 = self.dropout(d2)

        d3 = self.up3(d2)          # (64, 56, 56)
        a3 = self.attn3(d3, e2)    # (64, 56, 56)
        d3 = torch.cat([a3, e2], dim=1)  # (128, 56, 56)
        d3 = self.decoder3(d3)     # (64, 56, 56)
        d3 = self.dropout(d3)

        d4 = self.up4(d3)          # (64, 112, 112)
        
        # Upsample e1 to match d4's spatial size
        e1_upsampled = F.interpolate(e1, size=d4.shape[2:], mode='bilinear', align_corners=True)
        a4 = self.attn4(d4, e1_upsampled)  # (64, 112, 112)
        d4 = torch.cat([a4, e1_upsampled], dim=1)  # (128, 112, 112)
        d4 = self.decoder4(d4)     # (64, 112, 112)

        # Final upsampling to input size (224x224)
        out = self.final_upsample(d4)  # (64, 224, 224)
        return self.out(out)        # (n_classes, 224, 224)
    
class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        smooth = 1.0
        num_classes = inputs.shape[1]

        # Create a mask for valid pixels (exclude ignore_index)
        valid_mask = (targets != self.ignore_index)

        # Set ignore_index pixels to 0 temporarily (valid values only range from 0 to num_classes-1)
        targets = targets.clone()  # Clone to avoid modifying the original targets
        targets[~valid_mask] = 0  # Temporarily set ignore_index to a valid class index (e.g., 0)

        # One-hot encode the targets
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)

        # Reset ignore_index in the one-hot encoded tensor (set ignored pixels to 0 across all classes)
        targets_one_hot = targets_one_hot * valid_mask.unsqueeze(1)  # Zero out ignore_index pixels

        # Apply softmax to inputs
        inputs = torch.softmax(inputs, dim=1)

        # Compute intersection and union
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)  # Probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
