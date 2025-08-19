import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from torch.amp import GradScaler, autocast
import cv2

# --- 1. 定义形态学损失 ---
class MorphologicalLoss(nn.Module):
    def __init__(self, kernel_size=5, alpha=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.padding = kernel_size // 2
        self.kernel = self._create_circular_kernel(kernel_size)
        
    def _create_circular_kernel(self, size):
        kernel = torch.zeros((size, size))
        center = size // 2
        for i in range(size):
            for j in range(size):
                if (i - center)**2 + (j - center)**2 <= center**2:
                    kernel[i, j] = 1
        return kernel.view(1, 1, size, size)
    
    def forward(self, pred, target):
        # 处理预测结果（支持多分类和二分类）
        if pred.shape[1] > 1:
            pred_probs = F.softmax(pred, dim=1)
            pred_foreground = pred_probs[:, 1:].sum(dim=1, keepdim=True)
        else:
            pred_foreground = torch.sigmoid(pred)
        
        # 处理目标（转为one-hot）
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        target_foreground = target_onehot[:, 1:].sum(dim=1, keepdim=True)
        
        # 腐蚀损失
        eroded_pred = self._binary_erosion(pred_foreground)
        eroded_target = self._binary_erosion(target_foreground)
        erosion_loss = F.mse_loss(eroded_pred, eroded_target)
        
        # 膨胀损失
        dilated_pred = self._binary_dilation(pred_foreground)
        dilated_target = self._binary_dilation(target_foreground)
        dilation_loss = F.mse_loss(dilated_pred, dilated_target)
        
        return self.alpha * erosion_loss + (1 - self.alpha) * dilation_loss
    
    def _binary_erosion(self, x):
        return -F.max_pool2d(-x, kernel_size=self.kernel_size, 
                           stride=1, padding=self.padding)
    
    def _binary_dilation(self, x):
        return F.max_pool2d(x, kernel_size=self.kernel_size,
                          stride=1, padding=self.padding)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 将 inputs 转换为二值掩码
        if inputs.shape[1] > 1:  # 多分类任务
            inputs = F.softmax(inputs, dim=1)
            inputs = torch.argmax(inputs, dim=1)
        else:  # 二分类任务
            inputs = torch.sigmoid(inputs)
            inputs = (inputs > 0.5).float()
        # Flatten the inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # Compute intersection
        intersection = (inputs * targets).sum()
        # Compute Dice coefficient
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        # Compute Dice Loss
        return 1 - dice

# 定义 DeepLabv3+ 的 ASPP 模块
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=64, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[0], dilation=atrous_rates[0], bias=False)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[1], dilation=atrous_rates[1], bias=False)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[2], dilation=atrous_rates[2], bias=False)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.conv1x1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 定义 DeepLabv3+ 解码器
class DeepLabDecoder(nn.Module):
    def __init__(self, low_level_channels, num_classes, aspp_out_channels=64):
        super(DeepLabDecoder, self).__init__()
        self.low_level_conv = nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False)
        self.low_level_bn = nn.BatchNorm2d(48)
        self.low_level_relu = nn.ReLU()
        
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(aspp_out_channels + 48, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x, low_level_feat):
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=False)
        low_level_feat = self.low_level_conv(low_level_feat)
        low_level_feat = self.low_level_bn(low_level_feat)
        low_level_feat = self.low_level_relu(low_level_feat)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.decoder_conv(x)
        return x

# 修改后的 UltraLightSegmentation 模型
class UltraLightSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(UltraLightSegmentation, self).__init__()
        print("Initializing UltraLightSegmentation model with MobileNetV3-Small and DeepLabv3+ Decoder")
        self.backbone = mobilenet_v3_small(weights=None)
        self.aspp = ASPP(in_channels=576, out_channels=64)  # MobileNetV3-Small 输出通道数为 576，out_channels要和aspp_out_channels=64一致
        self.decoder = DeepLabDecoder(low_level_channels=16, num_classes=num_classes)  # 低层特征来自 conv1 (16 channels)

    def forward(self, x):
        input_size = x.size()[2:]  # 保存输入分辨率 (512, 640)
        low_level_feat = self.backbone.features[0](x)  # 提取低层特征
        x = self.backbone.features(x)  # 提取高层特征
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x
    
# def adjust_gamma(image, gamma=1.0):
#     # 构建查找表
#     inv_gamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** inv_gamma) * 255 
#                       for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(image, table)

# 自定义数据集类（保持不变）
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 320), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size  # Tuple (height, width)
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        # 检查文件数量是否匹配
        if len(self.images) != len(self.masks):
            raise ValueError(f"Number of images ({len(self.images)}) and masks ({len(self.masks)}) do not match.")

        # 检查文件名是否完全一致
        for img, msk in zip(self.images, self.masks):
            if not img.endswith(('.png', '.jpg', '.jpeg')):
                continue  # 跳过非图片文件
            if msk != img:
                raise ValueError(f"Filename mismatch: image {img} expects mask {img}, got {msk}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        try:
            # 读取图像
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)       
            # 亮度增强：RGB三通道乘以10倍并截断到255
            # image = np.clip(image.astype(np.float32) * 2, 0, 255).astype(np.uint8)
            # image = adjust_gamma(image, gamma=2.0)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f"Error loading {img_path} or {mask_path}: {e}")
            raise e

        # Resize image and mask using cv2
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)

        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # Convert to tensor
        image = transforms.ToTensor()(image)  # Scales to [0, 1] and converts to (C, H, W)
        mask = torch.from_numpy(mask).long()  # Convert mask to tensor, keep as integer for segmentation

        # Apply additional transforms (e.g., normalization) if provided
        if self.transform:
            image = self.transform(image)

        # Validate mask values
        mask_np = mask.numpy()
        unique_values = np.unique(mask_np)
        if not np.all(np.isin(unique_values, [0, 1])):
            print(f"Warning: Mask {mask_path} contains invalid values: {unique_values}")
        if mask_np.max() > 1:
            mask_np = (mask_np / 255).astype(np.uint8)
            mask = torch.from_numpy(mask_np)

        return image, mask
    
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    scaler = GradScaler()
    morph_loss = MorphologicalLoss(kernel_size=5, alpha=0.5).to(device)
    dice_loss = DiceLoss().to(device)
    morph_weight = 0.1  # 初始权重
    dice_weight = 0.3  # Dice Loss 的权重
    best_val_loss = float('inf')
    checkpoint_dir = './model_320'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_morph = 0.0
        running_dice = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(images)
                ce_loss = criterion(outputs, masks)
                morph_loss_value = morph_loss(outputs, masks)
                dice_loss_value = dice_loss(outputs, masks)
                total_loss = ce_loss + morph_weight * morph_loss_value + dice_weight * dice_loss_value

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item() * images.size(0)
            running_ce += ce_loss.item() * images.size(0)
            running_morph += morph_loss_value.item() * images.size(0)
            running_dice += dice_loss_value.item() * images.size(0)

        # 打印训练统计信息
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_ce = running_ce / len(train_loader.dataset)
        epoch_morph = running_morph / len(train_loader.dataset)
        epoch_dice = running_dice / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f} '
              f'(CE: {epoch_ce:.4f}, Morph: {epoch_morph:.4f}, Dice: {epoch_dice:.4f})')

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)  # 验证阶段仅计算CE损失
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        print(f'Validation CE Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f'New best model saved (Loss: {best_val_loss:.4f})')

        # 定期保存检查点
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), 
                       os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth'))

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'final_model.pth'))
    print('Training completed.')

# --- 5. 主程序 ---
if __name__ == '__main__':
    # 初始化设置
    torch.multiprocessing.set_start_method('spawn', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 数据集路径
    train_image_dir = 'C:/Users/admin/Desktop/dataset/train/images'
    train_mask_dir = 'C:/Users/admin/Desktop/dataset/train/masks'
    val_image_dir = 'C:/Users/admin/Desktop/dataset/test/images'
    val_mask_dir = 'C:/Users/admin/Desktop/dataset/test/masks'
    
    # 数据加载
    image_size = (256, 320)
    train_dataset = SegmentationDataset(train_image_dir, train_mask_dir, image_size=image_size)
    val_dataset = SegmentationDataset(val_image_dir, val_mask_dir, image_size=image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = UltraLightSegmentation(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # 参数量统计
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {count_parameters(model):,}")

    # 开始训练
    train_model(model, train_loader, val_loader, criterion, optimizer, 
               num_epochs=300, device=device)