import os
import torch
import torchvision
from data_augmentations import val_transform,train_transform
from vision_transformer import VisionTransformer
from trainer import Trainer

if __name__ == '__main__':

    batch_size = 128
    num_classes = 10
    epochs = 50
    dataset_path = os.path.join(os.sep,'Volumes','Storage','datasets','cifar')
    checkpoint_path = os.path.join('checkpoint.pt')
    train_ds = torchvision.datasets.CIFAR10(dataset_path,train = True, transform=train_transform, download= False)
    val_ds = torchvision.datasets.CIFAR10(dataset_path,train = False, transform=val_transform, download= False)

    micro_train_dataset = torch.utils.data.Subset(
        train_ds,
        torch.linspace(0, batch_size, steps=batch_size).long()
    )
    micro_val_dataset = torch.utils.data.Subset(
        val_ds,
        torch.linspace(0, batch_size, steps=batch_size).long()
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                            shuffle=True, num_workers=2,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                            shuffle=False, num_workers=2,pin_memory=True)

    micro_train_loader = torch.utils.data.DataLoader(
        micro_train_dataset, batch_size=batch_size, pin_memory=True, num_workers=2
    )
    micro_val_loader = torch.utils.data.DataLoader(
        micro_val_dataset, batch_size=batch_size, pin_memory=True, num_workers=2
    )
    model = VisionTransformer(
        patch_size = 4, 
        num_patches = 64,
        embedding_dim = 256, 
        num_channels = 3, 
        num_heads = 8, 
        num_layers = 6, 
        mlp_hidden= 512, 
        num_classes=10, 
        dropout=0.2
    )

    trainer = Trainer(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        epochs = epochs,
        max_lr = 1e-3,
        max_lr = 1e-5,
        checkpoint_path = checkpoint_path,
        checkpoint_interval = 10,
        print_interval = 100,
        label_smoothing = 0.0,
        use_cutmix = True,
        use_mixup = True,
        im_size = 32
    )

    trainer.fit()