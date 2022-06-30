import os
import torch
import torchvision
from data_augmentations import get_transform
from vision_transformer import VisionTransformer
from trainer import Trainer
import multiprocessing
if __name__ == '__main__':

    batch_size = 1024
    test_batch_size = 1024
    num_classes = 10
    epochs = 50
    workers = 1#multiprocessing.cpu_count()
    dataset_path = os.path.join(os.sep,'Volumes','Storage','datasets','cifar')
    checkpoint_path = os.path.join('checkpoint.pt')
    train_transform, test_transform = get_transform()
    train_ds = torchvision.datasets.CIFAR10(dataset_path,train = True, transform=train_transform, download= True)
    test_ds = torchvision.datasets.CIFAR10(dataset_path,train = False, transform=test_transform, download= True)

    micro_train_dataset = torch.utils.data.Subset(
        train_ds,
        torch.linspace(0, batch_size, steps=batch_size).long()
    )
    micro_test_dataset = torch.utils.data.Subset(
        test_ds,
        torch.linspace(0, batch_size, steps=batch_size).long()
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                            shuffle=True, num_workers=workers,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                            shuffle=False, num_workers=workers,pin_memory=True)

    micro_train_loader = torch.utils.data.DataLoader(
        micro_train_dataset, batch_size=batch_size, pin_memory=True, num_workers=workers
    )
    micro_val_loader = torch.utils.data.DataLoader(
        micro_test_dataset, batch_size=batch_size, pin_memory=True, num_workers=workers
    )
    
    model = VisionTransformer(
        patch_size = 4, 
        num_patches = 64,
        embedding_dim = 384,  
        num_channels = 3, 
        num_heads = 12, 
        num_layers = 7, 
        mlp_hidden= 384, 
        num_classes=10, 
        learnable_embedding=False,
        use_class_token = True,
        dropout=0.0
    )
    trainer = Trainer(
        model = model,
        train_loader = train_loader,
        val_loader = test_loader,
        epochs = epochs,
        warmup_epoch= 5,
        max_lr = 1e-3,
        min_lr = 1e-5,
        checkpoint_path = checkpoint_path,
        checkpoint_interval = 10,
        print_interval = 100,
        label_smoothing = 0.1,
        use_cutmix = False,
        use_mixup = False,
        im_size = 32
    )

    trainer.fit()