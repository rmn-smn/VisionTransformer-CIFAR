import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from data_augmentations import CutMix,MixUp
import warmup_scheduler

class Trainer():
    def __init__(self,
        model,
        train_loader, 
        val_loader,
        epochs,
        warmup_epoch,
        max_lr,
        min_lr,
        weight_decay = 5e-5, 
        checkpoint_path = '.',
        checkpoint_interval = 100,
        print_interval = 100,
        label_smoothing = 0.1,
        use_cutmix = False,
        use_mixup = False,
        im_size = None
    ):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.last_epoch = 0
        self.last_loss = 0
        self.max_iters = len(self.train_loader)*epochs
        self.max_lr = max_lr
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.weight_decay = weight_decay
        # self.optimizer = torch.optim.AdamW(
        #     model.parameters(), max_lr #, weight_decay=weight_decay
        # )
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr = max_lr, betas=(0.9, 0.999), weight_decay=weight_decay
        )
        #self.scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=min_lr)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=min_lr)
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=warmup_epoch, after_scheduler=self.base_scheduler)
        self.criterion  = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.use_cutmix = use_cutmix
        self.use_mixup = use_mixup
        if use_cutmix:
            self.cutmix = CutMix(im_size, beta=1.)
        if use_mixup:
            self.mixup = MixUp(alpha=1.)

        self.stats = {
            'lrs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
            }

    def save_checkpoint(self,epoch,file_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'stats': self.stats,
            }, file_path
        )
        print('checkpoint saved')

    def load_checkpoint(self,file_path):
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.last_epoch = checkpoint['epoch']
        self.stats = checkpoint['stats']
        print('checkpoint loaded')

    def accuracy(self,outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
    def _train(self):
        train_loss = []
        train_acc = []
        lrs = []
        self.model.train()
        for i,(images,labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            preds = self.model(images)

            if self.use_cutmix or self.use_mixup:
                if self.use_cutmix:
                    images, labels, rand_labels, lambda_= self.cutmix((images, labels))
                elif self.use_mixup:
                    if np.random.rand() <= 0.8:
                        images, labels, rand_labels, lambda_ = self.mixup((images, labels))
                    else:
                        images, labels, rand_labels, lambda_ = images, labels, torch.zeros_like(labels), 1.
                loss = self.criterion(preds, labels)*lambda_ + self.criterion(preds, rand_labels)*(1.-lambda_)
            else:
                loss = self.criterion(preds,labels)

            acc = self.accuracy(preds,labels)
            train_loss.append(loss)
            train_acc.append(acc)

            print('\riteration {}/{}  train loss={:.4f}, train acc={:.4f}'.format(i,len(self.train_loader), loss, acc), end='')

            # backprop
            loss.backward()

            # remove gradient from previous passes
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(self.get_lr())
            self.scheduler.step()

        return train_loss,train_acc
        
    @torch.no_grad()
    def _validate(self):
        val_loss = []
        val_acc = []
        self.model.eval()
        for i,(images,labels) in enumerate(self.val_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            preds = self.model(images)

            loss = F.cross_entropy(preds,labels)
            acc = self.accuracy(preds,labels)
            val_loss.append(loss)
            val_acc.append(acc)

            print('\riteration {}/{} val loss={:.4f}, val acc={:.4f}'.format(i,len(self.val_loader), loss, acc), end='')
        return val_loss,val_acc



    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
    def fit(self):
        print('running training')
        for epoch in range(self.last_epoch+1, self.epochs+1):
            print('current epoch: {}'.format(epoch))
            
    
            train_loss,train_acc = self._train()
            self.stats['train_loss'].append(torch.stack(train_loss).mean().item())
            self.stats['train_acc'].append(torch.stack(train_acc).mean().item())
            print('\repoch {} mean train loss={:.4f}, mean train acc={:.4f}'.format(
                    epoch,
                    self.stats['train_loss'][-1], 
                    self.stats['train_acc'][-1]
                ), end=''
            )
            print()

            val_loss,val_acc = self._validate()

            self.stats['val_loss'].append(torch.stack(val_loss).mean().item())
            self.stats['val_acc'].append(torch.stack(val_acc).mean().item())
            print('\repoch {} mean val loss={:.4f}, mean val acc={:.4f}'.format(
                    epoch,
                    self.stats['val_loss'][-1], 
                    self.stats['val_acc'][-1]
                ), end=''
            )
            self.stats['lrs'] = self.get_lr()
            print()

            self.last_epoch = epoch

            if self.last_epoch % self.checkpoint_interval == 0:
                self.save_checkpoint(epoch,self.checkpoint_path)
            #
        plt.title("Training history")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.plot(self.stats['train_loss'],label='train')
        plt.plot(self.stats['val_loss'],label='val')
        plt.legend()
        plt.show()
