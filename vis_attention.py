import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from vision_transformer import VisionTransformer
from data_augmentations import get_transform
import cv2
from vision_transformer import MultiHeadAttention 

checkpoint_path = os.path.join('checkpoint_384_noautoaugment_nomixcut.pt')
dataset_path = os.path.join(os.sep,'Volumes','Storage','datasets','cifar')

def load_checkpoint(file_path):
    checkpoint = torch.load(file_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

def make_prediction(img):
    x = transform(img)
    x.size()
    with torch.no_grad():
        model.eval()
        logits = model(x.unsqueeze(0))
        idx = F.softmax(logits,dim=1).argmax().item()
    
    idx_to_class = {v:k for k,v in ds.class_to_idx.items()}
    pred_class = idx_to_class[idx]
    return pred_class

def get_attention_map(img):

    #extract attention weights from attention layers
    attention_layers = [
        module.attention_weights.squeeze() for module in model.modules() 
        if isinstance(module, MultiHeadAttention)
    ]
    # shape: [n_layers,n_heads,n_patches,n_patches]
    att_mat = torch.stack(attention_layers, dim=0)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(-1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = np.zeros(((aug_att_mat.size(0),)+img.size))

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())

    # resample and gather masks
    joint_attentions[0] = aug_att_mat[0]
    layer_attn = joint_attentions[0, 0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask[0] = cv2.resize(layer_attn / layer_attn.max(), img.size)

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

        layer_attn = joint_attentions[n, 0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask[n] = cv2.resize(layer_attn / layer_attn.max(), img.size)
   
    return mask

def plot_attention_maps(original_img, att_map, pred_class):
    ncols = math.ceil(len(att_map+1)/3)
    fig, axs = plt.subplots(ncols=ncols,nrows=3, figsize=(8, 8))
    fig.suptitle(f'Prediction: {pred_class}')
    plt.axis('off')
    axs[0,0].set_title('Image')
    axs[0,0].imshow(original_img)
    axs[0,0].set_axis_off()
    for layer,ax in enumerate(axs.flat[1:len(att_map)+1]):       
        ax.set_title(f'Attention Map, Layer {layer}')
        _ = ax.imshow(original_img)
        _ = ax.imshow(att_map[layer], cmap="inferno", alpha=0.6)
        ax.set_axis_off()
    plt.show()


model = VisionTransformer(
    patch_size = 4, 
    num_patches = 64,
    embedding_dim = 384,  
    num_channels = 3, 
    num_heads = 12, 
    num_layers = 7, 
    mlp_hidden= 384, 
    num_classes=10, 
    learnable_embedding=True,
    use_class_token = True,
    dropout=0.0
)

load_checkpoint(checkpoint_path)
_,transform = get_transform()
ds = torchvision.datasets.CIFAR10(dataset_path,train = False, download= True)
image = ds[10][0]
pred_class = make_prediction(image)
att_map = get_attention_map(image)
plot_attention_maps(image, att_map, pred_class)
