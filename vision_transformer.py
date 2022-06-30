import math
import torch
from torch.nn import functional as F

from utils import seed_everything

class PatchLayer():
    '''
    Cut an input image into a set of patches

    Args:
    patch_size: size of the quardatic patches
    '''
    def __init__(self,num_channels,patch_size) -> None:
        super().__init__()
        self.patch_size = patch_size
    def __call__(self,image,flatten_patches = True):
        '''
        Args:
        image:      input image (shape: [b,c,h,w])

        Returns:
        patches:    tensor of patches 
                    (shape: [b, h'*w', c*p_h*p_w] or [b, h'*w', c, p_h, p_w])
        '''
        # https://discuss.pytorch.org/t/creating-nonoverlapping-
        # patches-from-3d-data-and-reshape-them-back-to-the-image/51210/5

        # (b, c, h', w, p_h*p_w)
        patches = image.unfold(2, self.patch_size, self.patch_size)
        # (b, c, h', w', p_h*p_w, p_h*p_w)
        patches = patches.unfold(3, self.patch_size, self.patch_size)
        # (b, h', w', p_h*p_w, p_h*p_w, c)
        patches = patches.permute(0,2,3,4,5,1)
        # (b, h'*w', c*p_h*p_w)
        if flatten_patches:
            return patches.flatten(3,4).flatten(3,4).flatten(1,2)
        else: 
            return patches.flatten(3,4).flatten(3,4)

def sincos2d_positional_embedding(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

class LearnablePositionalEmbedding(torch.nn.Module):
    '''
    Adds a class token and a learnable positional encoding to the set of 
    patches, where the class token is encoded as the first token

    Args:
    embedding_dim:  dimension of the embedding
    num_patches:    number of patsches
    '''
    def __init__(self,embedding_dim, num_patches) -> None:
        super().__init__()

        self.positional_embedding = torch.nn.Parameter(
            torch.randn(1,1+num_patches,embedding_dim)
        )
        self.class_token = torch.nn.Parameter(torch.randn(1,1,embedding_dim))

    def forward(self,patches):
        '''
        Args:
        patches:    tensor of patches (shape: [b, num_patches, embedding_dim])

        Returns:
        patches:    tensor of embedded patches 
                    (shape: [b, num_patches+1, embedding_dim])
        '''

        b = patches.shape[0]
        class_token = self.class_token.repeat(b,1,1)
        patches = torch.cat([class_token,patches],dim=1)
        patches = patches + self.positional_embedding

        return patches

class MultiHeadAttention(torch.nn.Module):
    '''
    Multi-Head Attention layer. Given a set of queries q, keys k and 
    values v, attention is performed. q, v, k are split along their embedding 
    dimension according to the number of heads and distributed among the
    attention heads. In each head dot product attention is computed. The outputs
    of all heads are concatenated to produce the final output.

    Args:
    embedding_dim:  dimension of the patch embeddings
    num_heads:      number of attention heads
    dropout:        dropout probability
    '''
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        assert embedding_dim % self.num_heads == 0, 'embedding_dim not divisible by num_heads'

        self.depth = embedding_dim // self.num_heads

        self.wq = torch.nn.Linear(embedding_dim,embedding_dim)
        self.wk = torch.nn.Linear(embedding_dim,embedding_dim)
        self.wv = torch.nn.Linear(embedding_dim,embedding_dim)

        self.dense = torch.nn.Linear(embedding_dim,embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.attention_weights = None

    def split_heads(self, x):
        """
        Split the last dimension into [num_heads, depth].
        Transpose the result such that the shape is 
        [batch_size, num_heads, len, embedding_dim]

        Args:
        x:  tensor (shape: [b, len, embedding_dim])

        Returns:
            tensor (shape: [b, num_heads, len, embedding_dim])
        """
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return torch.permute(x, [0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask = None):
        """
        Calculate the attention weights 
        A(q,k,v)=\sum(softmax((Q^T K)/sqrt(d))V).

        Args:
            q: query tensor (shape:  [..., len_q, depth_q])
            k: key tensor (shape:  [..., len_k, depth_k])
            v: value tensor (shape:  [..., len_v, depth_v])
            mask: tensor with suitable shape. Optional

        Returns:
           attention_weights : tensor (shape: [..., len_q, len_k])
           attention_logits: tensor (shape: [..., len_q, depth_v])
        """

        # compute score(Q,K) = (Q^T K)/sqrt(d)
        QtK = torch.matmul(q, k.permute(0,1,3,2))  # [..., len_q, len_k]
        # scale matmul_qk
        score = QtK / math.sqrt(self.embedding_dim)

        # add the mask to the scaled tensor.
        if mask is not None:
            score += (mask * -1e9)

        # compute Attention: A(Q,K,V) = sum(softmax(score))
        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = F.softmax(score, dim=-1)
        attention_logits = torch.matmul(attention_weights, v)

        return attention_logits, attention_weights

    def forward(self, v, k, q):
        '''
        Args: 
        v: tensor (shape: [b, len_v, embedding_dim])
        k: tensor (shape: [b, len_k, embedding_dim])
        q: tensor (shape: [b, len_q, embedding_dim])

        Returns:
        attention: tensor (shape: [b, len_q, embedding_dim])
        attention_weights: tensor (shape: [b, num_heads, len_q, len_k])
        '''
        batch_size = q.shape[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # [b, num_heads, len_i, depth]
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # attention_logits (shape: [b, num_heads, len_q, depth_v])
        # attention_weights (shape: [b, num_heads, len_q, len_k])
        attention_logits, self.attention_weights = self.scaled_dot_product_attention(
            q, k, v)

         # [b, len_q, num_heads, depth_v]
        attention_logits = torch.permute(attention_logits, [0, 2, 1, 3]) 

        # [b, len_q, embedding_dim]
        concat_attention = torch.reshape(attention_logits,
                                    (batch_size, -1, self.embedding_dim))
        # [b, len_q, embedding_dim]
        attention = self.dropout(self.dense(concat_attention)) 

        return attention, self.attention_weights

class EncoderLayer(torch.nn.Module):
    '''
    Transformer encoder layer consisting of a Multi-Head Attention layer 
    and a FFN. We use Layer normalization before MHA and FFN to improve 
    gradient flow (http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf)

    Args:
    embedding_dim:  dimension of the patch embeddings
    num_heads:      number of heads for multi head attention
    mlp_hidden:     hidden dimension of the MLPs
    dropout:        probability for dropout    
    '''
    def __init__(
        self, 
        embedding_dim, 
        num_heads, 
        mlp_hidden, 
        dropout = 0.1
    ):
        super().__init__()

        self.mha = MultiHeadAttention(
            embedding_dim=embedding_dim,num_heads=num_heads,dropout=dropout
        )
        self.layernorm_1 = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.layernorm_2 = torch.nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, mlp_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_hidden, embedding_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout)
        )

    def forward(self,x):
        '''
        Args:
        x:              tensor of image embedded patches (shape: [c,p,e])

        Returns:
        out:            tensor of image embedded patches (shape: [c,p,e])
        '''       
        ln1 = self.layernorm_1(x)
        x = x + self.mha(ln1, ln1, ln1)[0]

        ln2 = self.layernorm_2(x)
        x = x + self.mlp(ln2)      
        return x

class VisionTransformer(torch.nn.Module):
    '''
    Vision Transformer Model. The input image is preprocessed and cut into
    patches and embedded into a feature space. A positional encoding is added to
    the embedded patches and the encoded patches are fed to the transformer. A
    MLP head is attached to the transformer for classification.

    Args:
    num_channels:   number of image channels (e.g. 3 for RGB)
    patch_size:     size of the (quadratic) image patches
    num_patches:    total number of patches
    embedding_dim:  dimension of the patch embeddings
    num_heads:      number of heads for multi head attention
    num_layers:     number of transformer encoder layers
    mlp_hidden:     hidden dimension of the MLPs
    num_classes:    number of classes for classification
    dropout:        probability for dropout

    '''
    def __init__(
        self, 
        num_channels,
        patch_size,
        num_patches,
        embedding_dim,
        num_heads, 
        num_layers,
        mlp_hidden, 
        num_classes,
        learnable_embedding = True,
        use_class_token = True,
        dropout = 0.1
    ):
        super().__init__()
        self.learnable_embedding = learnable_embedding
        self.use_class_token = use_class_token
        self.patch_layer = PatchLayer(num_channels,patch_size)
        self.feature_embedding = torch.nn.Linear(num_channels*(patch_size**2), embedding_dim)

        if learnable_embedding:
            # self.positional_embedding = torch.nn.Parameter(
            #     torch.randn(1,1+num_patches,embedding_dim)
            # )
            # self.class_token = torch.nn.Parameter(torch.randn(1,1,embedding_dim))
            self.positional_embedding = LearnablePositionalEmbedding(embedding_dim,num_patches)

        self.transformer = torch.nn.Sequential(
            *[EncoderLayer(
                embedding_dim, num_heads, mlp_hidden, dropout=dropout
            ) 
            for _ in range(num_layers)]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.mlp_head = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, num_classes)
        )

    def forward(self,x):
        '''
        Args:
        x:              input image (shape: [b,c,h,w])

        Returns:
        out:            classification logits (shape: [1,num_classes])
        '''
        if self.learnable_embedding:
            x = self.patch_layer(x)
            x = self.feature_embedding(x)
            #class_token = self.class_token.repeat(x.shape[0],1,1)
            #x = torch.cat([class_token,x],dim=1)
            #x = x + self.positional_embedding
            x = self.positional_embedding(x)
        else:
            x = self.patch_layer(x,flatten_patches=False)
            x = self.feature_embedding(x)
            x = x.flatten(1,2) + sincos2d_positional_embedding(x)
        x = self.transformer(x)
        if self.use_class_token:
            cls = x[:,0]
        else:
            cls = x.mean(dim = 1)
        out = self.mlp_head(cls)
        return out




if __name__ == '__main__':
    seed_everything(42) 
    b,c,h,w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)

    patch_size = 2
    num_patches = (h//patch_size)*(w//patch_size)

    out = VisionTransformer(
        patch_size = patch_size, 
        num_patches = num_patches, 
        num_channels = 3, 
        embedding_dim = 384,
        num_heads = 12, 
        num_layers = 7, 
        mlp_hidden= 384, 
        num_classes=10, 
        learnable_embedding=False,
        use_class_token = False,
        dropout=0.1
    )(x) 
    print(out)