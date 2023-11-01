import torch
import numpy as np
import einops 
import torchvision
from model.rn import cos_similar

def patch2im(patch_index,im, patch_size=16):
    '''
    im: (c, w, h)
    patch_index: (1,1)
    return: (c, patch_size, patch_size)
    '''
    # print(patch_index.shape, im.shape, patch_size)
    # print(patch_index)
    # print(patch_index[0].item()*patch_size)
        
    return im[:, \
        patch_index[0]*patch_size:(patch_index[0]+1)*patch_size, \
        patch_index[1]*patch_size:(patch_index[1]+1)*patch_size]
    
def patch_match_1to1(im, indices, patch_size=16):
    '''
        im: (c,w,h)
        indices: (patch_size^2, 1), [1]=(patch_index)
        1 to 1
    '''
    x = []
    for i in indices:
        patch_index = np.unravel_index(i,(im.size(-1)//patch_size, im.size(-1)//patch_size))
        item = patch2im(patch_index, im, patch_size).unsqueeze(0)
        x.append(item)
    x= torch.cat(x,dim=0)
    return x
        
def patch_match(im, indices,patch_size=16):
    '''
        im: (b,c,w,h)
        indices: (m,im.shape.len)
        1 to N
    '''
    # print(im.shape)
    # x = torch.zeros((0,)+tuple(im.shape[1:]))
    # print(x)
    x = None
    for i in indices:
        patch_index = np.unravel_index(i[1],(im.size(-1)//patch_size,im.size(-1)//patch_size))
        item = patch2im(patch_index, im[i[0]], patch_size).unsqueeze(0)
        # print(item.shape)
        if x is None:
            x=item
        else:
            x= torch.cat([x, item])
    return x

def fea_process(sk_fea, im_fea, upsample=None):
    if upsample is not None:
        return upsample(sk_fea, im_fea)
    return sk_fea, im_fea

def patch_similarity(sk_fea, im_fea):
    # token_fea = einops.rearrange(token_fea,"b d h w -> b d (h w)") #token_fea = token_fea.view(batch, 768, 14, 14)
    # sk_fea, im_fea = fea_process(token_fea[sk.size(0)-1], token_fea[sk.size(0):])
    sk_fea, im_fea = fea_process(sk_fea, im_fea)
    print(sk_fea.shape, im_fea.shape)
    cos_scores = cos_similar(sk_fea, im_fea)
    print(cos_scores.shape)
    # np.savetxt("./logs/cos_scores",cos_scores.cpu()[0])

    return cos_scores

def sort_patch_similarity(cos_scores):
    # print(cos_scores.argsort(0).shape,cos_scores.argsort(0))
    # print(torch.argmax(einops.rearrange(cos_scores,"a b c -> b (a c)")))
    b = einops.rearrange(cos_scores,"a b c -> b (a c)")
    # a_th sketch's patches' similarity to c_th image's patch.
    
    # print(cos_scores.shape,cos_scores)

    max_indices = torch.empty((0,2), dtype=int)
    print(b)
    print(max_indices)

    for i in b:
        max_indices_item = torch.argmax(i)
        # print(i.shape)
        new = np.unravel_index(max_indices_item.cpu(),(cos_scores.shape[0],cos_scores.shape[2]))
        # print(torch.Tensor(new))
        max_indices = torch.cat((max_indices, torch.tensor(new, dtype=torch.int).unsqueeze(0)), 0)
        # print(max_indices)
        
        # print(np.unravel_index(b.values, (3, 196)))
        np.savetxt("./logs/max_indices",max_indices)
    return max_indices

def fea_sorted_similarity(sk_fea, im_fea):
    cos_scores = patch_similarity(sk_fea, im_fea)
    max_indices = sort_patch_similarity(cos_scores)
    return max_indices

def generate_patch_replaced_im(max_indices, im):
    '''
        max_indices: (patch_size^2)->(b,i), [0]=batch_num, [1]=flattened index, 16x16-> 196
        im: (b, c, h, w)
    '''
    # print(max_indices.shape, im.shape)
    im_replaced = patch_match(im,max_indices,16)
    # print(im_replaced.shape)

    im_replaced = torchvision.utils.make_grid(im_replaced,14,padding=0).to("cuda")
    # print(im_replaced.shape)
    return im_replaced
    
def generate_patch_replaced_im_1to1(max_indices, im):
    '''
        max_indices: (b,1,1), [1]=batch_num, [2]=flattened index, 16x16-> 196
        im: (b, c, h, w)
    '''
    x=[]
    for i in range(max_indices.size(0)):
        im_replaced = patch_match_1to1(im[i],max_indices[i],16)
        im_replaced = torchvision.utils.make_grid(im_replaced,14, padding=0).to("cuda")
        torchvision.utils.save_image(im_replaced,f"logs/im{i}.jpg")
        x.append(im_replaced)
    torch.cat(x,dim=0)
    return x

def cat_sk_im(sk, sk_index, im_replaced):
    im_replaced_sketch = torch.cat([im_replaced.unsqueeze(0),(sk[0].unsqueeze(0).to("cuda"))])
    print(im_replaced_sketch.shape)
    im_replaced_sketch = torchvision.utils.make_grid(im_replaced_sketch)

    torchvision.utils.save_image(im_replaced_sketch,f"logs/replaced{sk_index}.jpg")
    return