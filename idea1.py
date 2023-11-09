# %%
import os
import torch
from options import Option
from data_utils.dataset import load_data_test
from model.model import Model
from utils.util import setup_seed, load_checkpoint
import torchvision
import einops

# %%
args = Option().parse(jupyter=True)
args.load = "./checkpoints/sketchy_ext/best_checkpoint.pth"
args.batch = 2
args.valid_shrink_sk=6000
args.valid_shrink_im=10

print("test args:", str(args))

#%%
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.ap import calculate
from tqdm import tqdm
import time

os.environ["CUDA_VISIBLE_DEVICES"] = args.choose_cuda
print("current cuda: " + args.choose_cuda)
setup_seed(args.seed)

#%%
# prepare data
sk_valid_data, im_valid_data = load_data_test(args)

# prepare model
model = Model(args)
print(model)
model = model.half()

if args.load is not None:
    checkpoint = load_checkpoint(args.load)

cur = model.state_dict()
new = {k: v for k, v in checkpoint['model'].items() if k in cur.keys()}
print(new)
cur.update(new)
model.load_state_dict(cur)

if len(args.choose_cuda) > 1:
    model = torch.nn.parallel.DataParallel(model.to('cuda'))
model = model.cuda()

#%%
# model.eval()
# torch.set_grad_enabled(False)

# print('loading image data')
# sk_dataload = DataLoader(sk_valid_data, batch_size=args.test_sk, num_workers=args.num_workers, drop_last=False)
# print('loading sketch data')
# im_dataload = DataLoader(im_valid_data, batch_size=args.test_im, num_workers=args.num_workers, drop_last=False)

# dist_im = None
# all_dist = None

#%%
# for i, (sk, sk_label) in enumerate(tqdm(sk_dataload)):
#         #sk.shape=(20,3,224,224)
#         print(i)
#         if i == 0:
#             all_sk_label = sk_label.numpy()
#         else:
#             all_sk_label = np.concatenate((all_sk_label, sk_label.numpy()), axis=0)

#         sk_len = sk.size(0)
#         sk = sk.cuda()
#         #debug
#         print(sk[0].shape)
#         # cv2.imwrite(f"./logs/sk-{i}",sk[0].cpu().numpy())
#         if i==0:
#             grid_sk = torchvision.utils.make_grid(sk)
#             torchvision.utils.save_image(grid_sk,f"./logs/sk.jpg")
        
#         sk, sk_idxs = model(sk, None, 'test', only_sa=True)#sk.shape=(20,192,768)
#         for j, (im, im_label) in enumerate(tqdm(im_dataload)):
#             if i == 0 and j == 0:
#                 all_im_label = im_label.numpy()
#             elif i == 0 and j > 0:
#                 all_im_label = np.concatenate((all_im_label, im_label.numpy()), axis=0)

#             im_len = im.size(0)
#             im = im.cuda()
#             im, im_idxs = model(im, None, 'test', only_sa=True)

#             sk_temp = sk.unsqueeze(1).repeat(1, im_len, 1, 1).flatten(0, 1).cuda() #(400,197,768) #?difference
#             im_temp = im.unsqueeze(0).repeat(sk_len, 1, 1, 1).flatten(0, 1).cuda() #(400,197,768)
            
#             if args.retrieval == 'rn':
#                 feature_1, feature_2 = model(sk_temp, im_temp, 'test')
#             #? when retrieval == 'sa'
#             if args.retrieval == 'sa':
#                 feature_1, feature_2 = torch.cat((sk_temp[:, 0], im_temp[:, 0]), dim=0), None

#             # print(feature_1.size())    # [2*sk*im, 768] #2 means sk and im cls
#             # print(feature_2.size())    # [sk*im, 1]

#             if args.retrieval == 'rn':
#                 if j == 0:
#                     dist_im = - feature_2.view(sk_len, im_len).cpu().data.numpy()  # 1*args.batch
#                 else:
#                     dist_im = np.concatenate((dist_im, - feature_2.view(sk_len, im_len).cpu().data.numpy()), axis=1)
#             if args.retrieval == 'sa':
#                 dist_temp = F.pairwise_distance(F.normalize(feature_1[:sk_len * im_len]),
#                                                 F.normalize(feature_1[sk_len * im_len:]), 2)
#                 if j == 0:
#                     dist_im = dist_temp.view(sk_len, im_len).cpu().data.numpy()
#                 else:
#                     dist_im = np.concatenate((dist_im, dist_temp.view(sk_len, im_len).cpu().data.numpy()), axis=1)

#         if i == 0:
#             all_dist = dist_im
#         else:
#             all_dist = np.concatenate((all_dist, dist_im), axis=0)
#         print(all_dist.shape)
#         #all_dist.shape=(all_sk_label.size, all_im_label.size)
#     # print(all_sk_label.size, all_im_label.size)     # [762 x 1711] / 2
# class_same = (np.expand_dims(all_sk_label, axis=1) == np.expand_dims(all_im_label, axis=0)) * 1
# # print(all_dist.size, class_same.size)     # [762 x 1711] / 2

#%%
# print(class_same.shape)
# print(class_same)
# np.savetxt("./logs/all_dist",all_dist)
# np.savetxt("./logs/class_same",class_same)

# map_all, map_200, precision100, precision200 = calculate(all_dist, class_same, test=True)
# print(map_all,map_200,precision100,precision200)

#%%
# arg_sort_sim = all_dist.argsort()   # 得到从小到大索引值
# print(arg_sort_sim.shape)
# print(arg_sort_sim)
# np.savetxt("./logs/arg_sort_sim",torch.tensor(arg_sort_sim,dtype=int))

#%%
def patch2im(patch_index,im, patch_size=16):
    '''
    im: (c, w, h)
    patch_index: (2)
    return: (c, patch_size, patch_size)
    '''
    # print(patch_index.shape, im.shape, patch_size)
    # print(patch_index)
    # print(patch_index[0].item()*patch_size)
        
    return im[:, \
        patch_index[0]*patch_size:(patch_index[0]+1)*patch_size, \
        patch_index[1]*patch_size:(patch_index[1]+1)*patch_size]
    
def patch_match(im, indices,patch_size=16):
    '''
        im: (b,c,w,h)
        indices: (m,im.shape.len)
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

def patch_replace_data(im_index,im, patch_size=16):
    '''
    create an image from the image patch index
    
    im_index: (2), [=b_i, =n_i]
    im: (b,n), [b, n, ......]
    '''
    
    for i,v in enumerate(im_index):
        if i == 0:    
            # print(v)
            im_rtn = patch2im(v, im, patch_size)
            # print(im_rtn.shape)
        else:    
            im_rtn = torch.cat([im_rtn, patch2im(v,im,patch_size)])
    return im_rtn

from data_utils import patch_replaced

# %%
def select_sk_im():
    #sk 0->286, 1->108,3->1503
    sk_index= 0
    # im_index = arg_sort_sim[sk_index,:1]
    im_index = [286]
    return sk_index, im_index
# (sk_tmp, im_tmp) = patch_replace_data(max_indices, im_valid_data[im_index[0],im_index[1],im_index[2],])

sk_index, im_index = select_sk_im()
print(sk_index, im_index)
(sk,_) = sk_valid_data[sk_index]
sk = sk.unsqueeze(0)

tmp = [im_valid_data[i] for i in im_index]
im = [i[0].unsqueeze(0) for i in tmp]
im = torch.cat(im)
print(sk.shape, im.shape)

torchvision.utils.save_image(sk.cuda(),f"./logs/sk-{sk_index}.jpg")

# im_tmp = torchvision.utils.make_grid(im)
torchvision.utils.save_image(torchvision.utils.make_grid(torch.cat([sk.cuda(),im.cuda()])),f"./logs/im_{sk_index}_top_{len(im_index)}.jpg")
print(sk.shape, im.shape)

#%%
from model import rn

print(sk.shape, im.shape)

sk_sa, sk_idxs = model(sk.cuda(), None, 'test', only_sa=True)#sk_sa.shape=(20,192,768)
im_sa, im_idxs = model(im.cuda(), None, 'test', only_sa=True)#im_sa.shape=(20,192,768)


sk_im_sa = torch.cat((sk_sa, im_sa), dim=0)
print(sk_im_sa.shape)
ca_fea = model.ca(sk_im_sa)  # [2b, 197, 768]
# sk_fea,im_fea = model.encode(sk.cuda(),im.cuda())
# ca_fea = torch.cat([sk_fea,im_fea])
print(ca_fea.shape)
cls_fea = ca_fea[:, 0]  # [2b, 1, 768]
token_fea = ca_fea[:, 1:]  # [2b, 196, 768]
print(token_fea.shape)

# token_fea_tmp = einops.rearrange(token_fea, "b (h w) c -> b c h w", h=14)
# print(token_fea_tmp.shape)
# up_fea = model.output4VQGAN(token_fea_tmp)
# print(up_fea.shape)
# up_fea = einops.rearrange(up_fea, "b c h w -> b (h w) c")
# print(up_fea.shape)

batch = token_fea.size(0)

#%%
# token_fea = einops.rearrange(token_fea,"b d h w -> b d (h w)") #token_fea = token_fea.view(batch, 768, 14, 14)
def fea_process(sk_fea, im_fea, upsample=None):
    if upsample is not None:
        return upsample(sk_fea, im_fea)
    return sk_fea, im_fea

cos_scores = patch_replaced.patch_similarity(token_fea[sk.size(0)-1], token_fea[sk.size(0):])
# sk_fea, im_fea = fea_process(token_fea[sk.size(0)-1], token_fea[sk.size(0):])
# print(sk_fea.shape, im_fea.shape)
# cos_scores = rn.cos_similar(sk_fea, im_fea)
print(cos_scores.shape)
# print(cos_scores,file=open("logs/idea1/cos_scores","+w"))
np.savetxt("./logs/idea1/cos_scores",cos_scores.detach().cpu().numpy()[0])

#%%
# # print(cos_scores.argsort(0).shape,cos_scores.argsort(0))
# # print(torch.argmax(einops.rearrange(cos_scores,"a b c -> b (a c)")))
# b = einops.rearrange(cos_scores,"a b c -> b (a c)")
# # a_th sketch's patches' similarity to c_th image's patch.
 
# # print(cos_scores.shape,cos_scores)

# max_indices = torch.empty((0,2), dtype=int)
# print(b)
# print(max_indices)

# for i in b:
#     max_indices_item = torch.argmax(i)
#     # print(i.shape)
#     new = np.unravel_index(max_indices_item.cpu(),(cos_scores.shape[0],cos_scores.shape[2]))
#     # print(torch.Tensor(new))
#     max_indices = torch.cat((max_indices, torch.tensor(new, dtype=torch.int).unsqueeze(0)), 0)
#     # print(max_indices)
    
# # print(np.unravel_index(b.values, (3, 196)))
# np.savetxt("./logs/max_indices",max_indices)
max_indices=patch_replaced.sort_patch_similarity(cos_scores,to1=True)
print(max_indices)

max_indices_2=patch_replaced.sort_patch_similarity(cos_scores)
print(max_indices_2)

# %%
im_replaced_list = patch_replaced.generate_patch_replaced_im_1to1(max_indices,im)
torchvision.utils.save_image(torchvision.utils.make_grid(im_replaced_list),"logs/idea1/cup.jpg")

im_replaced_list_2 = patch_replaced.generate_patch_replaced_im(max_indices_2,im)
torchvision.utils.save_image(torchvision.utils.make_grid(im_replaced_list_2),"logs/idea1/cup2.jpg")

# %%
