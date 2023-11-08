import os
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from options import Option
from data_utils.dataset import load_data
from model.model import Model
from utils.util import build_optimizer, save_checkpoint, setup_seed
from utils.loss import triplet_loss, rn_loss
from utils.valid import valid_cls

from omegaconf import OmegaConf
import sys 
sys.path.append("./taming_transformers")
from taming_transformers.main import  instantiate_from_config
from data_utils import patch_replaced
from  data_utils.dataset import ScribbleExtendTrainSet, ScribbleTrainSet
import torchvision
from utils.util import load_checkpoint

def train():
    # train_data, sk_valid_data, im_valid_data = load_data(args,dataset_type="scribble")
    train_data, sk_valid_data, im_valid_data = load_data(args,scribble=True)
    
    TAMING_ROOT_PATH = "taming_transformers/"
    VQGAN_CKPT_PATH = TAMING_ROOT_PATH + "logs/idea3/configs/2020-11-20T12-54-32-project.yaml"
    vqgan = instantiate_from_config(OmegaConf.load(VQGAN_CKPT_PATH).model)
    vqgan.cuda()
    # print(vqgan)
    model = Model(args)
    model = model.cuda()
    
    #load best_ckkpt
    if args.load is not None:
        checkpoint = load_checkpoint(args.load)
        cur = model.state_dict()
        new = {k: v for k, v in checkpoint['model'].items() if k in cur.keys()}
        cur.update(new)
        model.load_state_dict(cur)

    # batch=15, lr=1e-5 / batch=30, lr=2e-5
    optimizer = build_optimizer(args, model)

    train_data_loader = DataLoader(
        train_data, args.batch, num_workers=1, drop_last=True, shuffle=True)  # batch=15

    start_epoch = 0
    accuracy = 0
    
    writer = SummaryWriter('./log')

    for i in range(start_epoch, args.epoch):
        print('------------------------train------------------------')
        epoch = i + 1
        model.train()
        torch.set_grad_enabled(True)

        start_time = time.time()
        num_total_steps = args.datasetLen // args.batch

        for index, (sk, im, sk_neg, im_neg, sk_label, im_label, _, _) in enumerate(train_data_loader):
            # prepare data
            sk = torch.cat([sk, sk_neg])
            im = torch.cat([im, im_neg])
            sk, im = sk.cuda(), im.cuda()
            torchvision.utils.save_image(torchvision.utils.make_grid(torch.cat([sk,im],dim=0)),f"logs/sk_im_{index}.jpg")
            # prepare rn truth
            target_rn = torch.cat((torch.ones(sk_label.size()), torch.zeros(sk_label.size())), dim=0) #(2b)
            target_rn = torch.clamp(target_rn, 0.01, 0.99).unsqueeze(dim=1) #(2b,1)
            target_rn = target_rn.cuda()

            # calculate feature
            cls_fea, rn_scores = model(sk, im) #cls_fea.shape=(4b, 768), rn_scores.shape=(2b,1)
            # writer.add_graph(model,input_to_model=(sk,im))
            
            # loss
            # The initial value of losstri should be around 1.00.
            losstri = triplet_loss(cls_fea, args) * 2 #'*2 ' loss weight = 2, Ret Loss
            # The initial value of lossrn should be around 1.00.
            lossrn = rn_loss(rn_scores, target_rn) * 4 #loss weitght = 4, rn loss
            loss = losstri + lossrn

            # vqgan replace generation
            sk_fea, im_fea = model.encode(sk, im)
            #batch = batch_size instead of previous 4*batch_size
            batch = sk_fea.size(0)//2
            patch_size = sk_fea.size(-1)
            c = sk_fea.size(1)
            
            #(2b, patch_size^2, channels, )
            sk_fea = sk_fea.view(2*batch, c, patch_size*patch_size).transpose(1,2)
            im_fea = im_fea.view(2*batch, c, patch_size*patch_size).transpose(1,2)
            
            sk_1 = sk[:1]
            im_1 = im[:1]
            im_2 = im[1:]
            torchvision.utils.save_image(torchvision.utils.make_grid(torch.cat([sk_1,im_1,im_2])),f"logs/sk_im_mixed.jpg")
            sk_11_fea, im_1_fea=model.encode(sk_1,im_1)
            sk_12_fea, im_2_fea =model.encode(sk_1,im_2)
            sk_cs=torch.nn.functional.cosine_similarity(sk_11_fea,sk_12_fea)
            im_cs=torch.nn.functional.cosine_similarity(im_1_fea,im_2_fea)
            
            
            print(torch.nn.functional.cosine_similarity(im_fea[0],im_fea[1]))
            max_indices = patch_replaced.fea_sorted_similarity(sk_fea, im_fea,to1=True)
            im_replaced = patch_replaced.generate_patch_replaced_im_1to1(max_indices,im)
            torchvision.utils.save_image(torchvision.utils.make_grid(im_replaced),f"logs/sk_im_rep_{index}.jpg")
            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #for debug
            # break
            
            # log
            step = index + 1
            if step % 30 == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime(
                    '%H:%M:%S', time.gmtime(remaining_time))
                print(f'epoch_{epoch} step_{step} eta {remaining_time}: loss:{loss.item():.3f} '
                      f'tri:{losstri.item():.3f} rn:{lossrn.item():.3f}')

        # if epoch >= 10:
        if epoch >= 5:
            print('------------------------valid------------------------')
            # log
            map_all, map_200, precision_100, precision_200 = valid_cls(
                args, model, sk_valid_data, im_valid_data)
            print(
                f'map_all:{map_all:.4f} map_200:{map_200:.4f} precision_100:{precision_100:.4f} precision_200:{precision_200:.4f}')
            # save
            if map_all > accuracy:
                accuracy = map_all
                precision = precision_100
                print("Save the BEST {}th model......".format(epoch))
                save_checkpoint(
                    {'model': model.state_dict(), 'epoch': epoch,
                     'map_all': accuracy, 'precision_100': precision},
                    args.save, f'best_checkpoint')


if __name__ == '__main__':
    args = Option().parse()
    print("train args:", str(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.choose_cuda
    print("current cuda: " + args.choose_cuda)
    setup_seed(args.seed)

    train()
