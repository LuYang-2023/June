import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import os
import time
from tqdm import tqdm
from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD Inference without mask")
parser.add_argument("--model_names", default=['ACM', 'ALCNet','DNANet', 'ISNet', 'RDIAN', 'ISTDU-Net'], nargs='+',  
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--pth_dirs", default=None, nargs='+',  help="checkpoint dir, default=None or ['NUDT-SIRST/ACM_400.pth.tar','NUAA-SIRST/ACM_400.pth.tar']")
parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K'], nargs='+', 
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()

## Set img_norm_cfg
if opt.img_norm_cfg_mean is not None and opt.img_norm_cfg_std is not None:
    opt.img_norm_cfg = dict()
    opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
    opt.img_norm_cfg['std'] = opt.img_norm_cfg_std

def test(): 
    test_set = InferenceSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False, pin_memory=True)
    
    net = Net(model_name=opt.model_name, mode='test').cuda()
    try:
        net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.load_state_dict(torch.load(opt.pth_dir, map_location=device)['state_dict'])
    net.eval()

    scaler = GradScaler()

    with torch.no_grad():
        for idx_iter, (img, size, img_dir) in tqdm(enumerate(test_loader)):
            img = Variable(img).cuda()
            with autocast():
                pred = net.forward(img)
                pred = pred[:,:,:size[0],:size[1]]        
            ### save img
            if opt.save_img == True:
                img_save = transforms.ToPILImage()(((pred[0,0,:,:]>opt.threshold).float()).cpu())
                save_path = os.path.join(opt.save_img_dir, opt.test_dataset_name, opt.model_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                img_save.save(os.path.join(save_path, img_dir[0] + '.png'))
            del img, pred  # 释放显存
            torch.cuda.empty_cache()

    print('Inference Done!')
   
if __name__ == '__main__':
    opt.f = open(os.path.join(opt.save_log, 'test_' + time.strftime("%Y_%m_%d_%H_%M_%S") + '.txt'), 'w')
    if opt.pth_dirs is None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_400.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = os.path.join(opt.save_log, opt.dataset_name, opt.model_name + '_400.pth.tar')
                test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    if dataset_name in pth_dir and model_name in pth_dir:
                        opt.test_dataset_name = dataset_name
                        opt.model_name = model_name
                        opt.train_dataset_name = pth_dir.split('/')[0]
                        print(pth_dir)
                        opt.f.write(pth_dir)
                        print(opt.test_dataset_name)
                        opt.f.write(opt.test_dataset_name + '\n')
                        opt.pth_dir = os.path.join(opt.save_log, pth_dir)
                        test()
                        print('\n')
                        opt.f.write('\n')
        opt.f.close()
