import argparse
from ast import arg
import os
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
import sys
from models import get_model
from networks.pretrain_model import PretrainModel
import torch.nn as nn
from PIL import Image
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
from dataset_paths import DATASET_PATHS
import random
import shutil
from scipy.ndimage.filters import gaussian_filter

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}





def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1
        temp[temp<thres] = 0

        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc

    return best_thres



def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    out.seek(0)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)


def add_gaussian_noise(img, std):
    if std is None or std <= 0:
        return img
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


def validate(model, loader, find_thres=False):

    with torch.no_grad():
        y_true, y_pred = [], []
        print ("Length of dataset: %d" %(len(loader)))
        for img, label in loader:
            in_tens = img.cuda()

            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # ================== save this if you want to plot the curves =========== #
    # torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ),  'baseline_predication_for_pr_roc_curve.pth' )
    # exit()
    # =================================================================== #

    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        print(f"警告: 数据集中只有 {len(unique_labels)} 个类别，无法计算AP")
        ap = 0.0
    else:
        # Get AP
        try:
            ap = average_precision_score(y_true, y_pred)
        except Exception as e:
            print(f"警告: 计算AP时出错: {e}")
            ap = 0.0

    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0


    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres






# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #




def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "PNG"]):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list





class RealFakeDataset(Dataset):
    def __init__(self,  real_path,
                        fake_path,
                        data_mode,
                        max_sample,
                        arch,
                        jpeg_quality=None,
                        gaussian_sigma=None,
                        gaussian_noise_std=None,
                        random_crop_min_scale=None):

        assert data_mode in ["wang2020", "ours"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_noise_std = gaussian_noise_std
        self.random_crop_min_scale = random_crop_min_scale

        # = = = = = = data path = = = = = = = = = #
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, data_mode, max_sample)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, data_mode, max_sample)
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list


        # = = = = = =  label = = = = = = = = = #

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        transform_ops = []
        if self.random_crop_min_scale is not None:
            min_scale = min(max(self.random_crop_min_scale, 0.1), 1.0)
            transform_ops.append(transforms.RandomResizedCrop(224, scale=(min_scale, 1.0)))
        else:
            transform_ops.append(transforms.CenterCrop(224))
        transform_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
        ])
        self.transform = transforms.Compose(transform_ops)


    def read_path(self, real_path, fake_path, data_mode, max_sample):

        if data_mode == 'wang2020':
            real_list = get_list(real_path, must_contain='0_real')
            fake_list = get_list(fake_path, must_contain='1_fake')
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)


        if max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = 100
                print("not enough images, max_sample falling to 100")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]

        #assert len(real_list) == len(fake_list)

        return real_list, fake_list



    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):

        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma)
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)
        if self.gaussian_noise_std is not None:
            img = add_gaussian_noise(img, self.gaussian_noise_std)

        img = self.transform(img)
        return img, label





if __name__ == '__main__':


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--data_mode', type=str, default=None, help='wang2020 or ours')
    parser.add_argument('--max_sample', type=int, default=1000, help='only check this number of images for both fake/real')

    parser.add_argument('--arch', type=str, default='res50')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')

    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_noise_std', type=float, default=None, help="Std of additive gaussian noise. e.g., 5, 10. Not apply if None")
    parser.add_argument('--random_crop_min_scale', type=float, default=None, help="Enable RandomResizedCrop with (scale,1.0). e.g., 0.8")


    opt = parser.parse_args()


    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)

    if opt.arch.startswith('CLIP'):
        state_dict = torch.load(opt.ckpt, map_location='cpu', weights_only=False)

        if 'model_state_dict' in state_dict:
            print("加载第一阶段CLIP预训练模型")
            model = PretrainModel(
                arch=opt.arch,
                output_dim=128,
                hidden_dim=768,
                batch_norm=True
            )
            model.load_state_dict(state_dict['model_state_dict'])

            classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(model.encoder_dim, 1)
            )

            class TestModel(nn.Module):
                def __init__(self, pretrain_model, classifier):
                    super().__init__()
                    self.pretrain_model = pretrain_model
                    self.classifier = classifier

                def forward(self, x):
                    features = self.pretrain_model.get_encoder_features(x)
                    return self.classifier(features)

            model = TestModel(model, classifier)

        elif 'model' in state_dict:
            print("加载第二阶段CLIP微调模型")

            pretrain_model = PretrainModel(
                arch=opt.arch,
                output_dim=128,
                hidden_dim=768,
                batch_norm=True
            )

            classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(pretrain_model.encoder_dim, 1)
            )

            class FinetuneTestModel(nn.Module):
                def __init__(self, pretrain_model, classifier):
                    super().__init__()
                    self.pretrain_model = pretrain_model
                    self.classifier = classifier

                def forward(self, x):
                    features = self.pretrain_model.get_encoder_features(x)
                    return self.classifier(features)

            model = FinetuneTestModel(pretrain_model, classifier)

            model_state = state_dict['model']
            model.load_state_dict(model_state, strict=False)
        else:
            raise ValueError("未知的CLIP模型格式")
    else:
        model = get_model(opt.arch)
        state_dict = torch.load(opt.ckpt, map_location='cpu', weights_only=False)

        if 'model' in state_dict:
            model_state = state_dict['model']
            model.load_state_dict(model_state)
        else:
            model.fc.load_state_dict(state_dict)

    print ("Model loaded..")
    model.eval()
    model.cuda()

    if (opt.real_path == None) or (opt.fake_path == None) or (opt.data_mode == None):
        dataset_paths = DATASET_PATHS
    else:
        dataset_paths = [ dict(real_path=opt.real_path, fake_path=opt.fake_path, data_mode=opt.data_mode) ]



    all_results = []

    for dataset_path in (dataset_paths):
        set_seed()

        dataset = RealFakeDataset(  dataset_path['real_path'],
                                    dataset_path['fake_path'],
                                    dataset_path['data_mode'],
                                    opt.max_sample,
                                    opt.arch,
                                    jpeg_quality=opt.jpeg_quality,
                                    gaussian_sigma=opt.gaussian_sigma,
                                    gaussian_noise_std=opt.gaussian_noise_std,
                                    random_crop_min_scale=opt.random_crop_min_scale,
                                    )

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
        ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, loader, find_thres=True)

        result = {
            'key': dataset_path['key'],
            'ap': ap,
            'r_acc0': r_acc0,
            'f_acc0': f_acc0,
            'acc0': acc0,
            'r_acc1': r_acc1,
            'f_acc1': f_acc1,
            'acc1': acc1,
            'best_thres': best_thres
        }
        all_results.append(result)

        with open( os.path.join(opt.result_folder,'ap.txt'), 'a') as f:
            f.write(dataset_path['key']+': ' + str(round(ap*100, 2))+'\n' )

        with open( os.path.join(opt.result_folder,'acc0.txt'), 'a') as f:
            f.write(dataset_path['key']+': ' + str(round(r_acc0*100, 2))+'  '+str(round(f_acc0*100, 2))+'  '+str(round(acc0*100, 2))+'\n' )

    if all_results:
        avg_ap = sum([r['ap'] for r in all_results]) / len(all_results)
        avg_r_acc0 = sum([r['r_acc0'] for r in all_results]) / len(all_results)
        avg_f_acc0 = sum([r['f_acc0'] for r in all_results]) / len(all_results)
        avg_acc0 = sum([r['acc0'] for r in all_results]) / len(all_results)
        avg_r_acc1 = sum([r['r_acc1'] for r in all_results]) / len(all_results)
        avg_f_acc1 = sum([r['f_acc1'] for r in all_results]) / len(all_results)
        avg_acc1 = sum([r['acc1'] for r in all_results]) / len(all_results)
        avg_best_thres = sum([r['best_thres'] for r in all_results]) / len(all_results)

        print(f"\n=== 所有数据集平均值 ===")
        print(f"平均 AP: {avg_ap*100:.2f}%")
        print(f"平均 Real Accuracy (0.5阈值): {avg_r_acc0*100:.2f}%")
        print(f"平均 Fake Accuracy (0.5阈值): {avg_f_acc0*100:.2f}%")
        print(f"平均 Overall Accuracy (0.5阈值): {avg_acc0*100:.2f}%")
        print(f"平均 Real Accuracy (最佳阈值): {avg_r_acc1*100:.2f}%")
        print(f"平均 Fake Accuracy (最佳阈值): {avg_f_acc1*100:.2f}%")
        print(f"平均 Overall Accuracy (最佳阈值): {avg_acc1*100:.2f}%")
        print(f"平均最佳阈值: {avg_best_thres:.4f}")

        with open( os.path.join(opt.result_folder,'ap.txt'), 'a') as f:
            f.write('AVERAGE: ' + str(round(avg_ap*100, 2))+'\n' )

        with open( os.path.join(opt.result_folder,'acc0.txt'), 'a') as f:
            f.write('AVERAGE: ' + str(round(avg_r_acc0*100, 2))+'  '+str(round(avg_f_acc0*100, 2))+'  '+str(round(avg_acc0*100, 2))+'\n' )

        with open( os.path.join(opt.result_folder,'acc1.txt'), 'a') as f:
            for result in all_results:
                f.write(result['key']+': ' + str(round(result['r_acc1']*100, 2))+'  '+str(round(result['f_acc1']*100, 2))+'  '+str(round(result['acc1']*100, 2))+'  '+str(round(result['best_thres'], 4))+'\n' )
            f.write('AVERAGE: ' + str(round(avg_r_acc1*100, 2))+'  '+str(round(avg_f_acc1*100, 2))+'  '+str(round(avg_acc1*100, 2))+'  '+str(round(avg_best_thres, 4))+'\n' )

