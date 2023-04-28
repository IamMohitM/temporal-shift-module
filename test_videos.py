# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

# Notice that this file has been modified to support ensemble testing

import argparse
from locale import normalize
import time
import os
from PIL import Image

import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.models import TSN
from ops.transforms import *
from torch.nn import functional as F

from brarista_ml.utils.video_utils import VideoUtils

# options
parser = argparse.ArgumentParser(description="TSM testing on the full validation set")

parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')
# for true test

parser.add_argument('--binary_node', "--bn", default=False, action="store_true", help='')
parser.add_argument('--clockwise', default=False, action="store_true", help='')
parser.add_argument('--anticlockwise', default=False, action="store_true", help='')
parser.add_argument('--no_binary_node', "--nbn", default=False, action="store_true", help='')

parser.add_argument('--input_size', type=int, required=False, default=224)
parser.add_argument('--crop_fusion_type', type=str, required=False, default='avg')
parser.add_argument('--img_feature_dim',type=int, required=False,default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default='imagenet')


args = parser.parse_args()
if args.anticlockwise:
    args.clockwise = False

if args.no_binary_node:
    args.binary_node = False

net = TSN(2, 8, "RGB",
                base_model="resnet50",
                consensus_type=args.crop_fusion_type,
                img_feature_dim=args.img_feature_dim,
                pretrain=args.pretrain,
                new_length=1,
                is_shift=True, shift_div=8, shift_place="blockres",
                non_local=False,
                binary_node=args.binary_node)


scale_size = net.scale_size
crop_size = net.crop_size


checkpoint = torch.load(args.weights)
checkpoint = checkpoint['state_dict']

    # base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                'base_model.classifier.bias': 'new_fc.bias',
                }
for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)

net.load_state_dict(base_dict)

cropping = torchvision.transforms.Compose([
            # GroupScale(net.scale_size),
            GroupCenterCrop(net.input_size),
        ])

transforms = torchvision.transforms.Compose([cropping, Stack(roll=False), ToTorchFormatTensor(div=True), GroupNormalize(net.input_mean, net.input_std) ])


net = torch.nn.DataParallel(net.cuda())
net.eval()




video_path = "/home/mohitm/repos/brarista_production/s3_buckets/br-cv-tech/Internal_File_Share/Edge_Case_Data/Rotation_Videos/v3/videos/"
user_ids = ["87fb98df-7fdb-4023-9161-2d72a2d6c33b",
"94578e9c-fcea-405f-b610-9c074e59a441",
"328d869a-ef25-429d-ae5e-76f7ec3b2d0d",
"96a242d2-378a-4f4b-86c9-9767bbf8d8e3",
"869793da-523e-4dd8-9726-72cbe2cb74cc",
"a8aff9e6-98a9-4421-95f0-66401ab07616",
"a2765dd9-9763-4d4e-9787-83c47e57abb4",
"dec0174d-fd6b-40d7-89d4-b1f8fe31874f",
"5e7ee74a-ef96-41ab-b7d7-f3a6ea639498",
"dd5430f4-2b95-400b-a1ce-e4908de5e264",
"7c4b8e58-2d9a-41c1-baca-fd61894b3618",
"0efb297e-5d32-4932-a3d2-1530aabf3649",
"5b16b575-cff2-420c-a07b-e1366ee31ea5",
"19383df3-142a-4358-a8c1-ac743b244044",
'709efac4-5fef-42ab-b310-c374cc3cec8e',
"fde73532-58c7-4e75-a103-1662266c959c",
'22b265ec-1fba-4d20-a355-b60fef07fa3a',
"eca81e1e-d418-419d-b326-8f63ad8ce48f",
"41082c6c-7dce-4ff7-a107-ed4b6df83a2b",
"adb6e516-4833-4635-b486-69b7b998caac",
"51bca862-5131-4492-a1f7-c69a4f834150",
"204b256c-40f6-47ac-b287-21a16dd0c10b",
"9e99396b-3e5e-4928-b6ac-502ae5d13119",
]

video_list = []


def preprocess(video_path):
    vu = VideoUtils(video_path)
    frames = vu.get_frames(rgb=True)
    bins = np.linspace(0, len(frames), 8+1, dtype = int)
    frame_list = [np.random.randint(bins[i], bins[i +1]) for i in range(len(bins)- 1)]

    sampled_frames=[Image.fromarray(frames[ind]) for ind in frame_list]



    process_data = transforms(sampled_frames)

    return process_data
    
def process_list(video_list):
    preds = []
    for v in video_list:
        # print(v)
        input_tensor = preprocess(v)[None]
        output = net(input_tensor).squeeze(axis=1)
        if args.binary_node:
            pred = int(torch.sigmoid(output) > 0.5)
        else:
            pred = torch.argmax(output).item()
        print(f"{v}: {pred}")
        preds.append(pred)
    return torch.tensor(preds)

    
class_type = 'clockwise' if args.clockwise else 'anticlockwise'
if args.clockwise:
    for user in user_ids:
        path = os.path.join(video_path, user)
        vids = os.listdir(path)
        for v in vids:
            video_list.append(os.path.join(path, v))
    classes = torch.zeros(len(video_list), dtype = int)
else:
    anti_cl_path = "/home/mohitm/repos/temporal-shift-module/data/anticlockwise_videos"
    video_list = [os.path.join(anti_cl_path, f) for f in os.listdir(anti_cl_path)]
    classes = torch.ones(len(video_list), dtype=int)

start_time = time.time()
preds = process_list(video_list)
print(f"Total time {time.time() - start_time}")
print(f"preds {preds}")
print(f"Truth {classes}")
print(f"{class_type} Accuracy - {(preds==classes).float().mean()}")





