from jadis.segment import *
import os.path as osp
import numpy as np
import mmseg
import mmcv

from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.apis import init_model, inference_model

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import torch, cv2, tqdm

data_root = '../documents/'
img_dir = 'images'

classes = ('background', 'road_network')
palette = [[0, 0, 0], [255, 255, 255]] # RGB

@DATASETS.register_module()
class MyDataset(BaseSegDataset):
    METAINFO = dict(
        classes=classes,
        palette=palette)
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)

config_file = 'pretrained_paris/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py'
checkpoint_file = 'pretrained_paris/best_mIoU.pth'

model = init_model(config_file, checkpoint_file, device='cuda:1')

if not torch.cuda.is_available():
    model = revert_sync_batchnorm(model)

# clean gpu memory when starting a new evaluation.
torch.cuda.empty_cache()

for map_path in tqdm.tqdm(glob.glob(f'{data_root}{img_dir}/*.*')):
    
    map_name = osp.basename(map_path).split('.')[0]
    output_folder = f'{data_root}predictions/'
    os.makedirs(output_folder, exist_ok = True)

    os.system("rm -rf workshop")
    os.system("rm -rf pred")
    os.makedirs('workshop/')
    os.makedirs('pred/')

    patch_size, border_width = 640, 64
    core_size = patch_size-border_width
    
    map_image = cv2.imread(map_path)
    rows, cols = makeImagePatches(map_image, patches_path = 'workshop', export = True, 
                                  patch_size = patch_size, border_width = border_width)
    
    for path in glob.glob(f'workshop/*.tif'):
                
        patch_name = path.split("/")[-1].split(".")[0]
        result = inference_model(model, path).seg_logits.data.detach().cpu().numpy()
        np.save(f'pred/{patch_name}.npy', result, allow_pickle=True)
        
    reconstitution = np.zeros((result.shape[0], patch_size+(rows*core_size), patch_size+(cols*core_size)))
    
    for row in range(rows):
        for col in range(cols):

            probs = np.load(os.path.join('pred/', str(row) + '_' + str(col) + '.npy'), allow_pickle=True)
         
            pre_row = pre_col = border_width//2
            post_row = post_col = patch_size-(border_width//2)

            if row == 0:
                pre_row = 0
            elif row == rows-1:
                post_row = patch_size
            if col == 0:
                pre_col = 0
            elif col == cols-1:
                post_col = patch_size

            reconstitution[:, pre_row+(row*core_size):(row*core_size)+post_row, 
                           pre_col+(col*core_size):(col*core_size)+post_col] = probs[:, 
                pre_row:post_row, pre_col:post_col]
            
    reconstitution = np.argmax(reconstitution[:map_image.shape[0], :map_image.shape[1]], axis=0)
        
    output = np.zeros((reconstitution.shape[0], reconstitution.shape[1], 3))
    for c, color in enumerate(palette):
        output[reconstitution == c] = color
        
    cv2.imwrite(os.path.join(output_folder, map_name + '.png'), output.astype('uint8'))
