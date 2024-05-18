import os
from segment_anything import  sam_model_registry, SamPredictor
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np

from dataset import SAMDataset
from utils.common import  plot_masks


def infer(data_dir, save_dir, pretrained_model):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = 'vit_h'
    # checkpoint = 'weights/sam_vit_b_01ec64.pth'
    checkpoint = pretrained_model
    
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)
    sam_model.eval()
    

    val_dataset = SAMDataset(data_dir, sam_model.image_encoder.img_size, split='test', return_index=True)
    
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    
    # load_model(sam_model, pretrained_model)
    predictor = SamPredictor(sam_model)

    os.makedirs(save_dir, exist_ok=True)

    print('starting inference')
    for idx, batch in enumerate(tqdm(val_dataloader)):
        
        input_image = batch['image'][0].numpy().astype(np.uint8) # batch size is only 1
        prompt_box = batch['input_boxes'][0]
        predictor.set_image(input_image)
        input_bbox = np.array(prompt_box)

        pred_masks, _, _ = predictor.predict(
            point_coords=None,
            box=input_bbox,
            multimask_output=False,
        )
        
        pred_masks = pred_masks[0].astype(bool)
        
        image_path = val_dataset.images[batch['index'][0]]
        gt_mask = batch['ground_truth_mask'][0].numpy().astype(bool)
        gt_box = batch['input_boxes'][0].numpy()
        
        plot_masks(input_image, pred_masks, image_path, save_dir, gt_mask, gt_box)


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser('SAM model inference')
    
    parser.add_argument('--data_dir', type=str, default='data', help='path to data directory')
    parser.add_argument('--save_dir', type=str, default='outputs/', help='path to save directory')
    parser.add_argument('--pretrained_model', type=str, default='checkpoints/sam_casia_finetune/model.pt', help='path to pretrained model')
    # parser.add_argument('--batch_size', type=int, default=2, help='batch size') # does not work with batch size > 1
    
    args = parser.parse_args()
    
    infer(args.data_dir, args.save_dir, args.pretrained_model)
    