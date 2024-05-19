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
# ==============Color======================= 
# import os
# from segment_anything import sam_model_registry, SamPredictor
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import torch
# import numpy as np
# import cv2
# from dataset import SAMDataset

# def plot_masks(image, pred_mask, image_path, save_dir, gt_mask=None, gt_box=None):
#     
#     dark_blue = [139, 30, 30]  
#     dark_green = [0, 128, 0]  
#     black = [0, 0, 0]  
    
#     
#     if gt_mask is not None:
#         gt_mask_resized = cv2.resize(gt_mask.astype(np.uint8) * 255, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
#         gt_mask_resized = gt_mask_resized > 127  # Convert back to boolean mask
#         gt_overlay = (gt_mask_resized[..., None] * np.array(dark_blue)).astype(np.uint8)
#         gt_result = cv2.addWeighted(image, 0.4, gt_overlay, 0.6, 0)
#         if gt_box is not None:
#             cv2.rectangle(gt_result, tuple(gt_box[:2]), tuple(gt_box[2:]), dark_green, 2)
#     else:
#         gt_result = image.copy()

#     # Create overlay for predicted mask
#     pred_overlay = (pred_mask[..., None] * np.array(dark_blue)).astype(np.uint8)
#     pred_result = cv2.addWeighted(image, 0.4, pred_overlay, 0.6, 0)
#     if gt_box is not None:
#         cv2.rectangle(pred_result, tuple(gt_box[:2]), tuple(gt_box[2:]), dark_green, 2)
    
#     
#     combined_result = np.hstack((gt_result, pred_result))
    
#
#     canvas = np.ones((combined_result.shape[0] + 50, combined_result.shape[1], 3), dtype=np.uint8) * 255
#     canvas[50:, :] = combined_result
    
#     font_scale = 1.5  # You can adjust this value as needed

#     
#     # label_font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#     # width = canvas.shape[1]
#     # cv2.putText(canvas, 'GT Mask', (width // 4 - 50, 30), font_scale, label_font, 1, black, 1, cv2.LINE_AA)
#     # cv2.putText(canvas, 'Predicted Mask', (3 * width // 4 - 100, 30), font_scale, label_font, 1, black, 1, cv2.LINE_AA)
    
#     
#     label_font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#     font_scale = 1.0  # You can adjust this value as needed
#     width = canvas.shape[1]
#     cv2.putText(canvas, 'GT Mask', (width // 4 - 50, 30), label_font, font_scale, black, 1, cv2.LINE_AA)
#     cv2.putText(canvas, 'Predicted Mask', (3 * width // 4 - 100, 30), label_font, font_scale, black, 1, cv2.LINE_AA)

#     save_path = os.path.join(save_dir, os.path.basename(image_path))
#     cv2.imwrite(save_path, canvas)
    
# def infer(data_dir, save_dir, pretrained_model):
#     device = "cuda:2" if torch.cuda.is_available() else "cpu"
#     model_type = 'vit_h'
#     checkpoint = pretrained_model
    
#     sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
#     sam_model.to(device)
#     sam_model.eval()
    
#     val_dataset = SAMDataset(data_dir, sam_model.image_encoder.img_size, split='test', return_index=True, out_mask_shape=0)
#     val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
#     predictor = SamPredictor(sam_model)

#     os.makedirs(save_dir, exist_ok=True)

#     print('starting inference')
#     for idx, batch in enumerate(tqdm(val_dataloader)):
#         input_image = batch['image'][0].numpy().astype(np.uint8)
#         prompt_box = batch['input_boxes'][0]
#         predictor.set_image(input_image)
#         input_bbox = np.array(prompt_box)

#         pred_masks, _, _ = predictor.predict(
#             point_coords=None,
#             box=input_bbox,
#             multimask_output=False,
#         )
        
#         pred_masks = pred_masks[0].astype(bool)
#         image_path = val_dataset.images[batch['index'][0]]
#         gt_mask = batch['ground_truth_mask'][0].numpy().astype(bool)
#         gt_box = batch['input_boxes'][0].numpy()
        
#         plot_masks(input_image, pred_masks, image_path, save_dir, gt_mask, gt_box)

# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser('SAM model inference')
#     parser.add_argument('--data_dir', type=str, default='data', help='path to data directory')
#     parser.add_argument('--save_dir', type=str, default='outputs/', help='path to save directory')
#     parser.add_argument('--pretrained_model', type=str, default='*/gamma_2.0/model.pt', help='path to pretrained model')
#     args = parser.parse_args()
#     infer(args.data_dir, args.save_dir, args.pretrained_model)
    
