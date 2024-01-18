# ND-iris-0405 dataset and IITD
import os
from segment_anything import sam_model_registry, SamPredictor
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import cv2
from dataset import SAMDataset

def smooth_mask(mask, kernel_size=3):
    """
    Apply morphological operations to smooth the mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Removes noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Closes small holes
    return mask

def plot_masks(image, pred_mask, image_path, save_dir, gt_mask=None, gt_box=None):
    # Define colors
    dark_blue = [139, 30, 30]  # BGR format for dark blue (visible against most backgrounds)
    dark_green = [0, 128, 0]  # BGR format for dark green
    black = [0,0,0]

    # Function to overlay mask on image
    def overlay_mask_on_image(orig_image, mask, color, alpha=0.4):
        # Convert color to a 3-channel array
        mask_color = np.array(color, dtype=np.uint8).reshape(1, 1, 3)

        # Create a binary mask
        mask_binary = mask > 0

        # Resize mask to match image size
        mask_resized = cv2.resize(mask_binary.astype(np.uint8), (orig_image.shape[1], orig_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create an image of the mask color with the same size as the original image
        color_overlay = np.full(orig_image.shape, mask_color, dtype=np.uint8)

        # Blend the color overlay with the original image
        blended_overlay = cv2.addWeighted(orig_image, 1 - alpha, color_overlay, alpha, 0)

        # Apply the blended overlay only where the mask is present
        image_with_mask = orig_image.copy()
        image_with_mask[mask_resized == 1] = blended_overlay[mask_resized == 1]

        return image_with_mask


    # Apply GT mask if available
    if gt_mask is not None:
        # Smooth the GT mask
        smoothed_gt_mask = smooth_mask(gt_mask.astype(np.uint8))
        gt_result = overlay_mask_on_image(image, smoothed_gt_mask, dark_blue)
        if gt_box is not None:
            cv2.rectangle(gt_result, tuple(gt_box[:2]), tuple(gt_box[2:]), dark_green, 2)
    else:
        gt_result = image.copy()


    # Apply Predicted mask
    pred_result = overlay_mask_on_image(image, pred_mask, dark_blue)
    if gt_box is not None:
        cv2.rectangle(pred_result, tuple(gt_box[:2]), tuple(gt_box[2:]), dark_green, 2)

    # Concatenate images horizontally
    combined_result = np.hstack((gt_result, pred_result))

    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, combined_result)  # Save the combined result

    
    # Make an empty canvas that's larger than our image for placing text outside of the image
    canvas = np.ones((combined_result.shape[0] + 50, combined_result.shape[1], 3), dtype=np.uint8) * 255
    canvas[50:, :] = combined_result
    
    
    # Adjust font size and position labels
    label_font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_scale = 1.3  # You can adjust this value as 2.2
    thickness = 2    # Increase for bolder text 4
    width = canvas.shape[1]
    cv2.putText(canvas, 'GT Mask', (width // 4 - 50, 30), label_font, font_scale, black, thickness, cv2.LINE_AA)
    cv2.putText(canvas, 'Predicted Mask', (3 * width // 4 - 100, 30), label_font, font_scale, black, thickness, cv2.LINE_AA)

    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, canvas)
    
def infer(data_dir, save_dir, pretrained_model):
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    model_type = 'vit_h'
    checkpoint = pretrained_model
    
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)
    sam_model.eval()
    
    val_dataset = SAMDataset(data_dir, sam_model.image_encoder.img_size, split='test', return_index=True, out_mask_shape=0)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    predictor = SamPredictor(sam_model)

    os.makedirs(save_dir, exist_ok=True)

    print('starting inference')
    for idx, batch in enumerate(tqdm(val_dataloader)):
        input_image = batch['image'][0].numpy().astype(np.uint8)
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
    parser.add_argument('--pretrained_model', type=str, default='*/model.pt', help='path to pretrained model')
    args = parser.parse_args()
    infer(args.data_dir, args.save_dir, args.pretrained_model)
    