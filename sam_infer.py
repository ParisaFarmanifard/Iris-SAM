# casia-interval-v3 dataset
import os
from segment_anything import  sam_model_registry, SamPredictor
import torch
import numpy as np
import cv2

from utils.common import load_model, plot_masks

def plot_masks(image, masks, image_file, save_dir, gt_box=None):
    mask_overlay = (masks[..., None] * np.array([0, 255, 0])).astype(np.uint8)
    result = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)
    if gt_box is not None:
        cv2.rectangle(result, tuple(gt_box[:2]), tuple(gt_box[2:]), (255, 0, 0), 2)
    save_path = os.path.join(save_dir, os.path.basename(image_file))
    cv2.imwrite(save_path, result)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points = param['points']
        image = param['image']
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        # Display the updated image
        if len(points) == 2:
            cv2.rectangle(image, points[0], points[1], (0, 255, 0), 2)
        cv2.imshow(param['name'], image)

def draw_box_points(image, name='Image', num_points=4):
    
    cv2.namedWindow(name)
    params = {'points': [], 'image': image, 'name': name}
    cv2.setMouseCallback(name, mouse_callback, params)

    print('Please draw top left and bottom right corners of bounding box on the image')

    cv2.imshow(name, image)

    while len(params['points']) < num_points:
        cv2.waitKey(1)
    
    return params['points']
    

def infer(image_path, save_dir, pretrained_model, extension='jpg'):
    

    if os.path.isdir(image_path):
        image_files = [f.path for f in os.scandir(image_path) if f.name.endswith(extension)]
    elif os.path.isfile(image_path) and os.path.exists(image_path):
        image_files = [image_path]
    
    else:
        print('image_path is not a valid file or directory')
        exit()
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = 'vit_h'
    checkpoint = pretrained_model
    
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)
    sam_model.eval()
        
    predictor = SamPredictor(sam_model)

    os.makedirs(save_dir, exist_ok=True)

    print('starting inference')
    for idx, image_file in  enumerate(image_files):
        image_name = os.path.basename(image_file)
        
        print(f'Processing image file: {image_name};  {idx+1}/{len(image_files)}')
        
        input_image = cv2.imread(image_file)
        
        # use cv2 to get 4 points as a bounding boux from user}
        bbox = draw_box_points(input_image, name=image_name)
        # bbox = [(0,0), (input_image.shape[1], input_image.shape[0])]
        print('BBox coordinates provided by user: ', bbox)
        
        input_image = input_image[:,:,::-1] # convert to RGB
        predictor.set_image(input_image)
        input_bbox = np.array(bbox)

        pred_masks, _, _ = predictor.predict(
            point_coords=None,
            box=input_bbox,
            multimask_output=False,
        )
        
        pred_masks = pred_masks[0].astype(bool)
                
        plot_masks(input_image, pred_masks, image_path, save_dir, gt_box=input_bbox.reshape(-1))
        mask_path = os.path.join(save_dir, f'{os.path.splitext(image_name)[0]}_pred_mask.png')
        cv2.imwrite(mask_path, pred_masks.astype(np.uint8)*255)
        
        # destroy the bbox input window
        cv2.destroyAllWindows()
        # exit()


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser('SAM model inference')
    parser.add_argument('--image_path', type=str, required=True, help='path to an image file or directory of images')
    parser.add_argument('--extension', type=str, default='jpg', help='image file extension, useful if image_path is a directory')
    parser.add_argument('--save_dir', type=str, default='results', help='path to save directory')
    parser.add_argument('--pretrained_model', type=str, default='weights/model.pt', help='path to pretrained model')
    # parser.add_argument('--batch_size', type=int, default=2, help='batch size') # does not work with batch size > 1
    
    args = parser.parse_args()
    
    infer(args.image_path, args.save_dir, args.pretrained_model, args.extension)
    
