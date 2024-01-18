# ND-iris-0405 dataset
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import cv2
# from transformers import  SamImageProcessor
from segment_anything.utils.transforms import ResizeLongestSide

from utils.common import get_bounding_box, resize_and_pad_image, undo_padding_and_resize

class SAMDataset(Dataset):      
    def __init__(self, data_dir, img_size, out_mask_shape=256, split='train', transform=None, return_index=False, identity_range=None):
        
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform  # Added transform as an attribute
        self.return_index = return_index
        self.out_mask_shape = out_mask_shape
        if out_mask_shape > 0:
            self.mask_transform = ResizeLongestSide(out_mask_shape)
        # Directories for images and masks
        image_dir = os.path.join(data_dir, split, "images")
        mask_dir = os.path.join(data_dir, split, "masks")
        
        if split == "train":
            self.transform = ResizeLongestSide(img_size)
        else:
            self.transform = None
        
        # Get list of image names from the image directory
        image_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        
        self.images = []
        self.masks = []
        
        for image_name in image_names:
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name)  # No renaming needed since both have .jpg extension
            
            if not os.path.exists(mask_path):
                print(f"Warning: Corresponding mask for image {image_name} not found. Skipping...")
                continue

            self.images.append(image_path)
            self.masks.append(mask_path)
        
        # Convert to numpy arrays for possible efficiency reasons (can be kept as lists if preferred)
        self.images = np.array(self.images)
        self.masks = np.array(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image {image_path}. Skipping index {idx}.")
            return None  # Don't raise an error, just return None
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            raise ValueError(f"Error converting image {image_path} to RGB: {e}")

        original_image_size = image.shape[:2]

        mask_path = self.masks[idx]
        ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if ground_truth_mask is None:
            raise ValueError(f"Error loading mask for image {image_path}. Skipping index {idx}.")


        box = get_bounding_box(ground_truth_mask)
        
        if self.transform is not None: # do not do this for validation
            image = self.transform.apply_image(image)
            image = torch.as_tensor(image)
            image = image.permute(2, 0, 1).contiguous()
            input_size = tuple(image.shape[-2:])
        
            box = self.transform.apply_boxes(box, original_image_size)
        else:
            input_size = tuple(image.shape[:2])
        
        if self.out_mask_shape > 0:
            ground_truth_mask = self.mask_transform.apply_image(ground_truth_mask)
            h, w = ground_truth_mask.shape[:2]
            padh = self.out_mask_shape - h
            padw = self.out_mask_shape - w
            ground_truth_mask = np.pad(ground_truth_mask, ((0, padh), (0, padw)), mode='constant', constant_values=0)
            ground_truth_mask[ground_truth_mask > 117] = 255
            ground_truth_mask[ground_truth_mask <= 117] = 0
        

        inputs = {}
        inputs['image'] = image
        inputs['input_boxes'] = box
        inputs['input_size'] = input_size
        inputs['original_image_size'] = np.array([original_image_size[0], original_image_size[1]])
        inputs["ground_truth_mask"] = ground_truth_mask
        
        if self.return_index:
            inputs["index"] = idx

        return inputs

# class SAMDataset(Dataset):
#     def __init__(self, data_dir, processor,  out_mask_shape=256, split="train", return_index=False):
    
#         self.processor = processor
#         self.out_mask_shape = out_mask_shape
#         self.return_index = return_index

#         split_file = os.path.join(data_dir, f"{split}.txt")

#         self.images = []
#         self.masks = []
#         with open(split_file, "r") as f:
#             for line in f:
#                 image_name = line.strip()
#                 image_path = os.path.join(data_dir, "images", image_name)
#                 mask_path = os.path.join(data_dir, "masks", image_name.replace(".jpg", ".png"))
                
#                 self.images.append(image_path)
#                 self.masks.append(mask_path)
#         self.images = np.array(self.images)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):

#         image = cv2.imread(self.images[idx])
#         ground_truth_mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
#         prompt = get_bounding_box(ground_truth_mask)
        
#         # ground_truth_mask = cv2.cvtColor(ground_truth_mask, cv2.COLOR_GRAY2RGB) 
#         # get bounding box prompt

#         # prepare image and prompt for the model
#         inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

#         # remove batch dimension which the processor adds by default
#         inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        

#         # add ground truth segmentation
#         # resized_mask = resize_and_pad_image(ground_truth_mask, (self.out_mask_shape, self.out_mask_shape))
                             
#         # make it 0 or 255
#         # resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)[1]
#         # resized_mask[resized_mask > 0] = 1
        
#         # masks = processor.image_processor.post_process_masks(
#             # torch.tensor(ground_truth_mask).unsqueeze(0).unsqueeze(0), inputs["original_sizes"].cpu(), (self.out_mask_shape, self.out_mask_shape))

#         # resized_mask = masks[0].squeeze(0).squeeze(0).numpy()
#         inputs["ground_truth_mask"] = ground_truth_mask
        
#         if self.return_index:
#             inputs["index"] = idx

#         return inputs

if __name__ == '__main__':
    # from transformers import SamProcessor
    from segment_anything import  sam_model_registry
    sam_model = sam_model_registry['vit_h'](checkpoint='weights/sam_vit_h_4b8939.pth')
    # processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    dataset = SAMDataset("data", 1024, split="train", out_mask_shape=256, return_index=False)
    
    # Iterate over the dataset
    for i in range(len(dataset)):
        try:
            item = dataset[i]
        except ValueError as e:
            print(e)  # Log the error
            continue  # Skip the rest of the processing for this iteration

        # Process the item
        input_image = sam_model.preprocess(item["image"][None, :, :, :])
        print(input_image.shape)

        mask = item["ground_truth_mask"]
        print(np.unique(mask))
        print(mask.shape, mask.dtype)

    item = dataset[5]
    for k, v in item.items():
        print(k)
        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            print(v.shape, v.dtype)
        else:
            print(v)
            
    input_image = sam_model.preprocess(item["image"][None, :, :, :])
    print(input_image.shape)
    
    # boxes = item["input_boxes"].numpy()
    # image = item["pixel_values"].permute(1,2,0).numpy()
    # image_name = dataset.images[5]
    
    # image = (image * 255).astype(np.uint8)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # # image = cv2.imread(image_name)
    
    # for box in boxes:
    #     x_min, y_min, x_max, y_max = box.astype(int)
        
    #     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
    
    mask = item["ground_truth_mask"]
    # # mask = cv2.resize(mask, (1024, 1024))#.astype(np.uint8) * 255
    # # mask = undo_padding_and_resize(mask, image.shape[:2]) *255
    # # mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    # # image = cv2.resize(image, (320, 320))
    # # mask = cv2.imread('S1144R02.png', cv2.IMREAD_GRAYSCALE)
    print(np.unique(mask))
    print(mask.shape, mask.dtype)
    
    
    # # add mask to image as overlay
    # mask = np.expand_dims(mask, axis=-1).repeat(3, axis=-1)
    # # mask = np.uint8(mask)*255
    # # print(image.shape, mask.shape, image.dtype, mask.dtype)
    # # image = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
    cv2.imwrite("test_mask.png", mask)
    