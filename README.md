# SAM for Iris
This repository contains the code for training and inference of SAM on Iris dataset.

## Requirements
- Download SAM checkpoints [SAM]([https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints])

All testing has  been done on Linux with Nvidia GPUs (Cuda). First, we need to make a conda environment with the following command:
```
conda create -n sam python=3.8
conda activate sam
```
Then, we need to install the requirements:
```
pip install -r requirements.txt
```

## Dataset
The dataset is provided in `data` directory along with images and masks. The directory also contains `train.txt` and `val.txt` files which contain the names of training and validation images respectively. The images are in `.jpg` format and masks are in `.png` format.

## Training
To train the model, run the following command:
```
python sam_train_losses.py --data_dir data  --epochs 100 --loss focal --lr 0.0001 --save_dir checkpoints/ND


Model checkpoints will be saved in `save_dir` (default: `checkpoints/ft`). The program will save two checkpoints, one with optimizer and one without optimizer. The optimizer checkpoint is used for resuming training.
The checkpoint without optimizer (`model.pt`) is used for inference. We already provide a fine-tuned checkpoint in `google drive`.
```
## Testing
We also provide a script to test the model on the val set. To test the model, run the following command:


python sam_t_orgcolor.py --data_dir data --pretrained_model *path_to_model/model.pt --save_dir *path_to_output/ND_test


## Inference
To run inference, we need to provide bbox coordinates for the segmentation region in the image. This is how SAM works,i.e. it requires a prompt along with image for prediction. In our case, the prompt is the bbox coordinates. We provide a script to run inference on a single image. To run inference, run the following command:

```     
python sam_infer.py --image_path {PATH_TO_IMAGE_OR_DIR} --pretrained_model weights/sam_model_casia_ft.pt --save_dir results
```

when  you run the above command, an image window will be opened. You will be asked to draw (click) the top left and bottom right coordinates of the bbox using mouse pointer.  Results will be saved in `save_dir` (default: `outputs/results`). The results include the overlay of predicted and ground truth masks. 

Teh inference command can either take a single image or a directory containing multiple images as input. In case of directory, also provide the extension of the images. For example, if the images are in `.jpg` format, run the following command:

```
python sam_infer.py --image_path {PATH_TO_IMAGE_DIR} -extension jpg --pretrained_model weights/sam_model_casia_ft.pt --save_dir results
```


If you get errors with cv2 while plotting bboxes on the image, uninstall and install OpenCV as follows:
```
pip uninstall opencv-python-headless -y 
pip install opencv-python --upgrade
```
## Acknowledgmenets
- We thank the University of Salzburg and Halmstad University for providing the ground truth datasets.
- We thank Meta AI for making the source code of [SAM](https://github.com/facebookresearch/segment-anything) publicly available. 




