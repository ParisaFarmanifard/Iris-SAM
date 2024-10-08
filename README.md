# Iris Segmentation Using a Foundation Model
This repository contains the code for training and inference of Iris-SAM (Published at ICPRAI2024).

## Abstract
**Iris segmentation** is a critical component of an iris biometric system and involves extracting the annular iris region from an ocular image. In this work, we develop a **pixel-level iris segmentation model** from a foundation model, viz., Segment Anything Model (SAM), that has been successfully used for segmenting arbitrary objects. The primary contribution of this work lies in the integration of different loss functions during the fine-tuning of SAM on ocular images. In particular, the importance of **Focal Loss** is borne out in the fine-tuning process since it strategically addresses the class imbalance problem (i.e., iris versus non-iris pixels). Experiments on **ND-IRIS-0405**, **CASIA-Iris-Interval-v3**, and **IIT-Delhi-Iris** datasets convey the efficacy of the trained model for the task of iris segmentation. For instance, on the ND-IRIS-0405 dataset, an average segmentation accuracy of **99.58%** was achieved, compared to the best baseline performance of **89.75%**.

## Requirements
- Download SAM's [checkpoints](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
- [Iris-SAM-checkpoints](https://drive.google.com/file/d/1Oqr93dMaHMZfwd2M3FvM-q5Xnn_d321x/view?usp=sharing)
 
First, we need to make a conda environment with the following command:
```
conda create -n iris-sam python=3.8
conda activate iris-sam
```
Then, we need to install the requirements:
```
pip install -r requirements.txt
```

## Dataset
The dataset is provided in `data` directory along with images and masks. The images are in `.jpg` format.

## Training
To train the model, run the following command:
```
python sam_train_losses.py --data_dir data  --epochs 100 --loss focal --lr 0.0001 --save_dir checkpoints/ND


Model checkpoints will be saved in `save_dir` (default: `checkpoints/ft`). The program will save two checkpoints, one with optimizer and one without optimizer. The optimizer checkpoint is used for resuming training.
The checkpoint without optimizer (`model.pt`) is used for inference. We already provide a fine-tuned checkpoint in `google drive` and suggest to use model.pt.
```
## Testing
We also provide a script to test the model on the val set. To test the model, run the following command:

```
python sam_test.py --data_dir data --pretrained_model *path_to_model/model.pt --save_dir *path_to_output/ND_test
```

## Inference
To run inference, we need to provide bbox coordinates for the segmentation region in the image (automated). This is how SAM works,i.e. it requires a prompt along with image for prediction. In our case, the prompt is the bbox coordinates. We provide a script to run inference on a single image. To run inference, run the following command:
```     
python sam_infer.py --image_path {PATH_TO_IMAGE_OR_DIR} --pretrained_model weights/model.pt --save_dir results
```

when  you run the above command, an image window will be opened. You will be asked to draw (click) the top left and bottom right coordinates of the bbox using mouse pointer.  Results will be saved in `save_dir` (default: `outputs/results`). The results include the overlay of predicted and ground truth masks. 

The inference command can either take a single image or a directory containing multiple images as input. In case of directory, also provide the extension of the images. For example, if the images are in `.jpg` format, run the following command:

```
python sam_infer.py --image_path {PATH_TO_IMAGE_DIR} -extension jpg --pretrained_model weights/model.pt --save_dir results
```
## Results
<p align="center" style="display: flex; justify-content: center; align-items: center;">
  <div style="width: 50%; text-align: center;">
    <img src="/assets/result3.jpg" alt="Ground Truth Mask" style="width: 100%;"/>
    <p>GT=GroundTruth , Pred=Predicted</p>
  </div>
  <div style="width: 50%; text-align: center;">
    <img src="/assets/result2.jpg" alt="Predicted Mask" style="width: 100%;"/>
    <p>GT=GroundTruth, Pred=Predicted</p>
  </div>
</p>

If you get errors with cv2 while plotting bboxes on the image, uninstall and install OpenCV as follows:
```
pip uninstall opencv-python-headless -y 
pip install opencv-python --upgrade
```
## Citations
If you find this repository useful, please consider giving a star ⭐ and a citation.
(https://arxiv.org/abs/2402.06497).
```
P. Farmanifard and A. Ross, "Iris-SAM: Iris Segmentation Using a Foundation Model," Proc. of 4th International Conference on Pattern Recognition and Artificial Intelligence (ICPRAI), (Jeju Island, South Korea), July 2024.

```
## Acknowledgmenets
- We thank the University of Salzburg and Halmstad University for providing the ground truth datasets.
- We thank Meta AI for making the source code of [SAM](https://github.com/facebookresearch/segment-anything) publicly available.
- MedSAM.





