from segment_anything import  sam_model_registry

import os

from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from torch.nn.functional import threshold, normalize
from torch.utils import tensorboard
import numpy as np

from dataset import SAMDataset
from utils.common import save_model, load_model


def train(args):
    

    model_type = 'vit_h'
    checkpoint = 'weights/sam_vit_h_4b8939.pth'
    # checkpoint = 'weights/sam_vit_b_01ec64.pth'
    
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam_model.to(device)
    sam_model.train()
    
    train_dataset = SAMDataset(args.data_dir, sam_model.image_encoder.img_size, out_mask_shape=256, split='train')
    # val_dataset = SAMDataset(args.data_dir, processor=processor, split='val')
    

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Initialize model
    # make sure we only compute gradients for mask decoder
    for name, param in sam_model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    
    
    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(sam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=0)

    loss_fn = torch.nn.MSELoss()
    
    if args.pretrained_model:
        cur_epoch = load_model(sam_model, args.pretrained_model, optimizer)
        start_epoch = cur_epoch + 1
    else:
        start_epoch = 0

    os.makedirs(args.save_dir, exist_ok=True)
    summary = tensorboard.SummaryWriter(args.save_dir)

    print('starting training from epoch', start_epoch)
    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []
        
        pbar = tqdm(train_dataloader, desc=f'epoch {epoch+1}/{args.epochs}; batch: {0}/{len(train_dataloader)}; loss: {0}')
        for idx, batch in enumerate(pbar):
            # forward pass
            input_images = sam_model.preprocess(batch['image'].to(device))
            prompt_boxes = batch['input_boxes'].to(device)
            # input_size = batch['input_size'][0]
            input_size = tuple(input_images.shape[-2:])
            original_image_size = batch['original_image_size'][0].cpu().numpy().tolist()
            
            
            # No grad here as we don't want to optimise the encoders
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_images)
                boxes_torch = torch.as_tensor(prompt_boxes, dtype=torch.float, device=device)
                # boxes_torch = boxes_torch[None, :]
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=boxes_torch,
                    masks=None,
                )
            pred_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
            # binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

            ground_truth_masks = batch["ground_truth_mask"].to(device)
            # gt_mask_resized = ground_truth_masks.unsqueeze(1).to(device)
            gt_binary_mask = torch.as_tensor(ground_truth_masks > 0, dtype=torch.float32).unsqueeze(1)
            loss = loss_fn(pred_masks, gt_binary_mask)
            # print(loss.item()  )

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())
            pbar.set_description(f'epoch {epoch+1}/{args.epochs}; batch: {idx+1}/{len(train_dataloader)}; loss: {loss.item():.6f}')
            
            cur_step = epoch * len(train_dataloader) + idx + 1
            summary.add_scalar('loss/step', loss.item(), cur_step)

        loss = np.mean(epoch_losses)
        print(f'EPOCH: {epoch+1}/{args.epochs}, loss: {loss:.6f}')
        summary.add_scalar('loss/epoch', loss, epoch+1)
        
        save_path = os.path.join(args.save_dir, f"model_with_optim.pt")
        
        # save with optimizer, useful for fine-tuning
        save_model(sam_model, save_path, epoch=epoch, optimizer=optimizer, data_parallel=False)
        
        # save model only, can be  loaded directly to predict
        save_path = os.path.join(args.save_dir, f"model.pt")
        save_model(sam_model, save_path, data_parallel=False, make_dict=False)
    


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser('SAM model training')
    
    parser.add_argument('--data_dir', type=str, default='data', help='path to data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints/sam_casia_finetune', help='path to save directory')
    parser.add_argument('--pretrained_model', type=str, default='', help='path to pretrained model')
    
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    # parser.add_argument('--batch_size', type=int, default=1, help='batch size') # doesnot work with batch size > 1
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')

    args = parser.parse_args()
    
    train(args)
    