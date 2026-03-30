"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import os

import config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import MyMOLO
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            # y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                # + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)



def main():

    # 1. Initialize the distributed process group
    # nccl is the standard backend for NVIDIA GPUs
    dist.init_process_group(backend="nccl")
    
    # 2. Get the local rank assigned by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 3. Initialize model and move it to the specific device first
    model = MyMOLO(config_path=config.CONFIG_PATH).to(device)
    
    # 4. Wrap the model in DDP
    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.TRAIN_DIR, test_csv_path=config.TEST_DIR
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)


        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        if config.SAVE_MODEL and local_rank == 0:
           save_checkpoint(model.module, optimizer, filename=f"checkpoint.pth.tar")
        
        if epoch > 1 and epoch % 20 == 0 and local_rank == 0:
            print(f"Currently epoch {epoch}")
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            
        if epoch == config.NUM_EPOCHS-1 and local_rank == 0:
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            print("Boxes gotten from get_evaluation_bboxes function")
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
