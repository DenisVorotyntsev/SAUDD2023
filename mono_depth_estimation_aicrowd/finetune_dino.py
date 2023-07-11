import os
import argparse

from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader

from mono_depth_estimation_aicrowd.preprocess import get_preprocess_for_dino
from mono_depth_estimation_aicrowd.dataset import DepthDataset
from mono_depth_estimation_aicrowd.my_models import DinoModel
from mono_depth_estimation_aicrowd.freeze import clc_frozen_params
from mono_depth_estimation_aicrowd.loss import SILoss
from mono_depth_estimation_aicrowd.callbacks import EarlyStopper
from mono_depth_estimation_aicrowd.predict_utils import (
    make_prediction_dino,
    save_predictions_batch,
)
from mono_depth_estimation_aicrowd.utils import plot_gradients


parser = argparse.ArgumentParser()
parser.add_argument("fold", type=int, help="the fold number")
args = parser.parse_args()
FOLD_NUMBER = args.fold
print(f"Training using fold number = {FOLD_NUMBER}")


debug = False
n_epochs = 20
BS = 2
NUM_ACCUMULATION_STEPS = 6
initial_lr = 0.00004
n_warmup_steps = 200
clip_grad_max_norm = 3
path_to_save_model = f"./mono_depth_estimation_aicrowd/models/"


device = torch.device("cuda")
model = DinoModel()
model.to(device)
clc_frozen_params(model)


transform_func = get_preprocess_for_dino()
criterion = SILoss()
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
early_stopper = EarlyStopper(patience=8, min_delta=0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)


n_dataloaer_processes = min([BS, os.cpu_count()])
print(f"Using CPU count: {n_dataloaer_processes}")
train_dataset = DepthDataset(
    fold_number=FOLD_NUMBER,
    dataset_mode="train",
    transform_func=transform_func,
    debug=debug,
    max_files=None,
)
val_dataset = DepthDataset(
    fold_number=FOLD_NUMBER,
    dataset_mode="test",
    transform_func=transform_func,
    debug=debug,
    max_files=None,
)
train_dataloader = DataLoader(
    train_dataset, batch_size=BS, num_workers=n_dataloaer_processes, shuffle=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BS, num_workers=n_dataloaer_processes, shuffle=False
)


pth_to_save_logs = os.path.join(path_to_save_model, f"fold_{FOLD_NUMBER}")
if not os.path.exists(pth_to_save_logs):
    os.makedirs(pth_to_save_logs)
with open(
    os.path.join(path_to_save_model, f"fold_{FOLD_NUMBER}", "log.txt"), "a"
) as fl:
    fl.write("\n\nnew train line")


n_steps = 0
best_score = None
for epoch in tqdm(range(n_epochs)):
    if epoch > 0 and epoch % 15 == 0:
        optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] / 2

    print(f"Epoch {epoch}, lr={optimizer.param_groups[0]['lr']}")

    running_loss = 0.0
    for i, data in tqdm(enumerate(train_dataloader)):
        if n_steps <= n_warmup_steps:
            optimizer.param_groups[0]["lr"] = (
                0.0 + initial_lr * n_steps / n_warmup_steps
            )
            print("Warmup, lr=", optimizer.param_groups[0]["lr"])
        else:
            optimizer.param_groups[0]["lr"] = initial_lr

        inputs, rel_depth_label = data
        inputs = inputs.to(device)
        rel_depth_label = rel_depth_label.to(device)

        pred_disparity_image = make_prediction_dino(model, inputs)
        loss = criterion(prediction=pred_disparity_image, target=rel_depth_label)
        running_loss += loss.item()
        loss = loss / NUM_ACCUMULATION_STEPS
        loss.backward()

        if n_steps % NUM_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_max_norm)
            optimizer.step()
            optimizer.zero_grad()

        n_steps += 1

        if i == 0:
            for k in range(pred_disparity_image.shape[0]):
                save_predictions_batch(
                    img=inputs[k, 0, ...].detach().cpu(),
                    predictions=pred_disparity_image[k, ...].detach().cpu(),
                    target=rel_depth_label[k, ...].detach().cpu(),
                    save_path=f"./preds/pred_train_{k}.png",
                )

    running_val_loss = 0.0
    for i, data in tqdm(enumerate(val_dataloader)):
        with torch.no_grad():
            inputs, rel_depth_label = data
            inputs = inputs.to(device)
            rel_depth_label = rel_depth_label.to(device)
            pred_disparity_image = make_prediction_dino(model, inputs)
            loss = criterion(prediction=pred_disparity_image, target=rel_depth_label)
            running_val_loss += loss

        if i == 0:
            for k in range(pred_disparity_image.shape[0]):
                save_predictions_batch(
                    img=inputs[k, 0, ...].detach().cpu(),
                    predictions=pred_disparity_image[k, ...].detach().cpu(),
                    target=rel_depth_label[k, ...].detach().cpu(),
                    save_path=f"./preds/pred_val_{k}.png",
                )

    train_loss = running_loss / len(train_dataloader)
    val_loss = running_val_loss / len(val_dataloader)
    score_line = f"Fold: {FOLD_NUMBER}. Epoch {epoch+1}, training loss: {train_loss:.4f} val loss: {val_loss:.4f}"
    print(score_line)

    if best_score is None:
        best_score = val_loss
    if val_loss <= best_score:
        best_score = val_loss
        print(f"Saving model with new best score={best_score}")
        torch.save(
            {
                "model": model.state_dict(),
                # 'optimizer': optimizer.state_dict()
            },
            os.path.join(path_to_save_model, f"fold_{FOLD_NUMBER}", "model.ckpt"),
        )
    else:
        print("Model is worse, skipping saving model")

    with open(
        os.path.join(path_to_save_model, f"fold_{FOLD_NUMBER}", "log.txt"), "a"
    ) as fl:
        fl.write(score_line + "\n")

    if early_stopper.early_stop(val_loss):
        print("Early stopped!")
        break

    print("\n\n")

with open("./mono_depth_estimation_aicrowd/models/mim-kfold-results.txt", "a") as fl:
    fl.write(score_line + "\n")

print("Done!")
