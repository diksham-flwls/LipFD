import argparse
import torch
import numpy as np
from data import AVLip
import torch.utils.data
from models import build_model
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score


def validate(model, loader, gpu_id, tensorboard_writer):
    print("validating...")
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, crops, label in loader:
            img_tens = img.to(device)
            crops_tens = [[t.to(device) for t in sublist] for sublist in crops]
            features = model.model.get_features(img_tens).to(device)

            y_pred.extend(model.model(crops_tens, features)[0].sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
    y_true = np.array(y_true)
    y_pred = np.where(np.array(y_pred) >= 0.5, 1, 0)

    # Get AP
    ap = average_precision_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tp, fn, fp, tn = cm.ravel()
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)
    acc = accuracy_score(y_true, y_pred)
    tensorboard_writer.add_scalar("Avg precision score", ap, model.total_steps)
    tensorboard_writer.add_scalar("Avg acc score", acc, model.total_steps)
    tensorboard_writer.add_scalar("Avg fpr", fpr, model.total_steps)
    tensorboard_writer.add_scalar("Avg fnr", fnr, model.total_steps)
    return ap, fpr, fnr, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--real_list_path", type=str, default="./datasets/val/0_real")
    parser.add_argument("--fake_list_path", type=str, default="./datasets/val/1_fake")
    parser.add_argument(
        "--max_sample", type=int, default=1000, help="max number of validate samples"
    )
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--data_label", type=str, default="val")
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument(
        "--ckpt", type=str, default="./checkpoints/experiment_name/model_epoch_29.pth"
    )

    opt = parser.parse_args()

    model = build_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    print("Model loaded..")
    model.eval()
    model.cuda()

    dataset = AVLip(opt)
    loader = data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True
    )
    ap, fpr, fnr, acc = validate(model, loader, gpu_id=[0])
    print(f"ap: {ap} acc: {acc} fpr: {fpr} fnr: {fnr}")
