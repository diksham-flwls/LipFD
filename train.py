from validate import validate
from data import create_dataloader
from trainer.trainer import Trainer
from options.train_options import TrainOptions
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import torch

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=True)
    val_opt.isTrain = False
    val_opt.data_label = "val"
    val_opt.real_list_path = "./datasets/val/0_real"
    val_opt.fake_list_path = "./datasets/val/1_fake"
    return val_opt


if __name__ == "__main__":
    breakpoint()
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
    model = Trainer(opt)

    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    tb_writer_train = SummaryWriter(log_dir=f"{opt.save_path}/train")
    tb_writer_val = SummaryWriter(log_dir=f"{opt.save_path}/val")

    print("Length of data loader: %d" % (len(data_loader)))
    print("Length of val  loader: %d" % (len(val_loader)))

    for epoch in range(opt.epoch):
        model.train()
        print("epoch: ", epoch + model.step_bias)
        train_loss = 0
        train_acc = 0
        counter = 0
        for i, (img, crops, label) in enumerate(data_loader):
            model.total_steps += 1
            counter += 1

            model.set_input((img, crops, label))
            model.forward()
            loss = model.get_loss()
            train_loss += loss

            model_pred = model.output.sigmoid().view(-1).cpu()
            pred_class = torch.where(model_pred > 0.5, 1, 0)
            target_class = label.view(-1).cpu()
            train_acc += accuracy_score(target_class, pred_class)

            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(
                    "Train loss: {}\tstep: {}".format(
                        model.get_loss(), model.total_steps
                    )
                )
                tb_writer_train.add_scalar(f"Train Loss", train_loss/counter, model.total_steps)
                tb_writer_train.add_scalar(f"Train acc", train_acc/counter, model.total_steps)

        if epoch % opt.save_epoch_freq == 0:
            print("saving the model at the end of epoch %d" % (epoch + model.step_bias))
            model.save_networks("model_epoch_%s.pth" % (epoch + model.step_bias))

        model.eval()
        ap, fpr, fnr, acc = validate(model, val_loader, opt.gpu_ids, tb_writer_val)
        print(
            "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
                epoch + model.step_bias, acc, ap, fpr, fnr
            )
        )
