import os
import argparse
import torch
from torch.utils.data import DataLoader

from data.process_data_new import load_rir_dataset, load_speech_dataset  # 使用新的数据加载函数
from trainer import Trainer
from dataloader_new import ReverbDataset  # 使用新的数据加载类
from model import FilteredNoiseShaper
from utils.utils import load_config


def main(args):
    # load config
    config_path = "/home/cxw/GAN/fins-main(rt60&vol)（64+64+bp+de+fuse+withoutcrossattention）/fins/config.yaml"
    config = load_config(config_path)
    print(config)

    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
        torch.cuda.set_device(args.device)
    else:
        args.device = "cpu"

    # Load RIR and speech datasets
    csv_path = ("/home/cxw/GAN/fins-main(rt60&vol)（64+64+bp+de+fuse+withoutcrossattention）/fins/read(BP).csv")  # CSV 文件路径
    train_rir_list, valid_rir_list, test_rir_list = load_rir_dataset(csv_path)
    train_source_list, valid_source_list, test_source_list = load_speech_dataset(csv_path)

    # Load dataset
    train_dataset = ReverbDataset(data=train_rir_list, config=config.dataset.params, use_noise=True)
    valid_dataset = ReverbDataset(data=valid_rir_list, config=config.dataset.params, use_noise=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.params.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.train.params.num_workers,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.train.params.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=config.train.params.num_workers,
    )

    print("Number of RIR data", len(train_rir_list), len(valid_rir_list))
    print("Number of speech data", len(train_source_list), len(valid_source_list))
    print("Number of batches", len(train_dataloader), len(valid_dataloader))

    # Load model
    model = FilteredNoiseShaper(config.model.params)

    # Run trainer
    trainer = Trainer(model, train_dataloader, valid_dataloader, config.train.params, config.eval.params, args)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default='cpu')
    parser.add_argument("--save_name", type=str, default="m")
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    args = parser.parse_args()

    main(args)
