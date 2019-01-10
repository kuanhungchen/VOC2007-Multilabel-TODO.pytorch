import argparse
import os
import time
from collections import deque

import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import Dataset
from config import Config
from model import Model


def _train(path_to_data_dir: str, path_to_checkpoints_dir: str):
    os.makedirs(path_to_checkpoints_dir, exist_ok=True)

    # TODO: CODE BEGIN
    # raise NotImplementedError
    dataset = Dataset(path_to_data_dir, mode=Dataset.Mode.TRAIN)
    dataloader = DataLoader(dataset, batch_size=Config.Batch_Size, shuffle=True)

    model = Model()
    model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=Config.Learning_Rate, momentum=Config.Momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.EveryStepsToDecayLR, gamma=Config.Gamma)

    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    should_stop = False
    # TODO: CODE END

    print('Start training')

    while not should_stop:
        # TODO: CODE BEGIN
        # raise NotImplementedError
        for batch_idx, (images, multilabels) in enumerate(dataloader):
            images = images.cuda()
            multilabels = multilabels.float().cuda()

            logits = model.train().forward(images)
            loss = model.loss(logits, multilabels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1

            if step == Config.StepToFinish:
                should_stop = True

            if step % Config.EveryStepsToDisplay == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_second = Config.EveryStepsToDisplay / elapsed_time
                avg_loss = sum(losses) / len(losses)
                lr = scheduler.get_lr()[0]
                print(f'[Step {step}] Avg. Loss = {avg_loss:.8f}, Learning Rate = {lr} ({steps_per_second:.2f} steps/sec)')

            if step % Config.EveryStepsToSave == 0 or should_stop:
                path_to_checkpoint = model.save(path_to_checkpoints_dir, step)
                print(f'Model saved to {path_to_checkpoint}')
        # TODO: CODE END

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-c', '--checkpoints_dir', default='./checkpoints', help='path to checkpoints directory')
        args = parser.parse_args()

        path_to_data_dir = args.data_dir
        path_to_checkpoints_dir = args.checkpoints_dir

        _train(path_to_data_dir, path_to_checkpoints_dir)

    main()
