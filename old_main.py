import argparse
import json
import os
import sys
import time
import torch
import random
import numpy as np
import traci
import traci.constants as tc
from sumolib import checkBinary


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="choose the mode",
        epilog="python run.py mode")

    # required input parameters
    parser.add_argument(
        'mode', type=str,
        help='train or test, simulate is the old version to train')

    # optional input parameters
    parser.add_argument(
        '--disp', type=bool, default=False,
        help='show the process while in training')


def train(sumoCmd):
    traci.start(sumoCmd)
    while step < 5000:
        traci.simulationStep()

    traci.close()


def main(args):
    flags = parse_args(args)
    # check gui option
    if flags.disp == True:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    # sumocfg file 위치
    sumoConfig = os.path.join(
        configs['current_path'], 'training_data', time_data, 'net_data', configs['file_name']+'_train.sumocfg')

    sumoCmd = [sumoBinary, "-c", sumoConfig, '--start']

    if flags.mode == 'train':
        train(sumoCmd)


if __name__ == '__main__':
    flags = parse_args(args)
    main(sys.argv[1:])
