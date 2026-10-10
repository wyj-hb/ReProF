import json
import argparse
from trainer import train
import os
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')  # 忽略所有警告
def main(name):
    args = setup_parser(name).parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    train(args)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser(name):
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default=f'./exps/{name}.json',
                        help='Json file of settings.')
    return parser

if __name__ == '__main__':
    main('ReProf')