import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os


def train(args):
    dataset_list = copy.deepcopy(args["dataset"])
    seed_list = copy.deepcopy(args["seed"])
    init_lr_list = copy.deepcopy(args["init_lr"])
    batch_size_list = copy.deepcopy(args["batch_size"])
    device = copy.deepcopy(args["device"])
    projection_dimensions = copy.deepcopy(args["M"])
    if not isinstance(projection_dimensions, list):
        projection_dimensions = [projection_dimensions]
    is_projection_sweep = len(projection_dimensions) > 1

    for dataset in dataset_list:
        for init_lr in init_lr_list:
            for batch_size in batch_size_list:
                for seed in seed_list:
                    for merge_type in args['fusion_type_list']:
                        for group_Num in args['K']:
                            for dimension in projection_dimensions:
                                args["dataset"] = dataset
                                args["init_lr"] = init_lr
                                args["batch_size"] = batch_size
                                args["seed"] = seed
                                args["device"] = device
                                args["M"] = dimension
                                args["projection_sweep"] = is_projection_sweep
                                args['group_Num'] = group_Num
                                args['merge_type'] = merge_type
                                args['dataset_name'] = os.path.basename(dataset)
                                _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(
        args["model_name"], args["convnet_type"], args["dataset_name"],
        f"{args['init_cls']}-{args['increment']}"
    )
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    log_suffix = "{}-{}_{}".format(
        args['init_cls'], args['increment'], args["seed"]
    )
    if args.get("projection_sweep", False):
        log_suffix += "_M{}".format(args["M"])
    logfilename = "logs/{}/{}/{}/{}".format(
        args["model_name"], args["convnet_type"], args["dataset_name"],
        log_suffix,
    )

    # Reset logging handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"], args["convnet_type"], args["shuffle"],
        args["seed"], args["init_cls"], args["increment"],
    )
    args["nb_classes"] = data_manager.get_total_classnum()
    args["nb_tasks"] = data_manager.nb_tasks

    model = factory.get_model(args["model_name"], args)
    cnn_curve = {"top1": [], "top5": []}
    nme_curve = {"top1": [], "top5": []}

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info("Trainable params: {}".format(count_parameters(model._network, True)))

        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        logging.info("CNN: {}".format(cnn_accy["grouped"]))
        cnn_curve["top1"].append(cnn_accy["top1"])

        # Per-class accuracy
        class_accuracy = {}
        for k, v in cnn_accy["pre_class_accuracy"].items():
            class_name = data_manager._idx_to_class.get(k)
            class_accuracy[class_name] = v
        logging.info("Accuracy per class: {}".format(class_accuracy))
        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))

        avg_cnn = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
        print('Average Accuracy (CNN):', avg_cnn)
        logging.info("Average Accuracy (CNN): {}".format(avg_cnn))

        if nme_accy is not None:
            logging.info("NME: {}".format(nme_accy["grouped"]))
            nme_curve["top1"].append(nme_accy["top1"])
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            avg_nme = sum(nme_curve["top1"]) / len(nme_curve["top1"])
            print('Average Accuracy (NME):', avg_nme)
            logging.info("Average Accuracy (NME): {}".format(avg_nme))
        else:
            logging.info("No NME accuracy.")


def _set_device(args):
    device_list = args["device"]
    gpus = []
    for device_id in device_list:
        if device_id == -1:
            gpus.append(torch.device("cpu"))
        else:
            gpus.append(torch.device("cuda:{}".format(device_id)))
    args["device"] = gpus


def _set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
