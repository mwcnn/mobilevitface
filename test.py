import torch
import torch.nn as nn
import torch.utils.data as data
from model import mobilevit, head
from utils.utils import get_val_data, perform_val
from IPython import embed
import sklearn, sys, cv2, argparse, os
import numpy as np
from image_iter import FaceDataset
import matplotlib.pyplot as plt


def main(args):
    print(args)
    MULTI_GPU = False
    DEVICE = torch.device("cuda:0")
    DATA_ROOT = args.data
    with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:
        NUM_CLASS, h, w = [int(i) for i in f.read().split(',')]
        
    if args.defian_layer:
        h = h*2
        w = w*2
        
    if args.network == "xxs":
        model = mobilevit.mobilevit_xxs((h, w), 512, args.defian_layer)
    elif args.network == "xs":
        model = mobilevit.mobilevit_xs((h, w), 512, args.defian_layer)
    elif args.network == "s":
        model = mobilevit.mobilevit_s((h, w), 512, args.defian_layer)
    else:
        raise ValueError("Only support mobilevit [xxs/xs/s]")

    model_root = args.model
    model.load_state_dict(torch.load(model_root))

    #debug
    w = torch.load(model_root)
    for x in w.keys():
        print(x, w[x].shape)
        
    #embed()
    TARGET = [i for i in args.target.split(',')]
    vers = get_val_data('./eval/', TARGET)
    acc = []

    for ver in vers:
        name, data_set, issame = ver
        accuracy, std, xnorm, best_threshold, roc_curve_tensor = perform_val(MULTI_GPU, DEVICE, 512, 
                                                                             args.batch_size, model, 
                                                                             data_set, issame)
        print('[%s]XNorm: %1.5f' % (name, xnorm))
        print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (name, accuracy, std))
        print('[%s]Best-Threshold: %1.5f' % (name, best_threshold))
        acc.append(accuracy)
        plt.figure(figsize=(8, 8))
        plt.imshow(roc_curve_tensor.numpy().transpose((1, 2, 0)))  # Ubah urutan dimensi jika perlu
        plt.axis('off')  # Matikan sumbu x dan y
        plt.title(name + 'ROC Curve')
        filename = name + '_ROC_curve.png'
        plt.savefig(filename, bbox_inches='tight')
        print('[%s]ROC Curve saved to [%s]' % (name, filename))
        plt.close()
    print('Average-Accuracy: %1.5f' % (np.mean(acc)))
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", 
                        help="model path", type=str)
    parser.add_argument("--data", default="./Data/ms1m-retinaface-t1/",
                        help="training set directory", type=str)
    parser.add_argument("--network", default="xxs",
                        help="which network, ['xxs','xs', 's']", type=str)
    parser.add_argument("--defian", action="store_true",
                        help="use defian layer, True/False")
    parser.add_argument("--target", default="lfw,talfw,sllfw,calfw,cplfw,cfp_fp,agedb_30",
                        help="verification targets", type=str)
    parser.add_argument("--batch_size", type=int, default=20, 
                        help="batch_size")
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))