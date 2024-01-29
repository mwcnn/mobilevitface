import os, argparse, sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter

from config import get_config
from image_iter import FaceDataset

from utils.utils import separate_irse_bn_paras, separate_resnet_bn_paras, separate_mobilefacenet_bn_paras
from utils.utils import get_val_data, perform_val, get_time, buffer_val, AverageMeter, train_accuracy

import time
from model import mobilevit, head
from IPython import embed
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from torchsummary import summary


def need_save(acc, highest_acc):
    do_save = False
    save_cnt = 0
    if acc[0] > 0.98:
        do_save = True
    for i, accuracy in enumerate(acc):
        if accuracy > highest_acc[i]:
            highest_acc[i] = accuracy
            do_save = True
        if i > 0 and accuracy >= highest_acc[i]-0.002:
            save_cnt += 1
    if save_cnt >= len(acc)*3/4 and acc[0]>0.99:
        do_save = True
    print("highest_acc:", highest_acc)
    return do_save


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for mobilevitface training parameter')
    parser.add_argument("-w", "--workers_id", default='cpu',
                        help="gpu ids or cpu", type=str)
    parser.add_argument("-e", "--epochs", default=125,
                        help="training epochs", type=int)
    parser.add_argument("-b", "--batch_size", default=256,
                        help="batch_size", type=int)
    parser.add_argument("-d", "--data_mode", default='casia_mod',
                        help="use which database, ['casia', 'casia_mod', 'retina', 'itb_face']", type=str)
    parser.add_argument("-n", "--net", default='s',
                        help="which network, ['xxs','xs', 's']", type=str)
    parser.add_argument("-l", "--loss_type", default='arcface',
                        help="loss type, ['adaface', 'arcface', 'cosface']", type=str)
    parser.add_argument("-t", "--target", default='lfw,talfw,calfw,cplfw,cfp_fp,agedb_30',
                        help="verification targets, ['lfw', 'talfw', 'calfw', 'cplfw', 'cfp_fp', 'agedb_30']",
                        type=str)
    parser.add_argument("-r", "--resume", default='',
                        help="resume training", type=str)
    parser.add_argument("-o", "--outdir", default='',
                        help="output dir", type=str)
    parser.add_argument("--defian",
                        help="use defian layer, True/False", action="store_true")
    
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    # Epoch parameters
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    args = parser.parse_args()
    
    #======= hyperparameters & data loaders =======#
    cfg = get_config(args)
    
    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)
    
    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train data are stored
    EVAL_PATH = cfg['EVAL_PATH'] # the parent root where your eval data are stored
    WORK_PATH = cfg['WORK_PATH'] # the root to buffer your checkpoints and to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support:  ['xxs', 'xs', 's']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['adaface', 'arcface', 'cosface']
    DEFIAN_LAYER = cfg['DEFIAN_LAYER'] # Use Defian layer or no
    INPUT_SIZE = cfg['INPUT_SIZE'] # support:  (128, 128), (256, 256)
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']
    DISP_FREQ = cfg['DISP_FREQ']
    VER_FREQ = cfg['VER_FREQ']
    BATCH_SIZE = cfg['BATCH_SIZE']
    NUM_EPOCH = cfg['NUM_EPOCH']
    DEVICE = cfg['DEVICE'] # use GPU or CPU
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    print('GPU_ID', GPU_ID)
    TARGET = cfg['TARGET'] # support ['lfw', 'talfw', 'calfw', 'cplfw', 'cfp_fp', 'agedb_30']
    
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    with open(os.path.join(WORK_PATH, 'config.txt'), 'w') as f:
        f.write(str(cfg))
    print("=" * 60)
    
    writer = SummaryWriter(os.path.join(WORK_PATH, "logs")) # writer for buffering intermedium results
    torch.backends.cudnn.benchmark = True
    
    with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:
        NUM_CLASS, h, w = [int(i) for i in f.read().split(',')]
        
    image_height, image_width = INPUT_SIZE
    
    if DEFIAN_LAYER:
        image_height = image_height / 2
        image_width = image_width / 2

    dataset = FaceDataset(os.path.join(DATA_ROOT, 'train.rec'), rand_mirror=True, target_size=(image_width, image_height))
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=len(GPU_ID), drop_last=True)
    
    TOTAL_TRAIN_DATA = len(dataset)
    
    print("Number of Training Classes: {}".format(NUM_CLASS))
    print("Total Training Data: {}".format(TOTAL_TRAIN_DATA))

    vers = get_val_data(EVAL_PATH, TARGET, INPUT_SIZE)
    highest_acc = [0.0 for t in TARGET]
    
    #======= model & loss & optimizer =======#
    if BACKBONE_NAME == "xxs":
        MVIT = mobilevit.mobilevit_xxs(INPUT_SIZE, EMBEDDING_SIZE, DEFIAN_LAYER)
    elif BACKBONE_NAME == "xs":
        MVIT = mobilevit.mobilevit_xs(INPUT_SIZE, EMBEDDING_SIZE, DEFIAN_LAYER)
    elif BACKBONE_NAME == "s":
        MVIT = mobilevit.mobilevit_s(INPUT_SIZE, EMBEDDING_SIZE, DEFIAN_LAYER)
    else:
        raise ValueError("Only support mobilevit [xxs/xs/s]")
    
    print("=" * 60)
    print(f"Mobilevit_{BACKBONE_NAME} Backbone Generated")
    print("=" * 60)
    
    # NLLLoss
    NLLLOSS = nn.CrossEntropyLoss()
    # IdentityLoss
    LMCL_LOSS = head.build_head(head_type=HEAD_NAME,
                                embedding_size=EMBEDDING_SIZE,
                                class_num=NUM_CLASS,
                                device_id=GPU_ID)
    
    criterion = [NLLLOSS, LMCL_LOSS]
    
    # All Optimizer
    OPTIMIZER = create_optimizer(args, MVIT)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)
    lr_scheduler, _ = create_scheduler(args, OPTIMIZER)
    
    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT:
        print("=" * 60)
        print(BACKBONE_RESUME_ROOT)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            MVIT.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        else:
            print("No Checkpoint Found at '{}' . Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT))
        print("=" * 60)
        
    if MULTI_GPU:
        # multi-GPU setting
        MVIT = nn.DataParallel(MVIT, device_ids = GPU_ID)
        MVIT = MVIT.to(DEVICE)
        LMCL_LOSS = nn.DataParallel(LMCL_LOSS, device_ids = GPU_ID)
        # LMCL_LOSS = LMCL_LOSS.to(DEVICE)
    else:
        # single-GPU setting
        MVIT = MVIT.to(DEVICE)
        # LMCL_LOSS = LMCL_LOSS.to(DEVICE)
    
    summary(MVIT, (3, image_width, image_height))
    
    #======= train & validation & save checkpoint =======#

    batch = 0  # batch index

    losses = AverageMeter()
    top1 = AverageMeter()
    
    MVIT.train()
    for epoch in range(NUM_EPOCH):
        lr_scheduler.step(epoch)
        last_time = time.time()
        
        for inputs, labels in iter(trainloader):
            # compute output
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()

            outputs, norms = MVIT(inputs.float())
            
            mlogits = criterion[1](outputs, norms, labels)
            loss = criterion[0](mlogits, labels)
            
            #print("outputs", outputs, outputs.data)
            # measure accuracy and record loss
            prec1= train_accuracy(outputs.data, labels, topk = (1,))

            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            
            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
            
            # dispaly training loss & acc every DISP_FREQ (buffer for visualization)
            if (((batch + 1) % DISP_FREQ == 0) and batch != 0) or (batch == (int(TOTAL_TRAIN_DATA / BATCH_SIZE) * NUM_EPOCH)):
                epoch_loss = losses.avg
                epoch_acc = top1.avg
                writer.add_scalar("Training/Training_Loss", epoch_loss, batch + 1)
                writer.add_scalar("Training/Training_Accuracy", epoch_acc, batch + 1)

                batch_time = time.time() - last_time
                last_time = time.time()

                print('Epoch {} Batch {}\t'
                      'Speed: {speed:.2f} samples/s\t'
                      'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch + 1, batch + 1, speed=inputs.size(0) * DISP_FREQ / float(batch_time),
                    loss=losses, top1=top1))
                #print("=" * 60)
                losses = AverageMeter()
                top1 = AverageMeter()

            if ((batch + 1) % VER_FREQ == 0) and batch != 0: #perform validation & save checkpoints (buffer for visualization)
                for params in OPTIMIZER.param_groups:
                    lr = params['lr']
                    break
                print("Learning rate %f"%lr)
                print("Perform Evaluation on", TARGET, ", and Save Checkpoints...")
                acc = []
                for ver in vers:
                    name, data_set, issame = ver
                    accuracy, std, xnorm, best_threshold, roc_curve = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, MVIT, data_set, issame)
                    buffer_val(writer, name, accuracy, std, xnorm, best_threshold, roc_curve, batch + 1)
                    print('[%s][%d]XNorm: %1.5f' % (name, batch+1, xnorm))
                    print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (name, batch+1, accuracy, std))
                    print('[%s][%d]Best-Threshold: %1.5f' % (name, batch+1, best_threshold))
                    acc.append(accuracy)

                # save checkpoints per epoch
                if need_save(acc, highest_acc):
                    if MULTI_GPU:
                        torch.save(MVIT.module.state_dict(), os.path.join(WORK_PATH, "Mobilevit_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch + 1, get_time())))
                    else:
                        torch.save(MVIT.state_dict(), os.path.join(WORK_PATH, "Mobilevit_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch + 1, get_time())))
                MVIT.train()  # set to training mode

            batch += 1 # batch index