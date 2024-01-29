import torch, os
import yaml
from IPython import embed


def get_config(args):
    configuration = dict(
        SEED = 1337,  # random seed for reproduce results
        INPUT_SIZE = (112, 112),  # support: (112, 112) and (224, 224) for Defian
        EMBEDDING_SIZE = 512,  # feature dimension
        DISP_FREQ = 50,
        VER_FREQ = 1000,
    )

    if args.workers_id == 'cpu' or not torch.cuda.is_available():
        configuration['GPU_ID'] = []
        print("check", args.workers_id, torch.cuda.is_available())
    else:
        configuration['GPU_ID'] = [int(i) for i in args.workers_id.split(',')]
    if len(configuration['GPU_ID']) == 0:
        configuration['DEVICE'] = torch.device('cpu')
        configuration['MULTI_GPU'] = False
    else:
        configuration['DEVICE'] = torch.device('cuda:%d' % configuration['GPU_ID'][0])
        if len(configuration['GPU_ID']) == 1:
            configuration['MULTI_GPU'] = False
        else:
            configuration['MULTI_GPU'] = True

    configuration['NUM_EPOCH'] = args.epochs
    configuration['BATCH_SIZE'] = args.batch_size

    if args.data_mode == 'retina':
        configuration['DATA_ROOT'] = './Data/ms1m-retinaface-t1/'
    elif args.data_mode == 'casia_mod':
        configuration['DATA_ROOT'] = './Data/faces_webface/'
    elif args.data_mode == 'casia':
        configuration['DATA_ROOT'] = './Data/faces_webface_112x112'
    elif args.data_mode == 'itb_face':
        configuration['DATA_ROOT'] = './Data/itb_face_112/'
    else:
        raise ValueError(args.data_mode)
    
    configuration['EVAL_PATH'] = './eval/'
    
    assert args.net in ['xxs', 'xs', 's']
    configuration['BACKBONE_NAME'] = args.net
    
    assert args.loss_type in ['adaface', 'arcface', 'cosface']
    configuration['HEAD_NAME'] = args.loss_type
    
    configuration['DEFIAN_LAYER'] = args.defian
    configuration['TARGET'] = [i for i in args.target.split(',')]

    if args.resume:
        configuration['BACKBONE_RESUME_ROOT'] = args.resume
    else:
        configuration['BACKBONE_RESUME_ROOT'] = ''  # the root to resume training from a saved checkpoint
    
    configuration['WORK_PATH'] = args.outdir  # the root to buffer your checkpoints
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    return configuration