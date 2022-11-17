
'''
Configs for training & testing
Written by Whalechen
'''

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        default='./data',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--label_list',
        default='./toy_data/DRF_label_sets/set_1/test.txt',
        type=str,
        help='Path for label list file')
    parser.add_argument(
        '--label_list_val',
        type=str,
        help ='Path for label list file validation set')
    parser.add_argument(
        '--results_file',
        default='./results/experiment.csv',
        type=str,
        help='Path for image list file')
    parser.add_argument(
        '--method',
        default='method2',
        type=str,
        help='Method of the current set')
    parser.add_argument(
        '--setnr',
        default='1',
        type=str,
        help='Set number')
    parser.add_argument(
        '--methodnr',
        default='3',
        type=str,
        help='Method number')
    parser.add_argument(
        '--version',
        default='1',
        type=str,
        help='Experiment version')
    parser.add_argument(
        '--augmentation',
        default='False',
        help='Use augmentation')
    parser.add_argument(
        '--save_trails',
        default=True,
        help='Store trails')
    parser.add_argument(
        '--save_intervals',
        default=10,
        type=int,
        help='Interation for saving model')
    parser.add_argument(
        '--n_seg_classes',
        default=2,
        type=int,
        help="Number of segmentation classes")
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=0.001,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='Number of jobs')
    parser.add_argument(
        '--batch_size', default=2, type=int, help='Batch Size')
    parser.add_argument(
        '--phase', default='train', type=str, help='Phase of train or test')
    parser.add_argument(
        '--n_epochs',
        default=200,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--input_D',
        # default=56,
        default=210,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        # default=448,
        default=140,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        # default=448,
        default=150,
        type=int,
        help='Input size of width')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help=
        'Path for resume model.')
    parser.add_argument(
        '--pretrain_path',
        default=None,
        type=str,
        help=
        'Path for pretrained model.')
    parser.add_argument(
        '--new_layer_names',
        #default=['maxpool2', 'flatten', 'linear', 'relu', 'sigmoid'],
        default=['classification1', 'classification2'],
        type=list,
        help='New layer except for backbone')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,
        help='Gpu id lists')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=10,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument(
        '--ci_test', action='store_true', help='If true, ci testing is used.')
    parser.add_argument(
        '--pretrained', action='store_true', help='If true, pre-trained weights used')
    args = parser.parse_args()
    args.save_folder = "./trails/DRF_models/method{}_v{}_set{}".format(args.method, args.version, args.setnr)
    return args
