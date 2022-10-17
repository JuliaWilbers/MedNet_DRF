
'''
Configs for training & testing
Written by Whalechen
'''

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        # default='./data',
        default='./toy_data',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--img_list',
        default='./data/train.txt',
        type=str,
        help='Path for image list file')
    parser.add_argument(
        '--results_file',
        default='./results/set_1.csv',
        type=str,
        help='Path for image list file')
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
        '--save_trails',
        default=False,
        help='Store trails')
    parser.add_argument(
        '--augmentation',
        default='False',
        help='Use augmentation')
    parser.add_argument(
        '--label_list',
        default='./toy_data/DRF_label_sets/set_1/test.txt',
        type=str,
        help='Path for label list file')
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
        '--save_intervals',
        default=50,
        type=int,
        help='Interation for saving model')
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
        '--im_dir',
        default='./toy_data/DRF_sets/set_1/test/',
        type=str,
        help='Image Directory')
    parser.add_argument(
        '--seg_dir',
        default='./toy_data/Segmentations',
        type=str,
        help='Segmentaion Directory')
    parser.add_argument(
        '--pretrain_path',
        default=None,
        type=str,
        help=
        'Path for pretrained model.')
    parser.add_argument(
        '--new_layer_names',
        default=['adaptive_avg_pool3d', 'flatten', 'linear', 'relu', 'sigmoid'],
        # default=['conv_seg'],
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
        '--set_name',
        default='set_1',
        type=str,
        help='Name of the current set')
    parser.add_argument(
        '--method',
        default='method2',
        type=str,
        help='Method of the current set')
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
    args.save_folder = "./trails/DRF_models/{}_{}_{}_{}".format(args.model, args.model_depth, args.set_name,
                                                                args.method)
    # args.results_train_file = "results/{}_{}_{}_{}.log".format(args.model, args.model_depth, args.phase, args.set_name)
    # args.results_test_file = "results/{}_{}_{}_{}.log".format(args.model, args.model_depth, args.phase, args.set_name)
    return args
