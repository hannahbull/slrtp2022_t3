import configargparse

def save_opts(args, fn):
    with open(fn, 'w') as fw:
        for items in vars(args):
            fw.write('%s %s\n' % (items, vars(args)[items]))

def str2bool(v):
    v = v.lower()
    if v in ('yes', 'true', 't', '1'):
        return True
    elif v in ('no', 'false', 'f', '0'):
        return False
    raise ValueError('Boolean argument needs to be true or false. '
                        'Instead, it is %s.' % v)

def load_opts():

    parser = configargparse.ArgumentParser(description="main")
    parser.register('type', bool, str2bool)

    parser.add('-c', '--config', is_config_file=True, help='config file path')

    parser.add_argument('--gpu_id', type=str, default='0', help='-1: all, 0-7: GPU index')

    parser.add_argument('--resume', type=str, default=None, help='', nargs='+')
    parser.add_argument('--test_only', action='store_true', help='Run only evaluation')
    parser.add_argument('--n_workers', type=int, default=0, help='Num data workers')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--random_subset_data', type=int, default=1e9, help='Choose random subset of data (for debugging)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Checkpoints directory to save model')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')


    parser.add_argument('--logits_only', action='store_true', help='Use logits only to obtain pseudolabels (use bslcp_challeng_logits as data)')

    parser.add_argument('--vocab_file_loc', type=str, default='bslcp_vocab_981.json', help='List of queries to use at evaluation')

    parser.add_argument('--queries_eval_file', type=str, default='data/dev.json', help='List of queries to use at evaluation')

    parser.add_argument('--train_data_loc', type=str, default='track3_data/bslcp_challenge_data/train', help='Location of training data')
    parser.add_argument('--train_labels_loc', type=str, default='track3_data/bslcp_challenge_data/train', help='Location of training labels')
    parser.add_argument('--val_data_loc', type=str, default='track3_data/bslcp_challenge_data/val', help='Location of val data')
    parser.add_argument('--val_labels_loc', type=str, default='track3_data/bslcp_challenge_data/val', help='Location of val labels')
    parser.add_argument('--test_data_loc', type=str, default='track3_data/bslcp_challenge_data/test', help='Location of test data')

    parser.add_argument('--test_output_loc', type=str, default='res/submission_dev.csv', help='Location of test submission output')

    parser.add_argument('--model', type=str, default='spotter', help='Model type')
    parser.add_argument('--dataset', type=str, default='spotting', help='Dataset type')
    parser.add_argument('--trainer', type=str, default='trainer', help='Trainer type')

    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--grad_clip_norm', type=float, default=None, help='Grad clip norm')
 
    args = parser.parse_args()

    return args