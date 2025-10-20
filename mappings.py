from cnn_trainer import NetworkConfiguration

#============== Mappings =================

def args_to_net_config(args, first_kernel, second_kernal):
    net_config = NetworkConfiguration()
    net_config.mode = args.mode
    net_config.model = args.model
    net_config.path = args.path
    net_config.dataset = args.dataset
    net_config.dataset_path = args.dataset_path
    net_config.first_layers = args.first_layers
    net_config.second_layers = args.second_layers
    net_config.first_kernel = first_kernel
    net_config.second_kernel = second_kernal
    net_config.iterate = args.iterate
    net_config.epochs = args.epochs
    net_config.compute = args.compute
    net_config.stride = args.stride
    net_config.dilation = args.dilation
    net_config.layers_num = args.layers_num
    net_config.channels = args.channels
    net_config.batch_size = args.batch_size
    return net_config

def config_to_args(args, config):
    args.model = config['model']
    args.path = config['path']
    args.dataset = config['dataset']
    args.dataset_path = config['dataset_path']
    args.first_layers = config['first_layers']
    args.second_layers = config['second_layers']
    args.layers_num = config['layers_num']
    args.dilation = config['dilation']
    args.stride = config['stride']
    args.iterate = config['iterate']
    args.epochs = config['epochs']
    args.compute = config['compute']
    args.channels = config['channels']
    args.batch_size = config['batch_size']
    return args