class DefaultConfigs(object):
    seed = 666
    # SGD
    weight_decay = 5e-4
    momentum = 0.9
    # learning rate
    init_lr = 0.01
    lr_epoch_1 = 0
    lr_epoch_2 = 50
    # model
    pretrained = True
    model = 'resnet18'     # resnet18 or maddg
    # training parameters
    gpus = "3"
    batch_size = 10
    norm_flag = True
    max_iter = 4000
    lambda_triplet = 1
    lambda_adreal = 0.5
    # test model name
    tgt_best_model_name = 'model_best_0.08_29.pth.tar' 
    # source data information
    src1_data = 'oulu'
    src1_train_num_frames = 1
    src2_data = 'replay'
    src2_train_num_frames = 1
    src3_data = 'msu'
    src3_train_num_frames = 1
    # target data information
    tgt_data = 'casia'
    tgt_test_num_frames = 2
    # paths information
    checkpoint_path = './' + tgt_data + '_checkpoint/' + model + '/DGFANet/'
    best_model_path = './' + tgt_data + '_checkpoint/' + model + '/best_model/'
    logs = './logs/'

config = DefaultConfigs()
