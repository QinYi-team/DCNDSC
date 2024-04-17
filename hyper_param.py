#配置超参数
class Hyper_param(object):
    # tr_root = r"E:\code\deal with opening dataset\process_data\high_data_model.h5"

    batch_size = 64
    epoch = 50
    lr = 0.00005
    lr_re_epoch = 1
    lr_re_rate = 0.90

    print_every = 300

    device = 'cuda:0'

    #  DCNDSC
    model_param = {'kernel_num1': 64, 'kernel_num2': 128, 'kernel_num3': 256, 'kernel_num4': 1152, 'kernel_output': 32}

