def raw_dataset(train):
    data_root = r"D:\zts\DCNDSC_open\process_data\data\nlz_NEPU.h5"
    from .read_dataset import DATASET
    dataset = DATASET(data_root, train)
    return dataset





