def get_path_dict(hostname, task):
    path_dict = {}
    if task == 'COD':
        path_dict['image_root'] = '/UEGAN/data/camo/img/'
        path_dict['gt_root'] = '/UEGAN/data/camo/gt/'
        path_dict['test_dataset_root'] ='/UEGAN/data/camo/Atest/'

    return path_dict