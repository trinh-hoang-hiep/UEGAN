def get_trainer(option): 
    if option['uncer_method'].lower() == 'ganabp': 
        from trainer.trainer_ganabp import train_one_epoch 

    return train_one_epoch 
