import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter


def clean_tensorboard():
    tensor_board_sess_path = 'C:/Users/Eaj59/AppData/Local/Temp/.tensorboard-info'
    temp_sess_files = os.listdir(tensor_board_sess_path)
    for i in temp_sess_files:
        temp_file_path = os.path.join(tensor_board_sess_path,i)
        os.remove(temp_file_path)


def create_tensor_board_callback(model_name):
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    run_id = model_name+'_'+run_id
    base_dir = os.getcwd()
    if 'log_dir' in os.listdir():
        pass
    else:
        os.mkdir('log_dir')
    log_dir_path = os.path.join(os.getcwd(),'log_dir')
    os.chdir(log_dir_path)
    os.mkdir(run_id)
    os.chdir(base_dir)
    model_run_path = os.path.join(log_dir_path,run_id)
    file_writer = SummaryWriter(model_run_path)
    return run_id, file_writer

