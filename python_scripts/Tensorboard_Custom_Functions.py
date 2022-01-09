def clean_tensorboard():
    import os
    tensor_board_sess_path = 'C:/Users/Eaj59/AppData/Local/Temp/.tensorboard-info/'
    temp_sess_files = os.listdir(tensor_board_sess_path)
    for i in temp_sess_files:
        temp_file_path = os.path.join(tensor_board_sess_path,i)
        os.remove(temp_file_path)


def create_tensor_board_callback(self,model_name):
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    run_id = model_name+'_'+run_id
    base_dir = 'C:/Users/Eaj59/Documents/RL_Projects/Project_2_DRL'
    os.chdir('C:/Users/Eaj59/Documents/RL_Projects/Project_2_DRL/log_dir')
    os.mkdir(run_id)
    os.chdir(run_id)
    text_file_name = 'model_summary_' + model_name +'.txt'
    text_file_name2 = 'model_hyper_parameters_' + model_name +'.txt'
    f = open(text_file_name,"w+")
    f.write(self.online.to_json())
    f.close()
    f2 = open(text_file_name2,"w+")
    f2.write(self.__dict__())
    f2.close()
    os.chdir(base_dir)
    root_log_dir = os.path.join(os.curdir,'log_dir')
    model_cb_path = os.path.join(root_log_dir,run_id)
    file_writer = tf.summary.create_file_writer(model_cb_path)

    return run_id, file_writer