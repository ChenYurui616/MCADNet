# Record the loss function, Learning rate
# 2023.09.12

import time
import os
import logging

# Save the loss function
def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    fh.write('until_{}_run_iter_num{}\n'.format(epoch, whole_iter_num))
    fh.write('{}_epoch_total_loss:{}\n'.format(epoch, epoch_total_loss))
    fh.write('{}_epoch_loss:{}\n'.format(epoch, epoch_loss))
    fh.write('\n')
    fh.close()

# Save the learning rate
def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr{}\n'.format(update_lr_group['lr']))
    fh.write('decode:update:lr{}\n'.format(update_lr_group['lr']))
    fh.write('\n')
    fh.close()

def create_logger(model_name):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    if not os.path.exists('./log/{}'.format(model_name)):
        os.makedirs('./log/{}'.format(model_name), exist_ok=True)
    log_file = './log/{}/{}_.log'.format(model_name, time_str)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(
        filename=str(log_file),
        format=head,
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #console = logging.StreamHandler()          # 终端输出+文件保存
    #logging.getLogger('').addHandler(console)
    text = logging.FileHandler(log_file)    # 仅文件保存
    logging.getLogger('').addHandler(text)

    return logger

