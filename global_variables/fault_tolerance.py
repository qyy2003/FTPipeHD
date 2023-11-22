from threading import Lock


start_iter_id = 0 # for restart the training process
fault_status = 0 # 0 for no fault, 1 for fault exists, 2 for fault is handling
data_pool = {}
term = 0 # when partition point updates or recovery happens, term changed

backward_timeout = 100000
total_batch_train_time = 0
alpha = 2

backup_interval = 100
global_backup_interval = 200
received_id = set() # the received id
timer_pool = {} 
weight_backup = dict(iter_id=-1, weight=None)

global_weight_lock_ = Lock()  # to guarantee the atomicity of the global weight
global_weight_backup_ = dict() # This is only used by master

needed_params = None # used by worker


def set_start_iter_id(_iter_id):
    global start_iter_id
    start_iter_id = _iter_id


def get_start_iter_id():
    global start_iter_id
    return start_iter_id

    
def get_fault_status():
    global fault_status
    return fault_status


def set_fault_status(status):
    global fault_status
    fault_status = status

    
def store_train_data(iter_id, data):
    global data_pool
    data_pool[iter_id] = data


def remove_train_data(iter_id):
    global data_pool
    if data_pool.get(iter_id) is not None:
        del data_pool[iter_id]

        
def get_train_data(iter_id):
    global data_pool
    if data_pool.get(iter_id) is not None:
        return data_pool[iter_id]
    else:
        return -1


def get_training_term():
    global term
    return term


def update_training_term():
    global term
    term += 1


def set_training_term(term_):
    global term
    term = term_
    

def get_backward_timeout():
    global backward_timeout
    return backward_timeout


def get_backup_interval():
    global backup_interval
    return backup_interval

    
def get_global_backup_interval():
    global global_backup_interval
    return global_backup_interval

    
def add_receive_id(iter_id):
    global received_id
    received_id.add(iter_id)


def is_received_id(iter_id):
    global received_id
    return iter_id in received_id


def remove_receive_id(iter_id):
    global received_id
    received_id.remove(iter_id)


def set_weight_backup(iter_id, weight):
    global weight_backup
    weight_backup['iter_id'] = iter_id
    weight_backup['weight'] = weight
    # print("!backing up idx{}")


def get_weight_backup():
    global weight_backup
    return weight_backup


def set_global_weight_backup(weight):
    global global_weight_lock_, global_weight_backup_
    with global_weight_lock_:
        global_weight_backup_.update(weight)


def get_global_weight_backup():
    global global_weight_backup_
    return global_weight_backup_


def store_needed_params(_params):
    global needed_params
    needed_params = _params


def get_needed_params():
    global needed_params
    return needed_params


def rm_needed_params():
    global needed_params
    needed_params = None


def update_backup_timeout(iter_id, time):
    """
        Update the backward timeout according to the moving average of the training time per batch
    """
    global total_batch_train_time, backward_timeout, alpha
    total_batch_train_time += time
    # backward_timeout = alpha * total_batch_train_time / (iter_id + 1)


def set_timer(iter_id, timer):
    """
        Store the timer function, and cancel it when the iter_id received
    """
    global timer_pool
    timer_pool[iter_id] = timer


def cancel_timer(iter_id):
    """
        Cancel the timer when receiving the iter_id
    """
    global timer_pool
    timer = timer_pool.pop(iter_id)
    timer.cancel()