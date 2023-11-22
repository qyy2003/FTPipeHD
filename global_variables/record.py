
recorder = None
forward_time = 0.0
backward_time = 0.0
time_per_batch = {}


def reset_time():
    global forward_time, backward_time
    forward_time = 0
    backward_time = 0


def set_recorder(recorder_):
    global recorder
    recorder = recorder_


def get_recorder():
    global recorder
    return recorder


def set_time_per_batch(iter_id, time, mode):
    global time_per_batch
    if mode == 0:
        # forward
        time_per_batch[iter_id] = time
    else:
        # backward
        time_per_batch[iter_id] = time - time_per_batch[iter_id]


def get_time_per_batch(iter_id):
    global time_per_batch
    return time_per_batch[iter_id]


def update_forward_time(time):
    global forward_time
    forward_time += time
    return


def update_backward_time(time):
    global backward_time
    backward_time += time
    return


def get_forward_time():
    global forward_time
    return forward_time


def get_backward_time():
    global backward_time
    return backward_time