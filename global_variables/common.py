logger = None
#urls = [
#    "http://localhost:5002",
#    "http://localhost:5003",
    #"http://localhost:5004",
    #"http://localhost:5005",
    # "http://localhost:5006"
#]
#workers = {
#    0: "http://localhost:5001"
#}

# workers = {
#     0: "http://10.15.90.46:50001"
# }
# urls = [
#     "http://10.15.90.46:50002",
#     "http://10.15.90.46:50003",
#     "http://10.15.198.53:50001",
#     "http://10.15.198.53:50002",
#     "http://10.15.198.53:50003",
# ]
IP="localhost"
# IP_Phone="192.168.125.176"
# urls = [
#     "http://"+IP_Phone+":50000"
# ]
urls = [
    #"http://10.192.208.238:5002",
    # "http://10.192.187.69:5002",
    # "http://192.168.137.90:50002"
    "http://"+IP+":5002",
    "http://"+IP+":5003"
]
workers = {
    '0': "http://"+IP+":50001"
}
device = "cpu"

stage_idx = -1
prev_idx = -1

sem = None

model_name = ""
model_args = {}

program_args_ = None


def set_model_name(model_name_):
    global model_name
    model_name = model_name_


def get_model_name():
    global model_name
    return model_name


def set_model_args(args_):
    global model_args
    model_args = args_


def get_model_args():
    global model_args
    return model_args


def get_workers():
    global workers
    return workers


def get_worker_num():
    global workers
    return len(workers)


def update_workers(workers_):
    global workers
    workers = workers_


def get_urls():
    global urls
    return urls


def log_message(msg):
    """
        Use the logger to log message into the file
    """
    global logger
    logger.log(msg)


def get_url_from_worker(stage_idx):
    global workers
    if stage_idx == len(workers):
        stage_idx = 0
    # print(workers)
    # print(stage_idx)
    # print(workers[str(stage_idx)])
    return workers[str(stage_idx)]


def set_stage_idx(idx):
    global stage_idx, prev_idx
    if stage_idx != -1:
        prev_idx = stage_idx
    stage_idx = idx


def get_stage_idx():
    global stage_idx
    return stage_idx


def get_prev_idx():
    global prev_idx
    return prev_idx


def get_semaphore():
    global sem
    return sem


def set_semaphore(sem_):
    global sem
    sem = sem_


def get_stage_idx():
    return stage_idx


def set_logger(logger_):
    global logger
    logger = logger_


def set_program_args(args):
    global program_args_
    program_args_ = args


def get_program_args():
    return program_args_


def get_is_checkpoint():
    return program_args_.checkpoint


def is_load_cp():
    return program_args_.load_cp

def set_device(str):
    global device
    device=str

def get_device():
    return device

