class Recorder:
    """
        Wrapper to collect the train info
    """
    def __init__(self):
        self.total_num = 0
        self.correct_num = 0
        self.train_loss = 0
        self.counter = 0
    
    def update_record(self, num, correct, loss):
        self.total_num += num
        self.correct_num += correct
        self.train_loss += loss
        self.counter += 1
    
    def reset_record(self):
        self.__init__()