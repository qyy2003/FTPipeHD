import torch
from global_variables.common import get_is_checkpoint, get_stage_idx
from models.mobilenetv2.model import MobileNetV2, SubMobileNetV2
from models.vgg.model import VGG, SubVGG
from models.har_cnn.model import HARCNN, SubHARCNN
from models.BERT.model import BERTForClassification,SubBERTForClassification,TryBert
from models.LLaMA.model import LLaMA,SubLLaMA
from models.GPT2.model import SubGPT2ForClassification

model_zoo = {
    'LLaMA':LLaMA,
    'BERTForClassification': BERTForClassification,#TryBert,
    'MobileNetV2': MobileNetV2,
    'VGG': VGG,
    'HARCNN': HARCNN
}

sub_model_zoo = {
    "GPT2":SubGPT2ForClassification,
    'LLaMALora':SubLLaMA,
    'BERTForClassification': SubBERTForClassification,
    'MobileNetV2': SubMobileNetV2,
    'VGG': SubVGG,
    'HARCNN': SubHARCNN
}

"""
    Init a model according to the model name specified in model_zoo
"""
def init_model(name, args):
    if model_zoo.get(name) is None:
            print("Model Name Error!")
            return 

    Model = model_zoo[name]
    model = Model(args)
    return model


"""
    Init a sub model according to the model name specified in sub_model_zoo
    The caller should specify the start point and end point
"""
def init_sub_model(name, args, start, end):
    if sub_model_zoo.get(name) is None:
        print("Model Name Error!")
        return 

    SubModel = sub_model_zoo[name]
    # print("name,start,end")
    sub_model = SubModel(start, end, args)
    # print(sub_model)
    # # for saving the initial model
    # if get_is_checkpoint():
    #     print("Saving the initial momdel weights")
    #     save_path = "./model_state/sub_model_%s.pkl" % get_stage_idx()
    #     torch.save(sub_model.state_dict(), save_path)
    return sub_model