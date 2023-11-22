from global_variables.common import get_stage_idx, get_url_from_worker, get_worker_num
from global_variables.training import get_partition_point, get_sub_model, get_total_layer
import network.fault_tolerance as ft_network
from utils.general import get_layer_from_point
from utils.tolist import state_dict_list

def replicate_weight(iter_id, replicate_type):
    """
        Perform the replication according to the replicate_type
        replicate_type: 0 for local replication, 1 for global replication
    """
    stage_idx = get_stage_idx()
    point = get_partition_point()
    start_layer, end_layer = get_layer_from_point(point, stage_idx)
    sub_model = get_sub_model() 
    weight_backup = {} # Convert weight to key-value map, key is the layer, value is the corresponding parameters

    if end_layer == -1:
        end_layer = get_total_layer() - 1
    
    for layer in range(start_layer, end_layer + 1):
        if layer < sub_model.origin_features_len:
            weight_backup[layer] = state_dict_list(sub_model.features[layer - start_layer].state_dict())
        else:
            # classifier
            if start_layer >= sub_model.origin_features_len:
                weight_backup[layer] = state_dict_list(sub_model.classifier[layer - start_layer].state_dict())
            else:
                weight_backup[layer] = state_dict_list(sub_model.classifier[layer - sub_model.origin_features_len].state_dict())

    if replicate_type == 0:
        # global replication
        next_idx = stage_idx + 1

        if next_idx == get_worker_num():
            next_idx = 0
        
        target_url = get_url_from_worker(next_idx)
        res = ft_network.send_backup_weight(target_url, iter_id, weight_backup)
    else:
        next_idx = 0
        target_url = get_url_from_worker(next_idx)

        res = ft_network.send_global_backup_weight(target_url, iter_id, weight_backup)