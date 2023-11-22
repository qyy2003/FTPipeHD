import numpy as np
from global_variables.training import get_total_layer
from global_variables.common import get_worker_num, get_url_from_worker
from global_variables.profiling import get_static_profiler

class DynamicScheduler:
    """
        Scheduler calculates the optimal partitioning point according to the profiler
    """
    def __init__(self) -> None:
        #  -> 常常出现在python函数定义的函数名后面，为函数添加元数据,描述函数的返回类型，从而方便开发人员使用。
        pass

    def calculate_partition_point_bk(self, is_average=False) -> list:
        total_layer = get_total_layer()
        stage_num = get_worker_num()

        # calculates the transmission time 
        profiler = get_static_profiler()
        transmission_time = [[0] * profiler.total_layers] * len(profiler.bandwidth)
        for idx, bw in profiler.bandwidth.items():
            for i, size in enumerate(profiler.output_size):
                transmission_time[idx][i] = 2 * size / bw
        
        # initialize
        partition_point = [0] * (stage_num - 1)
        dp = np.ones((profiler.total_layers, stage_num)) * float("inf")
        for i in range(0, total_layer):
            dp[i][0] = profiler.get_time_interval(0, i, 0) + profiler.get_time_interval(0, i, 1)
        
        # dynamic programming
        computing_power = profiler.computing_power
        for stage in range(1, stage_num):
            for j in range(stage, total_layer - stage_num + 1 + stage):
                for i in range(0, j):
                    print("Calculating the {} device training a {}-layer model with the last partition point is {}".format(stage + 1, j + 1, i + 1))
                    url = get_url_from_worker(stage)
                    
                    if is_average == True:
                        # 平均分配
                        last_stage_time = (profiler.get_time_interval(i + 1, j, 0) + profiler.get_time_interval(i + 1, j, 1))
                    else:
                        last_stage_time = (profiler.get_time_interval(i + 1, j, 0) + profiler.get_time_interval(i + 1, j, 1)) * computing_power[url] # 通过乘以计算力比例来预测时间
                    cur_trans_time = transmission_time[stage - 1][i]
                    slowest_time = max(dp[i][stage - 1], cur_trans_time, last_stage_time)
                    if slowest_time < dp[j][stage] and (stage == 1 or i > partition_point[stage - 1]):
                        dp[j][stage] = slowest_time
                        partition_point[stage - 1] = i
    
        print(partition_point)    
        return partition_point

    def calculate_partition_point(self, is_average=False) -> list:
        """
            Updated at Oct 16, 2022 
            Updated the backtrack way to get the partition point
        """
        total_layer = get_total_layer()
        stage_num = get_worker_num()

        # calculates the transmission time 
        profiler = get_static_profiler()
        transmission_time = [[0] * profiler.total_layers] * len(profiler.bandwidth)
        for idx, bw in profiler.bandwidth.items():
            for i, size in enumerate(profiler.output_size):
                transmission_time[idx][i] = 2 * size / bw
        
        # initialize
        partition_point = np.zeros((profiler.total_layers, stage_num))
        dp = np.ones((profiler.total_layers, stage_num)) * float("inf")
        for i in range(0, total_layer):
            dp[i][0] = profiler.get_time_interval(0, i, 0) + profiler.get_time_interval(0, i, 1)
        
        # dynamic programming
        computing_power = profiler.computing_power
        for stage in range(1, stage_num):
            for j in range(stage, total_layer - stage_num + 1 + stage):
                for i in range(0, j):
                    # print("Calculating the {} device training a {}-layer model with the last partition point is {}".format(stage, j, i))
                    url = get_url_from_worker(stage)
                    
                    if is_average == True:
                        # 平均分配
                        last_stage_time = (profiler.get_time_interval(i + 1, j, 0) + profiler.get_time_interval(i + 1, j, 1))
                    else:
                        last_stage_time = (profiler.get_time_interval(i + 1, j, 0) + profiler.get_time_interval(i + 1, j, 1)) * computing_power[url] # 通过乘以计算力比例来预测时间
                    cur_trans_time = transmission_time[stage - 1][i]
                    slowest_time = max(dp[i][stage - 1], cur_trans_time, last_stage_time)
                    if slowest_time < dp[j][stage]:
                        dp[j][stage] = slowest_time
                        partition_point[j][stage] = i
                        # print("Updating partition point {} at {}-layer model with {} device".format(i, j, stage))

        ret = [0] * (stage_num - 1)
        cur_layer = total_layer - 1
        for i in range(0, stage_num - 1):
            point = int(partition_point[cur_layer][stage_num - 1 - i])
            ret[stage_num - 2 - i] = point
            cur_layer = point
        
        print(ret)    
        return ret