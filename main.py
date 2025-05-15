import numpy as np

import Schedule_part


# 确定业务成环方案
def job_ring(fj, ufj, local_solution, single_traffic, pod, port):
    """
    确定业务成环方案
    :param fj: 固定拓扑业务索引
    :param ufj: 环拓扑业务索引
    :param local_solution: 业务放置方案
    :param single_traffic: 业务单连接流量
    :param pod: pod 数目
    :param port: 单 pod 端口数目
    :return: 返回所有业务的流量矩阵
    """
    job_num = len(local_solution)
    data_matrix_all = np.zeros(([pod, pod]))
    data_matrix_job = np.array([np.zeros([pod, pod]) for _ in range(job_num)])
    for i in fj:
        ps_local = local_solution[i][0]
        worker = local_solution[i][1]
        for j in worker:
            if j == ps_local:
                continue
            else:
                data_matrix_job[ps_local][j] += single_traffic[i]
                data_matrix_job[j][ps_local] += single_traffic[i]
        data_matrix_all += data_matrix_job[i]


# TPE 方案


# 分组方案


# 主函数

job_number = 30
job1 = Schedule_part.generate_job(job_number)
all_job_index = [job1[i][0] for i in range(0, len(job1))]
single_link_out, sum_traffic_out = Schedule_part.traffic_count(job1)
usage = 0.4
iter_num = 10
flop = 275
train_time = Schedule_part.job_set_train(job1, flop, usage)
pod_number = 8
b_link = 30
port_num = 8
t_recon = 0.1
solution_out, undeploy_out, fix_job, unfix_job = Schedule_part.deploy_server(all_job_index, job1, pod_number, 512, 4)
