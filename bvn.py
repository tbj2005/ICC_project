import numpy as np
import copy


def hungarian_max(matrix, size):
    """
    找到该矩阵的可分解最大权置换矩阵
    :param matrix:
    :param size:
    :return:
    """
    zero_num = np.count_nonzero(matrix)
    sort_index = np.argsort(- matrix, axis=None)[- zero_num:]
    sort_row_col = [(int(sort_index[i] / size), int(sort_index[i] % size)) for i in range(len(sort_index))]
    set_row_ava = [i for i in range(size)]
    set_col_ava = [i for i in range(size)]
    hungarian_index = []
    for i in range(zero_num):
        if len(set_row_ava) == 0 and len(set_col_ava) == 0:
            break
        (sort_row, sort_col) = sort_row_col[i]
        if sort_row in set_row_ava and sort_col in sort_row_col:
            hungarian_index.append((sort_row, sort_col))
    if len(set_row_ava) + len(set_col_ava) > 0:
        return np.zeros([size, size])
    else:



def diagonal_zero_stuff(raw_matrix, size):
    """
    将对角元素为 0 的矩阵进行填充，使其变为双随机矩阵。
    :param raw_matrix:
    :param size:
    :return:
    """
    matrix = copy.deepcopy(raw_matrix)
    max_ele = np.max(matrix)
    full_matrix = max_ele * np.ones([size, size]) - max_ele * np.eye(size)
    add_matrix = full_matrix - matrix

    return matrix


matrix_test = np.random.randint(1, 10, size=(4, 4))  # 生成随机正整数矩阵，取值范围为1到10
np.fill_diagonal(matrix_test, 0)  # 将对角线元素设为0
print(hungarian_max(matrix_test, 4))
