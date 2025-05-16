import numpy as np
import copy


def hungarian_max(matrix, size):
    """
    找到该矩阵的可分解最大权置换矩阵
    :param matrix:
    :param size:
    :return:
    """
    nonzero_num = np.count_nonzero(matrix)
    sort_index = np.argsort(-1 * matrix, axis=None)[: nonzero_num]
    sort_row_col = [(int(sort_index[i] / size), int(sort_index[i] % size)) for i in range(len(sort_index))]
    set_row_use = []
    set_col_use = []
    hungarian_index = []
    flag = []
    for i in range(nonzero_num):
        if len(set_row_use) == size and len(set_col_use) == size:
            break
        (sort_row, sort_col) = sort_row_col[i]
        hungarian_index.append((sort_row, sort_col))
        if sort_row not in set_row_use:
            set_row_use.append(sort_row)
        set_row_use_copy = copy.deepcopy(set_row_use)
        if sort_col not in set_col_use:
            set_col_use.append(sort_col)
        set_col_use_copy = copy.deepcopy(set_col_use)
        flag.append((set_row_use_copy, set_col_use_copy))
    if len(set_row_use) + len(set_col_use) > 2 * size:
        return np.zeros([size, size])
    else:
        match_degree = []
        k = size
        while 1:
            if len(match_degree) == size:
                break
            else:
                match_degree.append(hungarian_index[-1])
                hungarian_index = [hungarian_index[i] for i in range(len(hungarian_index))
                                   if hungarian_index[i][0] != hungarian_index[-1][0] and
                                   hungarian_index[i][1] != hungarian_index[-1][1]]
        print(match_degree)



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
