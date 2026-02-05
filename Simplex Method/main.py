import numpy as np

def simplex_method(A, b, c):
    """
    单纯形法求解max Z = c·x，s.t. A·x = b，x≥0（A含单位矩阵，初始基可行）
    参数:
        A: 约束矩阵 (m×n)，每行对应一个等式约束
        b: 右端常数向量 (m,)，要求b≥0
        c: 目标函数系数向量 (n,)
    返回:
        x_opt: 最优解 (n,)
        z_opt: 最优目标函数值
        iter_num: 迭代次数
    """
    m, n = A.shape  # m个约束，n个变量
    if np.any(b < 0):
        raise ValueError("右端项b必须非负，本代码仅处理初始基可行解为正的情况")

    iter_num = 0
    # 1. 寻找初始基变量（A中的单位矩阵列）
    basis = []
    for col in range(n):
        col_vec = A[:, col]
        # 判断是否为单位向量（仅有一个1，其余为0）
        if np.sum(col_vec == 1) == 1 and np.sum(col_vec != 0) == 1:
            basis.append(col)
        if len(basis) == m:
            break
    if len(basis) != m:
        raise ValueError("矩阵A中未找到m个单位矩阵列，无法确定初始基变量")
    basis = np.array(basis)  # 基变量索引

    # 2. 构造初始单纯形表: [A | b] (m行) + [检验数σ | -Z] (1行)
    # 初始Z=0，检验数σ = c - c_B·A（c_B是基变量的目标系数）
    c_B = c[basis]
    z = c_B @ A  # z_j = Σc_Bi·aij
    sigma = c - z  # 检验数
    table = np.hstack([A, b.reshape(-1, 1)])  # 约束行
    obj_row = np.hstack([sigma, np.array([-0.0])])  # 目标行: [σ1,σ2,...,σn, -Z]
    table = np.vstack([table, obj_row])

    while True:
        # 3. 判断最优性：检验数中无正数值则最优
        sigma = table[-1, :-1]
        max_sigma = np.max(sigma)
        if max_sigma < 1e-9:  # 允许浮点误差，正检验数小于阈值则终止
            break

        # 4. 选择进基变量：最大正检验数对应的列
        enter_col = np.argmax(sigma)

        # 5. 选择离基变量：最小比值法（b_i / a_ij, a_ij>0）
        ratios = []
        for row in range(m):
            a_ij = table[row, enter_col]
            if a_ij > 1e-9:  # 系数为正才计算比值
                ratios.append(table[row, -1] / a_ij)
            else:
                ratios.append(np.inf)  # 系数非正，比值为无穷大
        min_ratio = np.min(ratios)
        if min_ratio == np.inf:
            raise ValueError("线性规划问题无界，不存在有限最优解")
        leave_row = np.argmin(ratios)  # 离基变量对应的行

        # 6. 转轴运算（高斯消元，使进基列变为单位向量）
        pivot = table[leave_row, enter_col]  # 主元
        # 主元行归一化
        table[leave_row, :] /= pivot
        # 消去其他行的进基列元素
        for row in range(m + 1):
            if row != leave_row:
                factor = table[row, enter_col]
                table[row, :] -= factor * table[leave_row, :]

        # 7. 更新基变量
        basis[leave_row] = enter_col
        iter_num += 1

    #无穷多解判断逻辑 - ---------------------#
    has_infinite_solutions = False  # 初始化标记
    sigma_final = table[-1, :-1]  # 获取最终的检验数
    # 找出所有非基变量的索引（不在basis中的变量）
    non_basis = [col for col in range(n) if col not in basis]
    # 检查是否存在非基变量的检验数在浮点误差范围内等于0
    for col in non_basis:
        if abs(sigma_final[col]) < 1e-9:
            has_infinite_solutions = True
            break

    # 8. 构造最优解
    x_opt = np.zeros(n)
    for row in range(m):
        x_opt[basis[row]] = table[row, -1]  # 基变量取b列的值
    z_opt = -table[-1, -1]  # 目标函数值是目标行最后一列的相反数

    return x_opt, z_opt, iter_num


# ------------------- 测试例题 -------------------
if __name__ == "__main__":
    # 例题的约束矩阵A、右端项b、目标系数c
    A = np.array([
        [2, -3, 2, 1, 0],  # 约束1: 2x1-3x2+2x3+x4=15
        [1 / 3, 1, 5, 0, 1]  # 约束2: (1/3)x1+x2+5x3+x5=20
    ], dtype=np.float64)
    b = np.array([15, 20], dtype=np.float64)
    c = np.array([1, 2, 1, 0, 0], dtype=np.float64)  # 目标函数：x1+2x2+x3

    # 调用单纯形法函数
    x_opt, z_opt, iter_num = simplex_method(A, b, c)

    # 输出结果
    print("最优解 x_opt:")
    for i in range(len(x_opt)):
        print(f"x_{i + 1} = {x_opt[i]:.4f}")
    print(f"最优目标函数值 Z = {z_opt:.4f}")
    print(f"迭代次数: {iter_num}")