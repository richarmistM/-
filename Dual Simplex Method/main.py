'''
23122795 孙正涵 11-12周上机
'''
import numpy as np


def DSimplex_eye(A, b, c, tol=1e-10):
    """
    对偶单纯形法求解线性规划问题: max z = c^T x, s.t. Ax = b, x ≥ 0
    输入:
        A: 系数矩阵 (m×n, 包含单位矩阵, 秩为m)
        b: 右端项向量 (m×1, 所有元素≤0)
        c: 目标函数系数向量 (n×1)
        tol: 浮点数精度容错阈值
    输出:
        x_opt: 最优解 (n×1)
        fx_opt: 最优函数值
        iter: 迭代次数
    异常:
        ValueError: 问题无界或无可行解时抛出
    """
    # 类型转换与维度检查
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64).reshape(-1, 1)  # 确保是列向量
    c = np.array(c, dtype=np.float64).reshape(-1, 1)

    m, n = A.shape
    if m != len(b):
        raise ValueError(f"A的行数({m})与b的长度({len(b)})不匹配")
    if n != len(c):
        raise ValueError(f"A的列数({n})与c的长度({len(c)})不匹配")
    if not np.all(b <= tol):
        raise ValueError("b向量需满足所有元素≤0（对偶单纯形初始条件）")

    iter_num = 0  # 迭代次数
    # 初始化基变量索引（假设A的最后m列是单位矩阵）
    basis = list(range(n - m, n))  # 基变量列索引
    non_basis = list(range(n - m))  # 非基变量列索引

    while True:
        # 1. 检查原问题可行性（b≥-tol即认为可行）
        if np.all(b >= -tol):
            break

        # 2. 选择出基变量：b中最小元素对应的基变量
        out_basis_idx = np.argmin(b)  # 出基变量在基列表中的索引
        out_col = basis[out_basis_idx]  # 出基变量的列索引

        # 3. 计算基矩阵的逆和检验数
        B = A[:, basis]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            raise ValueError("基矩阵奇异，无法求逆（A的秩可能不足）")

        # 检验数 σ_j = c_j - c_B^T * B^{-1} * A_j （所有变量的检验数）
        c_B = c[basis]
        sigma = c - (c_B.T @ B_inv @ A).T  # 形状(n,1)
        sigma = sigma.flatten()  # 转为一维数组

        # 4. 计算B_inv*A（用于入基变量选择）
        B_inv_A = B_inv @ A
        # 出基行的系数（对应出基变量行）
        out_row_coeff = B_inv_A[out_basis_idx, :]

        # 5. 选择入基变量：θ = σ_j / out_row_coeff[j] (out_row_coeff[j] < -tol)
        theta_candidates = []
        valid_in_idxs = []  # 非基变量中符合条件的索引（在non_basis中的位置）

        for nb_idx, col in enumerate(non_basis):
            coeff = out_row_coeff[col]
            if coeff < -tol:  # 系数严格小于0才符合条件
                theta = sigma[col] / coeff
                theta_candidates.append(theta)
                valid_in_idxs.append(nb_idx)

        # 边界处理：无有效入基变量 → 问题无界
        if not valid_in_idxs:
            raise ValueError("无可行的入基变量，线性规划问题无界，无最优解")

        # 选择θ最小的非基变量作为入基变量
        min_theta_idx = np.argmin(theta_candidates)
        in_nb_idx = valid_in_idxs[min_theta_idx]  # 入基变量在non_basis中的索引
        in_col = non_basis[in_nb_idx]  # 入基变量的列索引

        # 6. 高斯消元更新b和基矩阵（避免重复求逆的误差）
        # 主元（出基行，入基列）
        pivot = B_inv_A[out_basis_idx, in_col]
        if abs(pivot) < tol:
            raise ValueError("主元接近0，数值不稳定")

        # 构造初等变换矩阵（行变换）
        # 步骤1：主元行归一化
        B_inv_A[out_basis_idx, :] /= pivot
        b[out_basis_idx, 0] /= pivot

        # 步骤2：消去其他行的入基列系数
        for i in range(m):
            if i != out_basis_idx:
                factor = B_inv_A[i, in_col]
                B_inv_A[i, :] -= factor * B_inv_A[out_basis_idx, :]
                b[i, 0] -= factor * b[out_basis_idx, 0]

        # 7. 更新基/非基变量索引
        basis[out_basis_idx] = in_col
        non_basis[in_nb_idx] = out_col

        iter_num += 1
        # 防止无限迭代（可选：根据实际场景调整最大迭代次数）
        if iter_num > 1000:
            raise RuntimeError("迭代次数超过1000次，可能存在数值问题或逻辑错误")

    # 构造最优解
    x_opt = np.zeros((n, 1))
    x_opt[basis] = b  # 基变量取b的值，非基变量为0
    fx_opt = (c.T @ x_opt).item()  # 最优函数值

    return x_opt, fx_opt, iter_num


# 自定义格式化最优解的函数
def format_opt_solution(x_opt, decimal_places=1):

    fmt_str = f"%.{decimal_places}f"
    formatted = []
    for val in x_opt.flatten():
        # 先按指定小数位格式化，再去掉末尾的.0（如果是整数）
        val_str = fmt_str % val
        if '.' in val_str and val_str.endswith('0'):
            val_str = val_str.rstrip('0').rstrip('.') if val_str.count('.') else val_str
        # 处理浮点数精度导致的-0.0情况
        if val_str == '-0' or val_str == '-0.0':
            val_str = '0'
        formatted.append(f"[{val_str}]")
    return formatted


# 测试示例
if __name__ == "__main__":
    # 测试用例（修正后的合法输入）
    A = np.array([
        [-1, -2, -1, 1, 0],
        [-2, 1, -3, 0, 1]
    ])
    b = np.array([[-3], [-4]])  # b≤0，满足初始条件
    c = np.array([[-2], [-3], [-4], [0], [0]])  # max z = -2x1 -3x2 -4x3

    try:
        x_opt, fx_opt, iter_num = DSimplex_eye(A, b, c)
        print("=== 对偶单纯形法求解结果 ===")
        # 格式化最优解输出
        formatted_x = format_opt_solution(x_opt, decimal_places=1)
        print("最优解 x_opt:")
        print('\n'.join(formatted_x))
        print(f"最优函数值 fx_opt: {fx_opt:.4f}")
        print(f"迭代次数: {iter_num}")
    except Exception as e:
        print(f"求解失败：{e}")