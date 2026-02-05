import numpy as np  # 用于向量、矩阵运算


# 1. 定义目标函数（计算函数值+梯度）
def myexam1(x):
    # x是numpy数组，形状为(2,)，对应x1, x2
    f = x[0] ** 2 + 2 * (x[1] ** 2)  # 计算目标函数值
    g = np.array([2 * x[0], 4 * x[1]])  # 计算梯度：[2x1, 4x2]
    return f, g


# 2. 实现梯度下降函数（添加最大迭代次数，优化步长）
def SteepDescent(f_name, x0, tstep, eps, max_iter=10000):
    """
    梯度下降法求解目标函数最小值
    :param f_name: 目标函数（返回f值和梯度g）
    :param x0: 初始点（列表/数组）
    :param tstep: 固定步长（不再衰减，更稳定）
    :param eps: 精度阈值（梯度范数小于eps则终止）
    :param max_iter: 最大迭代次数（避免无限循环）
    :return: 最优解xstar, 最优函数值fxstar, 迭代次数iter_num
    """
    # 初始化
    x = np.array(x0, dtype=np.float64)  # 初始点，转成numpy数组
    iter_num = 0  # 迭代次数

    while iter_num < max_iter:  # 增加最大迭代次数限制
        # 1. 计算当前点的函数值和梯度
        f, g = f_name(x)

        # 2. 判断终止条件：梯度的L2范数 < eps
        grad_norm = np.linalg.norm(g)
        # 打印每10次迭代的信息，方便调试
        if iter_num % 10 == 0:
            print(f"迭代{iter_num}次 | 梯度范数：{grad_norm:.6f} | 当前x：{x} | 函数值：{f:.6f}")
        if grad_norm < eps:
            print(f"\n满足精度要求,最终梯度范数：{grad_norm:.6f}")
            break

        # 3. 固定步长（更稳定，避免衰减导致收敛停滞）
        alpha = tstep

        # 4. 更新x：x_new = x_old - alpha * 梯度
        x = x - alpha * g

        # 5. 迭代次数+1
        iter_num += 1

    # 最终结果补充打印
    final_f, final_g = f_name(x)
    final_grad_norm = np.linalg.norm(final_g)
    print(f"\n迭代结束 | 总迭代次数：{iter_num} | 最终梯度范数：{final_grad_norm:.6f}")

    # 返回最优解、最优函数值、迭代次数
    return x, final_f, iter_num


# 3. 调用函数求解
if __name__ == "__main__":
    # 调整步长为0.1（固定步长，收敛更快）
    xstar, fxstar, iter_num = SteepDescent(
        f_name=myexam1,
        x0=[1, 1],  # 初始点
        tstep=0.1,  # 固定步长（核心修改：取消衰减）
        eps=1e-3,  # 精度
        max_iter=10000  # 最大迭代次数（兜底）
    )

    # 输出最终结果（核心修改：将f格式改为e，即科学计数法）
    print("\n===== 最终结果 =====")
    print(f"最优解 xstar：[{xstar[0]:.6e}, {xstar[1]:.6e}]")  # 科学计数法，保留6位小数
    print(f"最优函数值 fxstar：{fxstar:.6e}")  # 函数值也改为科学计数法（可选，更统一）
    print(f"总迭代次数：{iter_num}")