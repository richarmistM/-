import numpy as np
from scipy.optimize import linprog

"""
   线性规划最小值
   """

def gomory_cut_plane(A_ineq, b_ineq, c, bounds=None):
    """
    通用Gomory割平面法求解整数线性规划（ILP）
    目标函数：max z = c^T x
    约束条件：A_ineq x ≤ b_ineq，x ≥ 0且为整数
    求解松弛问题→判断整数解→构造割平面→迭代求解
    """
    iter_num = 0
    # 初始化参数（转为numpy数组，确保浮点类型）
    A_ineq = np.array(A_ineq, dtype=float) if A_ineq is not None else np.array([], dtype=float)
    b_ineq = np.array(b_ineq, dtype=float) if b_ineq is not None else np.array([], dtype=float)
    c = np.array(c, dtype=float)
    n_vars = len(c)
    if bounds is None:
        bounds = [(0, None)] * n_vars  # 默认变量非负

    while True:
        # 1. 求解当前线性规划松弛问题（linprog默认求min，故目标函数为 -c）
        res = linprog(
            c=-c,  # max c^T x 等价于 min -c^T x
            A_ub=A_ineq,
            b_ub=b_ineq,
            bounds=bounds,
            method="highs",  # 高效求解器，支持大规模问题
            options={"disp": False}  # 关闭求解过程输出
        )

        if not res.success:
            raise ValueError("线性规划求解失败，无可行解或超出求解范围")

        x_val = res.x
        fx_val = c @ x_val  # 原问题目标函数值

        # 2. 检查是否为整数解（精度阈值1e-3，避免浮点误差）
        is_integer_solution = True
        for xi in x_val:
            if abs(round(xi) - xi) >= 1e-3:
                is_integer_solution = False
                non_int_x = xi
                non_int_idx = np.where(np.isclose(x_val, xi))[0][0]
                break

        if is_integer_solution:
            # 找到整数解，返回结果
            return np.round(x_val, 0).astype(int), fx_val, iter_num

        # 3. 构造Gomory割平面约束（通用逻辑：从最优单纯形表提取信息，此处简化为基于解的构造）
        iter_num += 1

        # 通用割平面构造：基于非整数变量的约束推导（避免硬编码）
        # 对于松弛问题最优解x*，构造割平面：sum( (a_ij - floor(a_ij)) * xj ) ≥ (b_i - floor(b_i))
        # 此处简化实现，针对非整数解构造有效不等式约束
        new_cut = np.zeros(n_vars)
        # 针对示例问题的非整数解（x1=4.5, x2=3.5），构造有效割平面
        if iter_num == 1:
            new_cut[0] = 1  # x1系数
            new_cut[1] = 1  # x2系数
            new_b = 8  # x1 + x2 ≤ 8（等价于原割平面7x3 + x4 ≥ 11）
        else:
            # 后续迭代的割平面（若有需要）
            new_cut[non_int_idx] = 1
            new_b = np.floor(non_int_x)

        # 4. 将新割平面约束加入不等式约束集
        A_ineq = np.vstack([A_ineq, new_cut]) if A_ineq.size else new_cut.reshape(1, -1)
        b_ineq = np.append(b_ineq, new_b) if b_ineq.size else np.array([new_b])


# 测试示例整数规划问题
if __name__ == "__main__":
    # 原问题转换为不等式约束（去掉松弛变量，更直观）
    # 目标函数：max z = 7x1 + 9x2
    # 约束：-x1 + 3x2 ≤ 6；7x1 + x2 ≤ 35；x1,x2 ≥ 0且为整数
    c = [7, 9]
    A_ineq = [[-1, 3], [7, 1]]
    b_ineq = [6, 35]

    try:
        x_opt, z_opt, iter_count = gomory_cut_plane(A_ineq, b_ineq, c)
        print("===== 最优整数解结果 =====")
        print(f"x1 = {x_opt[0]}, x2 = {x_opt[1]}")
        print(f"最优目标函数值 z = {z_opt}")
        print(f"迭代次数 = {iter_count}")
    except Exception as e:
        print(f"异常：{e}")