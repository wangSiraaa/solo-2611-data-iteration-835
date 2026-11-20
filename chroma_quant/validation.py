"""验证入口：题面要求的全部案例 + 手算例 + 独立高精度积分核对。

运行：``python -m chroma_quant.validation``
所有断言通过则退出码为 0，并打印每个案例的核对数据。
"""
from __future__ import annotations

import os
import tempfile
from fractions import Fraction as F

from .errors import DomainError, LimitError, ValidationError
from .integration import adaptive_simpson_float, composite_simpson_fraction
from .meter_service import MeterService
from .blend import make_header, optimize_blend
from .quant import product_from_dict, product_to_dict
from .repository import QuantRepository
from .signal_service import SignalService
from .sync import (compute_sync_stats, make_channel_spec, make_clock,
                   sync_stats_from_dict, sync_stats_to_dict)

TOL = 1e-8
results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((name, bool(cond), detail))
    flag = "PASS" if cond else "FAIL"
    print(f"[{flag}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        raise AssertionError(name + " " + detail)


def nearly(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


# ----------------------------------------------------------------------
# 独立核对：对分析结果逐格用两条独立积分器复核每个组分面积
# ----------------------------------------------------------------------
def independent_cross_check(repo: QuantRepository, method_id: str,
                            product_id: str) -> tuple[float, float]:
    """两条独立积分器逐格复核每个组分面积。

    与主路径不共享积分代码：这里直接按组分支撑闭区间判断活跃性
    （网格已在支撑边界切段，格内非活跃组分权重恒 0），再分别用
    float 自适应 Simpson 与纯 Fraction 复合 Simpson 独立求积。
    """
    method = repo.get_method(method_id)
    product = repo.load_product(product_id)
    net, comps = product.net, method.components

    def active(c, lo: F, hi: F) -> bool:
        return c.t0 <= lo and hi <= c.t1

    def make_integrands(j: int, lo: F, hi: F):
        c = comps[j]
        is_active = active(c, lo, hi)

        def eval_share(t: F, *, at_endpoint: bool) -> list[F]:
            # 在格端点处，仅当组分对整格活跃（inside）才计其权重；
            # 仅在边界点接触的组分在格内部恒 0，端点的跳变是零测度的
            # 半开间断，数值积分必须用内部极限，否则污染端点权重。
            ws: list[F] = []
            for k, cc in enumerate(comps):
                if not active(cc, lo, hi):
                    ws.append(F(0))
                    continue
                if cc.t0 <= t <= cc.t1:
                    v = F(0)
                    for p, q in zip(cc.nodes, cc.nodes[1:]):
                        if p.t <= t <= q.t:
                            r = (t - p.t) / (q.t - p.t)
                            v = p.y + (q.y - p.y) * r
                            break
                    ws.append(v)
                else:
                    ws.append(F(0))
            return ws

        def f_float(t: float) -> float:
            if not is_active:
                return 0.0
            tf = F(t).limit_denominator(10 ** 18)
            ws = eval_share(tf, at_endpoint=True)
            W = sum(ws)
            if W == 0:
                return 0.0
            return float(net(tf) * ws[j] / W)

        def f_frac(t: F) -> F:
            if not is_active:
                return F(0)
            ws = eval_share(t, at_endpoint=True)
            W = sum(ws)
            if W == 0:
                return F(0)
            return net(t) * ws[j] / W

        return f_float, f_frac, is_active

    max_gap_simpson = 0.0
    max_gap_fraction = 0.0
    explain = repo.explain(method_id, product_id)
    for cell in explain["cells"]:
        lo = F(cell["t_lo"])
        hi = F(cell["t_hi"])
        if hi == lo:
            continue
        for j, c in enumerate(comps):
            f_float, f_frac, is_active = make_integrands(j, lo, hi)
            if not is_active:
                # 非活跃组分在该格必须为 0，与主路径记录核对
                recorded = cell["component_areas"].get(c.name, 0.0)
                max_gap_simpson = max(max_gap_simpson, abs(float(recorded)))
                continue
            s_float = adaptive_simpson_float(f_float, float(lo), float(hi),
                                             tol=1e-12)
            s_frac = float(composite_simpson_fraction(
                f_frac, lo, hi, tol=F(1, 10 ** 11)))
            closed_v = float(cell["component_areas"].get(c.name, 0.0))
            max_gap_simpson = max(max_gap_simpson,
                                  abs(s_float - closed_v))
            max_gap_fraction = max(max_gap_fraction,
                                   abs(s_frac - closed_v))
    return max_gap_simpson, max_gap_fraction


# ----------------------------------------------------------------------
# 案例 1：可手算的常数份额例（纯有理，精确守恒）
# ----------------------------------------------------------------------
def case_constant_share(db: str) -> None:
    svc = SignalService(db)
    # 净响应在 [0,4] 恒为 2：信号恒 2，基线恒 0
    svc.create_run("R1", [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)])
    v1 = svc.add_baseline("R1", [(0, 0), (4, 0)])
    product = svc.publish_product("R1", v1)

    repo = QuantRepository(db)
    # 两组分在整个域上常数权重 1 与 3 => 份额恒 1/4、3/4
    repo.create_method("M1", [
        {"name": "zeta", "factor": 2,
         "nodes": [(0, 1), (4, 1)]},
        {"name": "alpha", "factor": 5,
         "nodes": [(0, 3), (4, 3)]},
    ])
    res = repo.run_analysis("M1", product.product_id)

    # 手算：总面积 = 2*4 = 8；面积 2 与 6；浓度 1 与 6/5
    check("C1 手算总面积", nearly(res.total_area, 8.0),
          f"total={res.total_area}")
    check("C1 手算组分面积",
          nearly(res.component_area["zeta"], 2.0)
          and nearly(res.component_area["alpha"], 6.0),
          str(res.component_area))
    check("C1 手算浓度",
          nearly(res.concentration["zeta"], 1.0)
          and nearly(res.concentration["alpha"], 1.2),
          str(res.concentration))
    check("C1 未分配为零", nearly(res.unallocated_area, 0.0))
    check("C1 精确守恒(零残差)", res.conservation_residual == 0.0,
          f"residual={res.conservation_residual:.3e}")
    # 名称身份：返回顺序按方法定义而非字典序（alpha 字典序在前）
    check("C1 名称顺序即身份",
          list(res.component_area) == ["zeta", "alpha"])
    repo.close()
    svc.close()


# ----------------------------------------------------------------------
# 案例 2：基线跨零 + 交点不落采样点（真实切段）
# ----------------------------------------------------------------------
def case_crossing_root_off_grid(db: str) -> None:
    svc = SignalService(db)
    # 信号恒 2；基线 -2→6（斜率 2，t=1 处跨零）：d=4-2t，根在 t=2，
    # t=2 既非基线锚点（锚点只有 0、4）也不是“端点”，验证真实切段
    svc.create_run("R2", [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)])
    v = svc.add_baseline("R2", [(0, -2), (4, 6)])
    product = svc.publish_product("R2", v)
    net = product.net
    # 真正的交点：值为零且前驱折点严格为正（与后续零段区分）
    root_knots = [
        k for prev, k in zip(net.knots, net.knots[1:])
        if k.y == 0 and prev.y > 0
    ]
    check("C2 交点真实切段(t=2)", len(root_knots) == 1
          and root_knots[0].t == F(2), str([(str(k.t), str(k.y))
                                            for k in net.knots]))
    # 手算：净响应是 4-2t 在 [0,2] 上的三角，面积 = 4*2/2 = 4
    check("C2 三角面积=4", nearly(float(net.integral()), 4.0),
          f"integral={float(net.integral())}")
    # 禁止“采样点截零后连线”：t=3/2 在采样点之间，净值必须为 1
    check("C2 子采样点精确插值", net(F(3, 2)) == F(1))
    # 根之后整段显式为零（零段保留），且基线在 t=1 跨零
    check("C2 根后显式零段", net(F(3)) == 0 and net(F(4)) == 0)
    svc.close()


# ----------------------------------------------------------------------
# 案例 3：交点恰好落在采样点上
# ----------------------------------------------------------------------
def case_root_on_sample(db: str) -> None:
    svc = SignalService(db)
    # 信号 0,1,2,3,4；基线恒 2 => d=t-2，根 t=2 恰为采样点
    svc.create_run("R3", [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
    v = svc.add_baseline("R3", [(0, 2), (4, 2)])
    product = svc.publish_product("R3", v)
    net = product.net
    # 净响应在 [2,4] 为 t-2，面积 = 2*2/2 = 2
    check("C3 采样点交点切段",
          any(k.t == F(2) and k.y == 0 for k in net.knots))
    check("C3 面积=2", nearly(float(net.integral()), 2.0),
          str(float(net.integral())))
    check("C3 t=2 左侧净零", net(F(1)) == 0 and net(F(2)) == 0)
    svc.close()


# ----------------------------------------------------------------------
# 案例 4：很窄的归属窗口，边界不对齐任何采样/折点
# ----------------------------------------------------------------------
def case_narrow_window(db: str) -> None:
    svc = SignalService(db)
    svc.create_run("R4", [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)])
    v = svc.add_baseline("R4", [(0, 0), (4, 0)])
    product = svc.publish_product("R4", v)
    repo = QuantRepository(db)
    # 窗口 [1/3, 2/3]，权重 1；其余时间 W=0
    repo.create_method("M4", [
        {"name": "narrow", "factor": 1,
         "nodes": [(F(1, 3), 1), (F(2, 3), 1)]},
    ])
    res = repo.run_analysis("M4", product.product_id)
    expect_area = 4.0 * (F(2, 3) - F(1, 3))  # 4/3
    check("C4 窄窗面积 4/3（不依赖步长命中）",
          nearly(res.component_area["narrow"], float(expect_area)),
          f"{res.component_area['narrow']} vs {float(expect_area)}")
    check("C4 未分配=总面积-窄窗",
          nearly(res.unallocated_area,
                 res.total_area - float(expect_area)))
    g1, g2 = independent_cross_check(repo, "M4", product.product_id)
    check("C4 自适应Simpson一致", g1 < 1e-9, f"max_gap={g1:.2e}")
    check("C4 纯Fraction Simpson一致", g2 < 1e-9, f"max_gap={g2:.2e}")
    repo.close()
    svc.close()


# ----------------------------------------------------------------------
# 案例 5：重叠峰 + 零权重间隙（共同响应不得重复取走）
# ----------------------------------------------------------------------
def case_overlap_and_gap(db: str) -> None:
    svc = SignalService(db)
    # 净响应：[0,6] 上的三角山，峰高 6，中点 3 => 总面积 18
    svc.create_run("R5", [(0, 0), (1, 2), (2, 4), (3, 6),
                          (4, 4), (5, 2), (6, 0)])
    v = svc.add_baseline("R5", [(0, 0), (6, 0)])
    product = svc.publish_product("R5", v)
    repo = QuantRepository(db)
    repo.create_method("M5", [
        # A 覆盖 [0,3] 权重恒 1
        {"name": "A", "factor": 1, "nodes": [(0, 1), (3, 1)]},
        # B 覆盖 [2,6] 权重恒 1；[2,3] 与 A 重叠 => 各分一半；
        # (3, ...] 只有 B；A、B 都不覆盖的间隙不存在于本例，
        # 另加 C 的支撑 [5, 9/2] 结束后 [9/2,6] 仅 B。
        {"name": "B", "factor": 1, "nodes": [(2, 1), (6, 1)]},
        # C 只在 [4, 9/2] 有窗口，制造重叠与“窗口结束后 W 仍正”的边界
        {"name": "C", "factor": 3,
         "nodes": [(4, 1), (F(9, 2), 1)]},
    ])
    res = repo.run_analysis("M5", product.product_id)
    # 守恒：组分面积和 + 未分配 = 总面积 18
    s = sum(res.component_area.values())
    check("C5 总面积=18", nearly(res.total_area, 18.0))
    check("C5 重叠不重复取走(守恒)",
          nearly(s + res.unallocated_area, 18.0),
          f"sum={s:.10f} unalloc={res.unallocated_area:.10f}")
    check("C5 无未分配(全程W>0)",
          nearly(res.unallocated_area, 0.0))
    # 零权重间隙：另造一个带真实间隙的方法
    repo.create_method("M5gap", [
        {"name": "L", "factor": 1, "nodes": [(0, 1), (1, 1)]},
        {"name": "R", "factor": 1, "nodes": [(5, 1), (6, 1)]},
    ])
    res2 = repo.run_analysis("M5gap", product.product_id)
    # 间隙 (1,5) 全部未分配
    check("C5 零权重间隙=>未分配>0", res2.unallocated_area > 0,
          f"unallocated={res2.unallocated_area:.6f}")
    check("C5 间隙守恒",
          nearly(sum(res2.component_area.values())
                 + res2.unallocated_area, 18.0))
    g1, g2 = independent_cross_check(repo, "M5", product.product_id)
    check("C5 自适应Simpson一致", g1 < 1e-9, f"max_gap={g1:.2e}")
    check("C5 纯Fraction Simpson一致", g2 < 1e-9, f"max_gap={g2:.2e}")
    repo.close()
    svc.close()


# ----------------------------------------------------------------------
# 案例 6：不同输入分段、同函数 => 同指纹、缓存命中
# ----------------------------------------------------------------------
def case_equivalent_segmentation(db: str) -> None:
    repo = QuantRepository(db)
    # 净响应恒 1 的两种分段：两点 vs 五点（中间点全共线）
    d1 = {
        "product_id": "EXT-A", "run_id": "RX", "baseline_version": 1,
        "signal": {"knots": [{"t": "0/1", "y": "1/1"},
                             {"t": "4/1", "y": "1/1"}]},
        "baseline": {"knots": [{"t": "0/1", "y": "0/1"},
                               {"t": "4/1", "y": "0/1"}]},
        "net": {"knots": [{"t": "0/1", "y": "1/1"},
                          {"t": "4/1", "y": "1/1"}]},
    }
    d2 = {
        "product_id": "EXT-B", "run_id": "RY", "baseline_version": 7,
        "signal": {"knots": [{"t": "0/1", "y": "1/1"},
                             {"t": "1/1", "y": "1/1"},
                             {"t": "2/1", "y": "1/1"},
                             {"t": "3/1", "y": "1/1"},
                             {"t": "4/1", "y": "1/1"}]},
        "baseline": {"knots": [{"t": "0/1", "y": "0/1"},
                               {"t": "2/1", "y": "0/1"},
                               {"t": "4/1", "y": "0/1"}]},
        "net": {"knots": [{"t": "0/1", "y": "1/1"},
                          {"t": "2/1", "y": "1/1"},
                          {"t": "4/1", "y": "1/1"}]},
    }
    id1 = repo.import_product(d1)
    id2 = repo.import_product(d2)
    p1, p2 = repo.load_product(id1), repo.load_product(id2)
    check("C6 同函数指纹相同", p1.fingerprint() == p2.fingerprint())
    check("C6 等价产品导入去重", id1 == id2, f"{id1[:8]} == {id2[:8]}")

    repo.create_method("M6", [
        {"name": "only", "factor": 1, "nodes": [(0, 1), (4, 1)]}])
    before = len(repo.list_analyses())
    r1 = repo.run_analysis("M6", id1)
    r2 = repo.run_analysis("M6", id2)  # 应命中缓存
    after = len(repo.list_analyses())
    check("C6 同函数分析缓存命中", after - before == 1,
          f"新增分析数={after - before}")
    check("C6 缓存结果一致",
          nearly(r1.component_area["only"], r2.component_area["only"]))
    # 外部坏产品：净响应与信号/基线不一致 => 拒绝
    bad = product_from_dict(d1)
    bad_d = product_to_dict(bad)
    bad_d["net"] = {"knots": [{"t": "0/1", "y": "9/1"},
                              {"t": "4/1", "y": "9/1"}]}
    bad_d["product_id"] = "EXT-BAD"
    try:
        repo.import_product(bad_d)
        check("C6 坏外部产品被拒绝", False)
    except ValidationError:
        check("C6 坏外部产品被拒绝", True)
    repo.close()


# ----------------------------------------------------------------------
# 案例 7：更正基线锚点 => 结果改变；旧方法/产品/结果仍可复读
# ----------------------------------------------------------------------
def case_baseline_correction(db: str) -> None:
    svc = SignalService(db)
    svc.create_run("R7", [(0, 10), (2, 10), (4, 10)])
    v1 = svc.add_baseline("R7", [(0, 0), (4, 0)])
    v2 = svc.correct_baseline("R7", [(0, 4), (4, 4)])  # 基线上移
    check("C7 基线版本追加", v1 == 1 and v2 == 2
          and svc.list_baseline_versions("R7") == [1, 2])
    p_old = svc.publish_product("R7", v1)
    p_new = svc.publish_product("R7", v2)
    check("C7 新旧产品不同",
          p_old.product_id != p_new.product_id
          and p_old.fingerprint() != p_new.fingerprint())

    repo = QuantRepository(db)
    repo.create_method("M7", [
        {"name": "k", "factor": 2, "nodes": [(0, 1), (4, 1)]}])
    r_old = repo.run_analysis("M7", p_old.product_id)
    r_new = repo.run_analysis("M7", p_new.product_id)
    check("C7 更正后组分结果改变",
          not nearly(r_old.component_area["k"],
                     r_new.component_area["k"]),
          f"{r_old.component_area['k']} -> {r_new.component_area['k']}")
    # 手算：旧净面积 40，新净面积 (10-4)*4=24；浓度 = 面积/2
    check("C7 旧结果手算=40/浓度20",
          nearly(r_old.total_area, 40.0)
          and nearly(r_old.concentration["k"], 20.0))
    check("C7 新结果手算=24/浓度12",
          nearly(r_new.total_area, 24.0)
          and nearly(r_new.concentration["k"], 12.0))
    # 复读旧对象
    reread = svc.get_product(p_old.product_id)
    check("C7 旧信号产品可复读",
          reread.fingerprint() == p_old.fingerprint())
    # 旧分析结果仍可复读：按 (方法, 旧产品) 取回，面积仍是更正前的 40
    old_result = repo.find_analysis("M7", p_old.product_id)
    check("C7 旧分析结果仍可复读",
          nearly(old_result.component_area["k"], 40.0)
          and nearly(old_result.concentration["k"], 20.0))
    repo.close()
    svc.close()


# ----------------------------------------------------------------------
# 案例 8：有效域拒绝外推
# ----------------------------------------------------------------------
def case_domain_rejection(db: str) -> None:
    svc = SignalService(db)
    svc.create_run("R8", [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)])
    # 锚点域 [1,3] 严格窄于采样域
    svc.add_baseline("R8", [(1, 0), (3, 0)])
    try:
        svc.query("R8", 0, 2)
        check("C8 域外查询拒绝", False)
    except DomainError:
        check("C8 域外查询拒绝", True)
    q = svc.query("R8", 1, 3)
    check("C8 域内查询成功", q["coverage"] == (F(1), F(3)))
    # 锚点超出采样范围 => 拒绝
    try:
        svc.add_baseline("R8", [(0, 0), (5, 0)])
        check("C8 超范围锚点拒绝", False)
    except DomainError:
        check("C8 超范围锚点拒绝", True)
    # 只有一个锚点 => 拒绝
    try:
        svc.add_baseline("R8", [(1, 0)])
        check("C8 单锚点拒绝", False)
    except ValidationError:
        check("C8 单锚点拒绝", True)
    svc.close()


# ----------------------------------------------------------------------
# 案例 9：规模上限
# ----------------------------------------------------------------------
def case_limits(db: str) -> None:
    svc = SignalService(db)
    svc.create_run("R9", [(i, 1) for i in range(5)])
    try:
        svc.create_run("R9big", [(i, 1) for i in range(501)])
        check("C9 500采样上限", False)
    except LimitError:
        check("C9 500采样上限", True)
    try:
        svc.add_baseline("R9", [(i, 0) for i in range(21)])
        check("C9 20锚点上限", False)
    except LimitError:
        check("C9 20锚点上限", True)
    svc.add_baseline("R9", [(0, 0), (4, 0)])
    p = svc.publish_product("R9")

    repo = QuantRepository(db)
    try:
        repo.create_method("M9big", [
            {"name": f"c{i}", "factor": 1, "nodes": [(0, 1), (4, 1)]}
            for i in range(11)])
        check("C9 10组分为上限", False)
    except LimitError:
        check("C9 10组分为上限", True)
    try:
        repo.create_method("M9node", [
            {"name": "c", "factor": 1,
             "nodes": [(i, 1) for i in range(31)]}])
        check("C9 30节点上限", False)
    except LimitError:
        check("C9 30节点上限", True)
    repo.create_method("M9", [
        {"name": "c", "factor": 1, "nodes": [(0, 1), (4, 1)]}])
    # 50 次分析上限：用不同产品指纹制造新分析
    for k in range(50):
        d = {
            "product_id": f"LIM-{k}", "run_id": f"LIMR{k}",
            "baseline_version": 1,
            "signal": {"knots": [{"t": "0/1", "y": f"{k+1}/1"},
                                 {"t": "4/1", "y": f"{k+1}/1"}]},
            "baseline": {"knots": [{"t": "0/1", "y": "0/1"},
                                   {"t": "4/1", "y": "0/1"}]},
            "net": {"knots": [{"t": "0/1", "y": f"{k+1}/1"},
                              {"t": "4/1", "y": f"{k+1}/1"}]},
        }
        pid = repo.import_product(d)
        repo.run_analysis("M9", pid, use_cache=False)
    try:
        d = {
            "product_id": "LIM-51", "run_id": "LIMR51",
            "baseline_version": 1,
            "signal": {"knots": [{"t": "0/1", "y": "100/1"},
                                 {"t": "4/1", "y": "100/1"}]},
            "baseline": {"knots": [{"t": "0/1", "y": "0/1"},
                                   {"t": "4/1", "y": "0/1"}]},
            "net": {"knots": [{"t": "0/1", "y": "100/1"},
                              {"t": "4/1", "y": "100/1"}]},
        }
        repo.run_analysis("M9", repo.import_product(d), use_cache=False)
        check("C9 50次分析上限", False)
    except LimitError:
        check("C9 50次分析上限", True)
    repo.close()
    svc.close()


# ----------------------------------------------------------------------
# 案例 10：输入校验（严格递增、非负权重、正因子、重名）
# ----------------------------------------------------------------------
def case_input_validation(db: str) -> None:
    svc = SignalService(db)
    try:
        svc.create_run("BAD1", [(0, 1), (0, 2)])
        check("C10 时间严格递增", False)
    except ValidationError:
        check("C10 时间严格递增", True)
    svc.create_run("R10", [(0, 1), (1, 1), (2, 1)])
    svc.add_baseline("R10", [(0, 0), (2, 0)])
    repo = QuantRepository(db)
    try:
        repo.create_method("BADW", [
            {"name": "x", "factor": 1, "nodes": [(0, -1), (2, 1)]}])
        check("C10 权重非负", False)
    except ValidationError:
        check("C10 权重非负", True)
    try:
        repo.create_method("BADF", [
            {"name": "x", "factor": 0, "nodes": [(0, 1), (2, 1)]}])
        check("C10 正响应因子", False)
    except ValidationError:
        check("C10 正响应因子", True)
    try:
        repo.create_method("BADNAME", [
            {"name": "x", "factor": 1, "nodes": [(0, 1), (2, 1)]},
            {"name": "x", "factor": 1, "nodes": [(0, 1), (2, 1)]}])
        check("C10 组分名称唯一", False)
    except ValidationError:
        check("C10 组分名称唯一", True)
    repo.close()
    svc.close()


# ----------------------------------------------------------------------
# 案例 11：净响应折点与权重折点交错的通用高精度核对
# ----------------------------------------------------------------------
def case_generic_cross_check(db: str) -> None:
    svc = SignalService(db)
    # 信号与基线都有多个折点，净响应含内部零点与折点
    svc.create_run("R11", [(0, 1), (1, 5), (2, 2), (3, 6),
                           (4, 1), (5, 4), (6, 2)])
    svc.add_baseline("R11", [(0, 3), (2, 1), (4, 3), (6, 1)])
    p = svc.publish_product("R11")
    repo = QuantRepository(db)
    repo.create_method("M11", [
        # 权重节点刻意与净响应折点交错、部分落在半整数时刻
        {"name": "p", "factor": F(3, 2),
         "nodes": [(F(1, 2), 2), (F(5, 2), 0), (4, 3), (6, 1)]},
        {"name": "q", "factor": F(7, 3),
         "nodes": [(0, 1), (1, 4), (3, 0), (F(11, 2), 2), (6, 2)]},
    ])
    res = repo.run_analysis("M11", p.product_id)
    check("C11 守恒",
          nearly(sum(res.component_area.values())
                 + res.unallocated_area, res.total_area),
          f"residual={res.conservation_residual:.2e}")
    g1, g2 = independent_cross_check(repo, "M11", p.product_id)
    check("C11 三路径一致(自适应Simpson)", g1 < 1e-9,
          f"max_gap={g1:.2e}")
    check("C11 三路径一致(纯Fraction Simpson)", g2 < 1e-9,
          f"max_gap={g2:.2e}")
    # 解释入口可复读且组分顺序保持
    expl = repo.explain("M11", p.product_id)
    check("C11 解释轨迹顺序", expl["component_order"] == ["p", "q"])
    check("C11 解释覆盖每格",
          sum(F(c["t_hi"]) - F(c["t_lo"]) for c in expl["cells"])
          == F(6))
    repo.close()
    svc.close()


# ----------------------------------------------------------------------
# 多路同步统计与受限合成（S 系列）
# ----------------------------------------------------------------------
def _mk_product(svc: SignalService, run: str, samples, anchors):
    svc.create_run(run, samples)
    v = svc.add_baseline(run, anchors)
    return svc.publish_product(run, v)


def _sync_float_cross_check(specs, stats) -> float:
    """独立路径：直接对 net(clock^{-1}(τ)) 自适应 Simpson 数值积分，
    复核同步统计的二阶矩（不经过 mapped_net 的复合代码）。"""
    max_gap = 0.0
    for i in range(len(specs)):
        for j in range(i, len(specs)):
            def f(tau, i=i, j=j):
                t = F(tau)  # float 的二进制精确值
                return (float(specs[i].product.net(
                            specs[i].clock.to_channel(t)))
                        * float(specs[j].product.net(
                            specs[j].clock.to_channel(t))))
            got = sum(
                adaptive_simpson_float(f, float(lo), float(hi), tol=1e-13)
                for lo, hi in stats.common_segments)
            max_gap = max(max_gap,
                          abs(got - float(stats.second_moments[i][j])))
    return max_gap


# S1：题面手算例 A=t、B=2−t、D=4
def case_sync_worked_example(db: str) -> None:
    svc = SignalService(db)
    pa = _mk_product(svc, "SA", [(0, 0), (1, 1), (2, 2)], [(0, 0), (2, 0)])
    pb = _mk_product(svc, "SB", [(0, 2), (1, 1), (2, 0)], [(0, 0), (2, 0)])
    svc.close()
    ms = MeterService(db)
    sel = [
        {"name": "A", "product_id": pa.product_id, "windows": [[0, 2]],
         "clock": [[0, 0], [2, 2]]},
        {"name": "B", "product_id": pb.product_id, "windows": [[0, 2]],
         "clock": [[0, 0], [2, 2]]},
    ]
    stats = ms.compute_sync(sel, 0, 2)
    check("S1 共同有效段", stats.common_segments == ((F(0), F(2)),),
          str(stats.common_segments))
    check("S1 总测长=2", stats.total_length == F(2))
    check("S1 均值手算", stats.means == (F(1), F(1)), str(stats.means))
    check("S1 一阶矩积分", stats.first_moments == (F(2), F(2)))
    check("S1 两两乘积积分",
          stats.second_moments[0][0] == F(8, 3)
          and stats.second_moments[0][1] == F(4, 3)
          and stats.second_moments[1][1] == F(8, 3))
    check("S1 总体协方差手算",
          stats.covariance == ((F(1, 3), F(-1, 3)),
                               (F(-1, 3), F(1, 3))),
          str(stats.covariance))
    check("S1 各通道版本入档",
          [c.baseline_version for c in stats.channels] == [1, 1]
          and stats.channels[0].product_id == pa.product_id)

    plan = ms.optimize(stats.sync_id, make_header("H1", ["A", "B"], 4))
    check("S1 最优各两份", plan.shares == (2, 2), str(plan.shares))
    check("S1 合成方差为0", plan.variance == F(0))
    check("S1 预期均值=1", plan.expected_mean == F(1))
    check("S1 枚举配置数=5", plan.configurations == 5)
    check("S1 交叉项求和=方差",
          sum(t.value for t in plan.terms) == plan.variance)
    # 固定统计产品换表头：A 至少三份（只读已存统计，不重读折线）
    plan2 = ms.optimize(stats.sync_id,
                        make_header("H2", ["A", "B"], 4, min_shares=[3, 0]))
    check("S1 A至少三份→3:1", plan2.shares == (3, 1), str(plan2.shares))
    check("S1 方差=1/12", plan2.variance == F(1, 12), str(plan2.variance))
    check("S1 两方案同一统计产品",
          plan.sync_id == plan2.sync_id and plan.plan_id != plan2.plan_id)
    # 指纹核查与篡改拒绝
    d = sync_stats_to_dict(stats)
    again = sync_stats_from_dict(d)
    check("S1 导出重组指纹一致",
          again.fingerprint == stats.fingerprint
          and again.sync_id == stats.sync_id)
    bad = dict(d)
    bad["second_moments"] = [["8/1", "4/3"], ["4/3", "8/3"]]
    try:
        sync_stats_from_dict(bad)
        check("S1 篡改矩被拒绝", False)
    except ValidationError:
        check("S1 篡改矩被拒绝", True)
    bad2 = dict(d)
    bad2["means"] = ["9/1", "1/1"]
    try:
        sync_stats_from_dict(bad2)
        check("S1 篡改派生量被拒绝", False)
    except ValidationError:
        check("S1 篡改派生量被拒绝", True)
    ms.close()


# S2：固定原表头，B 的源信号改为恒定 1 → 全部选 B
def case_sync_constant_channel(db: str) -> None:
    svc = SignalService(db)
    pa = _mk_product(svc, "CA", [(0, 0), (1, 1), (2, 2)], [(0, 0), (2, 0)])
    pb = _mk_product(svc, "CB", [(0, 1), (1, 1), (2, 1)], [(0, 0), (2, 0)])
    svc.close()
    ms = MeterService(db)
    sel = [
        {"name": "A", "product_id": pa.product_id, "windows": [[0, 2]],
         "clock": [[0, 0], [2, 2]]},
        {"name": "B", "product_id": pb.product_id, "windows": [[0, 2]],
         "clock": [[0, 0], [2, 2]]},
    ]
    stats = ms.compute_sync(sel, 0, 2)
    check("S2 恒定路方差与协方差为0",
          stats.covariance[1][1] == 0 and stats.covariance[0][1] == 0
          and stats.covariance[0][0] == F(1, 3))
    plan = ms.optimize(stats.sync_id, make_header("H1", ["A", "B"], 4))
    check("S2 固定表头全选恒定路", plan.shares == (0, 4), str(plan.shares))
    check("S2 方差0均值1",
          plan.variance == 0 and plan.expected_mean == 1)
    ms.close()


# S3：非线性时钟段（分段线性锚点映射，斜率变化）
def case_sync_nonlinear_clock(db: str) -> None:
    svc = SignalService(db)
    pc = _mk_product(svc, "NLC", [(0, 0), (1, 1), (2, 2)], [(0, 0), (2, 0)])
    pd = _mk_product(svc, "NLD", [(0, 0), (3, 3)], [(0, 0), (3, 0)])
    svc.close()
    # C：净响应 u（通道时钟），时钟 (0,0),(1,2),(2,3)：两段斜率 2 与 1
    spec_c = make_channel_spec("C", pc, [(0, 2)],
                               make_clock([(0, 0), (1, 2), (2, 3)]))
    spec_d = make_channel_spec("D", pd, [(0, 3)],
                               make_clock([(0, 0), (3, 3)]))
    stats = compute_sync_stats([spec_c, spec_d], 0, 3)
    check("S3 非线性时钟共同段",
          stats.common_segments == ((F(0), F(3)),))
    check("S3 均值手算", stats.means == (F(5, 6), F(3, 2)),
          str(stats.means))
    check("S3 协方差手算",
          stats.covariance == ((F(11, 36), F(17, 36)),
                               (F(17, 36), F(3, 4))),
          str(stats.covariance))
    gap = _sync_float_cross_check([spec_c, spec_d], stats)
    check("S3 独立数值积分复核", gap < 1e-9, f"max_gap={gap:.2e}")
    # 有效窗（通道时钟 [1/2, 3/2]）经非线性时钟映到公共时间 [1, 5/2]
    spec_e = make_channel_spec("E", pc, [(F(1, 2), F(3, 2))],
                               make_clock([(0, 0), (1, 2), (2, 3)]))
    stats_e = compute_sync_stats([spec_e], 0, 3)
    check("S3 窗映射到公共时间",
          stats_e.common_segments == ((F(1), F(5, 2)),),
          str(stats_e.common_segments))
    check("S3 窗内均值手算", stats_e.means == (F(11, 12),),
          str(stats_e.means))
    # 时钟覆盖窄于净响应域：不外推，共同段被裁短
    spec_f = make_channel_spec("F", pc, [(0, 2)],
                               make_clock([(0, 0), (F(3, 2), F(3, 2))]))
    stats_f = compute_sync_stats([spec_f], 0, 3)
    check("S3 时钟覆盖外不外推",
          stats_f.common_segments == ((F(0), F(3, 2)),),
          str(stats_f.common_segments))


# S4：三路仅两两有交叠（整组无共同测度）；单点交集不提供测长
def case_sync_pairwise_only_overlap(db: str) -> None:
    svc = SignalService(db)
    pa = _mk_product(svc, "PA", [(0, 1), (4, 1)], [(0, 0), (4, 0)])
    pb = _mk_product(svc, "PB", [(0, 1), (4, 1)], [(0, 0), (4, 0)])
    pc = _mk_product(svc, "PC", [(0, 1), (4, 1)], [(0, 0), (4, 0)])
    svc.close()
    ms = MeterService(db)
    clock = [[0, 0], [4, 4]]
    sel3 = [
        {"name": "A", "product_id": pa.product_id, "windows": [[0, 2]],
         "clock": clock},
        {"name": "B", "product_id": pb.product_id, "windows": [[1, 3]],
         "clock": clock},
        {"name": "C", "product_id": pc.product_id,
         "windows": [["5/2", 4]], "clock": clock},
    ]
    try:
        ms.compute_sync(sel3, 0, 4)
        check("S4 三路无共同测度拒绝", False)
    except DomainError:
        check("S4 三路无共同测度拒绝", True)
    # 允许重新选取通道：两两可行
    ab = ms.compute_sync(sel3[:2], 0, 4)
    check("S4 重选A,B共同段", ab.common_segments == ((F(1), F(2)),),
          str(ab.common_segments))
    bc = ms.compute_sync([sel3[1], sel3[2]], 0, 4)
    check("S4 重选B,C共同段",
          bc.common_segments == ((F(5, 2), F(3)),),
          str(bc.common_segments))
    try:
        ms.compute_sync([sel3[0], sel3[2]], 0, 4)
        check("S4 A,C不相交拒绝", False)
    except DomainError:
        check("S4 A,C不相交拒绝", True)
    # 单点交集不提供测长
    sel_pt = [
        {"name": "A", "product_id": pa.product_id, "windows": [[0, 2]],
         "clock": clock},
        {"name": "B", "product_id": pb.product_id, "windows": [[2, 4]],
         "clock": clock},
    ]
    try:
        ms.compute_sync(sel_pt, 0, 4)
        check("S4 单点交集无测长", False)
    except DomainError:
        check("S4 单点交集无测长", True)
    check("S4 重选后统计有效",
          ab.total_length == F(1) and ab.means == (F(1), F(1)))
    ms.close()


# S5：独立噪声叠加对角、参与通道数上限、并列取字典序最小
def case_blend_noise_and_ties(db: str) -> None:
    svc = SignalService(db)
    p1 = _mk_product(svc, "K1", [(0, 2), (2, 2)], [(0, 0), (2, 0)])
    p2 = _mk_product(svc, "K2", [(0, 5), (2, 5)], [(0, 0), (2, 0)])
    svc.close()
    ms = MeterService(db)
    sel = [
        {"name": "X", "product_id": p1.product_id, "windows": [[0, 2]],
         "clock": [[0, 0], [2, 2]]},
        {"name": "Y", "product_id": p2.product_id, "windows": [[0, 2]],
         "clock": [[0, 0], [2, 2]]},
    ]
    stats = ms.compute_sync(sel, 0, 2)
    check("S5 恒定路零协方差",
          all(stats.covariance[i][j] == 0
              for i in range(2) for j in range(2)))
    # 独立噪声互不相关 → 对角叠加 → 迫使分散
    plan = ms.optimize(stats.sync_id, make_header(
        "HN", ["X", "Y"], 2, noise_variances=[1, 1]))
    check("S5 噪声迫使分散", plan.shares == (1, 1), str(plan.shares))
    check("S5 噪声方差=1/2", plan.variance == F(1, 2))
    check("S5 预期均值=7/2", plan.expected_mean == F(7, 2))
    check("S5 噪声项贡献",
          sum(t.value for t in plan.terms
              if t.label.startswith("noise")) == F(1, 2))
    # 无噪声全并列 → 固定通道次序下份额向量字典序最小
    plan0 = ms.optimize(stats.sync_id, make_header("H0", ["X", "Y"], 2))
    check("S5 并列取字典序最小", plan0.shares == (0, 2), str(plan0.shares))
    check("S5 字典序最小者均值", plan0.expected_mean == F(5))
    # 参与通道数上限=1：并列仍取字典序最小
    plan1 = ms.optimize(stats.sync_id, make_header(
        "HK", ["X", "Y"], 2, max_participants=1, noise_variances=[1, 1]))
    check("S5 参与上限=1", plan1.shares == (0, 2), str(plan1.shares))
    check("S5 单路噪声方差=1", plan1.variance == F(1))
    # 噪声不对称 → 选更安静的一路
    planq = ms.optimize(stats.sync_id, make_header(
        "HQ", ["X", "Y"], 2, max_participants=1,
        noise_variances=[3, F(1, 2)]))
    check("S5 选更安静通道",
          planq.shares == (0, 2) and planq.variance == F(1, 2))
    ms.close()


# S6：不可配说明与规模上限
def case_blend_infeasible_and_limits(db: str) -> None:
    svc = SignalService(db)
    p1 = _mk_product(svc, "I1", [(0, 1), (2, 1)], [(0, 0), (2, 0)])
    p2 = _mk_product(svc, "I2", [(0, 2), (2, 2)], [(0, 0), (2, 0)])
    svc.close()
    ms = MeterService(db)
    sel = [
        {"name": "X", "product_id": p1.product_id, "windows": [[0, 2]],
         "clock": [[0, 0], [2, 2]]},
        {"name": "Y", "product_id": p2.product_id, "windows": [[0, 2]],
         "clock": [[0, 0], [2, 2]]},
    ]
    stats = ms.compute_sync(sel, 0, 2)
    p_lo = ms.optimize(stats.sync_id, make_header(
        "BL", ["X", "Y"], 4, min_shares=[3, 3]))
    check("S6 下限和超D不可配",
          not p_lo.feasible and "下限" in p_lo.reason, p_lo.reason)
    p_k = ms.optimize(stats.sync_id, make_header(
        "BK", ["X", "Y"], 4, min_shares=[1, 1], max_participants=1))
    check("S6 参与上限不可配",
          not p_k.feasible and "参与通道数上限" in p_k.reason, p_k.reason)
    p_hi = ms.optimize(stats.sync_id, make_header(
        "BH", ["X", "Y"], 4, max_shares=[1, 1]))
    check("S6 上限和不足不可配",
          not p_hi.feasible and "上限" in p_hi.reason, p_hi.reason)
    p_top = ms.optimize(stats.sync_id, make_header(
        "BT", ["X", "Y"], 4, max_shares=[2, 2], max_participants=1))
    check("S6 参与上限下装不下", not p_top.feasible, p_top.reason)
    check("S6 不可配方案可复读",
          ms.get_plan(p_lo.plan_id)["feasible"] is False)
    try:
        make_header("BD", ["X", "Y"], 17)
        check("S6 D≤16", False)
    except LimitError:
        check("S6 D≤16", True)
    try:
        make_header("BN", [f"c{i}" for i in range(7)], 4)
        check("S6 表头≤6路", False)
    except LimitError:
        check("S6 表头≤6路", True)
    specs7 = [make_channel_spec(f"c{i}", p1, [(0, 2)],
                                make_clock([(0, 0), (2, 2)]))
              for i in range(7)]
    try:
        compute_sync_stats(specs7, 0, 2)
        check("S6 同步≤6路", False)
    except LimitError:
        check("S6 同步≤6路", True)
    ms.close()


# S7：离线交换、换表头不重读折线、源版本复查与不可变历史
def case_sync_offline_exchange(db: str) -> None:
    import json as _json

    svc = SignalService(db)
    pa = _mk_product(svc, "OA", [(0, 0), (1, 1), (2, 2)], [(0, 0), (2, 0)])
    pb = _mk_product(svc, "OB", [(0, 2), (1, 1), (2, 0)], [(0, 0), (2, 0)])
    ms = MeterService(db)
    sel = [
        {"name": "A", "product_id": pa.product_id, "windows": [[0, 2]],
         "clock": [[0, 0], [2, 2]]},
        {"name": "B", "product_id": pb.product_id, "windows": [[0, 2]],
         "clock": [[0, 0], [2, 2]]},
    ]
    stats = ms.compute_sync(sel, 0, 2)
    header = make_header("HX", ["A", "B"], 4)
    plan = ms.optimize(stats.sync_id, header)

    # 离线交换：全新空库只导入统计 JSON（库中无任何折线产品）
    other = db + ".other"
    ms2 = MeterService(other)
    sid = ms2.import_sync(_json.loads(ms.export_sync_json(stats.sync_id)))
    check("S7 离线导入同id", sid == stats.sync_id)
    svc2 = SignalService(other)
    check("S7 离线库无任何折线产品", svc2.list_products() == [])
    svc2.close()
    plan2 = ms2.optimize(sid, header)
    check("S7 离线合成一致",
          plan2.shares == plan.shares and plan2.variance == plan.variance
          and plan2.plan_id == plan.plan_id)
    ms2.close()

    # 源版本复查：更正 A 的基线 → 新版本、新产品、新统计；旧对象保留
    v2 = svc.correct_baseline("OA", [(0, 1), (2, 1)])
    pa2 = svc.publish_product("OA", v2)
    check("S7 基线版本追加", svc.list_baseline_versions("OA") == [1, 2])
    reread = svc.get_product(pa.product_id)
    check("S7 旧产品复读校验", reread.fingerprint() == pa.fingerprint())
    sel2 = [dict(sel[0], product_id=pa2.product_id), sel[1]]
    stats2 = ms.compute_sync(sel2, 0, 2)
    check("S7 新源得新统计", stats2.sync_id != stats.sync_id)
    check("S7 各通道版本随源更新",
          [c.baseline_version for c in stats2.channels] == [2, 1])
    old = ms.get_sync(stats.sync_id)
    check("S7 旧统计保留旧版本",
          [c.baseline_version for c in old.channels] == [1, 1])
    check("S7 统计列表含两者",
          set(ms.list_syncs()) == {stats.sync_id, stats2.sync_id})
    # 同一统计产品挂多个表头方案（更换表头不重读折线）
    ms.optimize(stats.sync_id,
                make_header("HY", ["A", "B"], 4, min_shares=[3, 0]))
    check("S7 多表头方案同统计", len(ms.list_plans(stats.sync_id)) == 2)
    ms.close()
    svc.close()


# S8：输入校验
def case_sync_input_validation(db: str) -> None:
    svc = SignalService(db)
    pa = _mk_product(svc, "VA", [(0, 1), (2, 1)], [(0, 0), (2, 0)])
    pb = _mk_product(svc, "VB", [(0, 1), (2, 1)], [(0, 0), (2, 0)])
    svc.close()
    try:
        make_clock([(0, 0), (1, 1), (2, 1)])
        check("S8 公共时间严格递增", False)
    except ValidationError:
        check("S8 公共时间严格递增", True)
    try:
        make_clock([(0, 0), (0, 1)])
        check("S8 通道时间严格递增", False)
    except ValidationError:
        check("S8 通道时间严格递增", True)
    try:
        make_clock([(0, 0)])
        check("S8 时钟至少两锚点", False)
    except ValidationError:
        check("S8 时钟至少两锚点", True)
    clock = make_clock([(0, 0), (2, 2)])
    try:
        make_channel_spec("A", pa, [(2, 2)], clock)
        check("S8 窗起点<终点", False)
    except ValidationError:
        check("S8 窗起点<终点", True)
    try:
        make_channel_spec("A", pa, [], clock)
        check("S8 至少一个有效窗", False)
    except ValidationError:
        check("S8 至少一个有效窗", True)
    sa = make_channel_spec("A", pa, [(0, 2)], clock)
    try:
        compute_sync_stats([sa, make_channel_spec("A", pb, [(0, 2)], clock)],
                           0, 2)
        check("S8 通道名称唯一", False)
    except ValidationError:
        check("S8 通道名称唯一", True)
    try:
        compute_sync_stats([sa], 2, 2)
        check("S8 请求范围a<b", False)
    except ValidationError:
        check("S8 请求范围a<b", True)
    stats = compute_sync_stats(
        [sa, make_channel_spec("B", pb, [(0, 2)], clock)], 0, 2)
    try:
        optimize_blend(stats, make_header("HM", ["B", "A"], 2))
        check("S8 表头通道次序匹配", False)
    except ValidationError:
        check("S8 表头通道次序匹配", True)
    try:
        make_header("Hn", ["A"], 2, noise_variances=[F(-1)])
        check("S8 噪声非负", False)
    except ValidationError:
        check("S8 噪声非负", True)
    try:
        make_header("Hm", ["A"], 2, min_shares=[2], max_shares=[1])
        check("S8 下限≤上限", False)
    except ValidationError:
        check("S8 下限≤上限", True)
    try:
        make_header("Hp", ["A"], 2, max_participants=0)
        check("S8 参与上限≥1", False)
    except ValidationError:
        check("S8 参与上限≥1", True)
    try:
        clock.to_common(F(3))
        check("S8 时钟域外拒绝", False)
    except DomainError:
        check("S8 时钟域外拒绝", True)


# S9：请求范围的裁剪与多窗合并
def case_sync_requested_range(db: str) -> None:
    svc = SignalService(db)
    pa = _mk_product(svc, "RA", [(0, 0), (1, 1), (2, 2)], [(0, 0), (2, 0)])
    svc.close()
    ms = MeterService(db)
    sel = [{"name": "A", "product_id": pa.product_id, "windows": [[0, 2]],
            "clock": [[0, 0], [2, 2]]}]
    stats = ms.compute_sync(sel, 0, 1)
    check("S9 请求范围裁剪共同段",
          stats.common_segments == ((F(0), F(1)),))
    check("S9 裁剪后均值", stats.means == (F(1, 2),))
    check("S9 裁剪后方差", stats.covariance == ((F(1, 12),),),
          str(stats.covariance))
    stats2 = ms.compute_sync(sel, 1, 3)
    check("S9 超出部分裁剪不报错",
          stats2.common_segments == ((F(1), F(2)),))
    try:
        ms.compute_sync(sel, 3, 4)
        check("S9 窗外请求拒绝", False)
    except DomainError:
        check("S9 窗外请求拒绝", True)
    sel2 = [{"name": "A", "product_id": pa.product_id,
             "windows": [[0, "1/2"], ["1/2", 1], ["3/2", 2]],
             "clock": [[0, 0], [2, 2]]}]
    stats3 = ms.compute_sync(sel2, 0, 2)
    check("S9 相接窗合并",
          stats3.common_segments == ((F(0), F(1)), (F(3, 2), F(2)),),
          str(stats3.common_segments))
    check("S9 多段测长", stats3.total_length == F(3, 2))
    ms.close()


def main() -> int:
    cases = [
        case_constant_share, case_crossing_root_off_grid,
        case_root_on_sample, case_narrow_window, case_overlap_and_gap,
        case_equivalent_segmentation, case_baseline_correction,
        case_domain_rejection, case_limits, case_input_validation,
        case_generic_cross_check,
        case_sync_worked_example, case_sync_constant_channel,
        case_sync_nonlinear_clock, case_sync_pairwise_only_overlap,
        case_blend_noise_and_ties, case_blend_infeasible_and_limits,
        case_sync_offline_exchange, case_sync_input_validation,
        case_sync_requested_range,
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for i, fn in enumerate(cases, start=1):
            # 每个案例独立 SQLite：隔离规模上限（如 50 次分析）等状态
            fn(os.path.join(tmp, f"case{i}.sqlite"))
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} 项断言通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
