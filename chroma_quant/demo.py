"""可复跑端到端演示：合成信号 → 基线产品 → 重叠峰定量 → 基线更正 → 复读。

运行：``python -m chroma_quant.demo``
演示库写入临时目录，结束后打印路径；所有输入均为本地合成有理数，
不涉及任何真实仪器或化学物质识别。
"""
from __future__ import annotations

import json
import os
import tempfile
from fractions import Fraction as F

from .quant import product_to_dict
from .repository import QuantRepository
from .signal_service import SignalService


def main() -> None:
    workdir = tempfile.mkdtemp(prefix="chroma_demo_")
    db = os.path.join(workdir, "demo.sqlite")
    print(f"演示数据库: {db}\n")

    svc = SignalService(db)
    # 一条含两个重叠峰的合成信号（整数有理采样，线性插值）
    samples = [(t, _signal_y(t)) for t in range(0, 9)]
    svc.create_run("DEMO", samples, note="本地合成重叠峰信号")

    # 基线 v1：略微上倾
    v1 = svc.add_baseline("DEMO", [(0, 1), (8, 3)])
    p1 = svc.publish_product("DEMO", v1)
    print(f"基线 v{v1} 产品: {p1.product_id[:16]}…")
    print(f"  有效域: {p1.net.domain}")
    print(f"  净响应总面积(精确): {p1.net.integral()} "
          f"≈ {float(p1.net.integral()):.6f}\n")

    repo = QuantRepository(db)
    # 两个重叠组分 + 一个时间很窄的归属窗口，名称顺序即身份（非字典序）
    repo.create_method("OVL", [
        {"name": "zeta", "factor": F(3, 2),
         "nodes": [(0, 2), (3, 2), (5, 0), (8, 0)]},
        {"name": "alpha", "factor": F(7, 3),
         "nodes": [(0, 0), (2, 0), (4, 3), (6, 3), (8, 1)]},
        {"name": "tiny", "factor": 1,
         "nodes": [(F(7, 3), 1), (F(8, 3), 1)]},  # 宽度 1/3 的窄窗
    ])
    r1 = repo.run_analysis("OVL", p1.product_id)
    _print_result("基线 v1 定量结果", r1)

    # 独立交付：信号服务自己的区间净面积（不经过组分方法）
    area = svc.interval_area("DEMO", F(1), F(7), v1)
    print(f"信号服务独立区间面积 [1,7] = {float(area):.6f}\n")

    # 基线更正：锚点整体下移，新版本；旧产品/结果保留
    v2 = svc.correct_baseline("DEMO", [(0, 0), (8, 1)])
    p2 = svc.publish_product("DEMO", v2)
    r2 = repo.run_analysis("OVL", p2.product_id)
    _print_result(f"基线 v{v2} 更正后定量结果", r2)

    # 复读旧结果
    old = repo.find_analysis("OVL", p1.product_id)
    print("旧结果复读: zeta 面积 = "
          f"{old.component_area['zeta']:.6f}（仍为更正前值）")

    # 外部等价分段产品：导出 JSON 到新库，按净响应函数指纹去重复用
    ext_db = os.path.join(workdir, "external.sqlite")
    ext_repo = QuantRepository(ext_db)
    pid = ext_repo.import_product(product_to_dict(p2))
    ext_repo.create_method("OVL", [
        {"name": "zeta", "factor": F(3, 2),
         "nodes": [(0, 2), (3, 2), (5, 0), (8, 0)]},
        {"name": "alpha", "factor": F(7, 3),
         "nodes": [(0, 0), (2, 0), (4, 3), (6, 3), (8, 1)]},
        {"name": "tiny", "factor": 1,
         "nodes": [(F(7, 3), 1), (F(8, 3), 1)]},
    ])
    r_ext = ext_repo.run_analysis("OVL", pid)
    print("\n外部库分析与原库一致: "
          f"{abs(r_ext.total_area - r2.total_area) < 1e-12}")
    svc.close()
    repo.close()
    ext_repo.close()

    meter_demo(workdir)
    print(f"演示文件保留在: {workdir}")


# ----------------------------------------------------------------------
# 多路净响应同步统计与受限读数合成（端到端）
# ----------------------------------------------------------------------
def meter_demo(workdir: str) -> None:
    from .blend import make_header
    from .errors import DomainError
    from .meter_service import MeterService

    print("== 多路同步统计与受限合成 ==")
    db = os.path.join(workdir, "meter.sqlite")
    svc = SignalService(db)
    # A: 净响应 t；B: 净响应 2−t；公共时间 [0,2]，两路无缺口
    svc.create_run("MA", [(0, 0), (1, 1), (2, 2)])
    svc.add_baseline("MA", [(0, 0), (2, 0)])
    pa = svc.publish_product("MA")
    svc.create_run("MB", [(0, 2), (1, 1), (2, 0)])
    svc.add_baseline("MB", [(0, 0), (2, 0)])
    pb = svc.publish_product("MB")
    # C: 净响应 u，但时钟是非线性的：通道 [0,1]→公共 [0,2]、
    #    通道 [1,2]→公共 [2,3]（分段线性锚点映射）
    svc.create_run("MC", [(0, 0), (1, 1), (2, 2)])
    svc.add_baseline("MC", [(0, 0), (2, 0)])
    pc = svc.publish_product("MC")
    # D: 净响应 τ，公共时间 [0,3]
    svc.create_run("MD", [(0, 0), (3, 3)])
    svc.add_baseline("MD", [(0, 0), (3, 0)])
    pd = svc.publish_product("MD")
    # E/F/G: 三路仅两两有交叠（恒定读数 1，有效窗错开）
    wins = {"ME": [[0, 2]], "MF": [[1, 3]], "MG": [["5/2", 4]]}
    prods = {}
    for run in wins:
        svc.create_run(run, [(0, 1), (4, 1)])
        svc.add_baseline(run, [(0, 0), (4, 0)])
        prods[run] = svc.publish_product(run)

    ms = MeterService(db)
    sel_ab = [
        {"name": "A", "product_id": pa.product_id, "windows": [[0, 2]],
         "clock": [[0, 0], [2, 2]]},
        {"name": "B", "product_id": pb.product_id, "windows": [[0, 2]],
         "clock": [[0, 0], [2, 2]]},
    ]
    stats = ms.compute_sync(sel_ab, 0, 2)
    _print_sync("A,B 同步统计（公共时间 0..2）", stats)

    print("-- 表头 H0: D=4，无附加噪声，份额任意 --")
    plan = ms.optimize(stats.sync_id, make_header("H0", ["A", "B"], 4))
    _print_plan(plan)
    print("-- 同一统计产品换表头 H1: A 至少 3 份（不重读折线） --")
    plan1 = ms.optimize(stats.sync_id,
                        make_header("H1", ["A", "B"], 4, min_shares=[3, 0]))
    _print_plan(plan1)

    print("-- 固定 H0，B 的源信号改为恒定 1 --")
    svc.create_run("MB1", [(0, 1), (2, 1)])
    svc.add_baseline("MB1", [(0, 0), (2, 0)])
    pb1 = svc.publish_product("MB1")
    stats_b1 = ms.compute_sync(
        [sel_ab[0], dict(sel_ab[1], product_id=pb1.product_id)], 0, 2)
    _print_plan(ms.optimize(stats_b1.sync_id,
                            make_header("H0", ["A", "B"], 4)))

    print("-- 非线性时钟通道 C 与 D 的同步统计（公共时间 0..3） --")
    stats_cd = ms.compute_sync([
        {"name": "C", "product_id": pc.product_id, "windows": [[0, 2]],
         "clock": [[0, 0], [1, 2], [2, 3]]},
        {"name": "D", "product_id": pd.product_id, "windows": [[0, 3]],
         "clock": [[0, 0], [3, 3]]},
    ], 0, 3)
    _print_sync("C,D 同步统计", stats_cd)

    print("-- 三路仅两两有交叠：整组无共同测度，重新选路 --")
    sel3 = [{"name": r[1:], "product_id": prods[r].product_id,
             "windows": wins[r], "clock": [[0, 0], [4, 4]]}
            for r in ("ME", "MF", "MG")]
    try:
        ms.compute_sync(sel3, 0, 4)
        print("  意外成功？！")
    except DomainError as exc:
        print(f"  三路同选被拒绝: {exc}")
    _print_sync("重选 E,F", ms.compute_sync(sel3[:2], 0, 4))

    print("-- 两路恒定读数 + 独立读出噪声（D=2，σ²=1） --")
    stats_ef = ms.compute_sync(sel3[:2], 0, 4)
    _print_plan(ms.optimize(stats_ef.sync_id, make_header(
        "HN", ["E", "F"], 2, noise_variances=[1, 1])))

    print("-- 不可配表头：两路下限 3+3 > D=4 --")
    bad = ms.optimize(stats.sync_id, make_header(
        "HBAD", ["A", "B"], 4, min_shares=[3, 3]))
    print(f"  feasible={bad.feasible}  原因: {bad.reason}")

    # 离线交换：统计产品导出 JSON，在没有任何折线产品的新库中合成
    exported = ms.export_sync_json(stats.sync_id)
    off_db = os.path.join(workdir, "offline.sqlite")
    ms_off = MeterService(off_db)
    sid = ms_off.import_sync(json.loads(exported))
    plan_off = ms_off.optimize(sid, make_header("H0", ["A", "B"], 4))
    print(f"-- 离线交换: 新库（无折线产品）复算份额 "
          f"{plan_off.shares}，方差 {plan_off.variance}，"
          f"plan_id 一致={plan_off.plan_id == plan.plan_id} --\n")
    ms_off.close()
    ms.close()
    svc.close()


def _print_sync(title: str, stats) -> None:
    print(f"== {title} ==")
    print(f"  sync_id: {stats.sync_id}")
    print(f"  共同有效段: "
          f"{[(str(lo), str(hi)) for lo, hi in stats.common_segments]}"
          f"  测长={stats.total_length}")
    print(f"  各通道版本: "
          f"{[(c.name, c.baseline_version) for c in stats.channels]}")
    print(f"  均值: {[str(m) for m in stats.means]}")
    print(f"  总体协方差: "
          f"{[[str(x) for x in row] for row in stats.covariance]}")


def _print_plan(plan) -> None:
    if not plan.feasible:
        print(f"  不可配: {plan.reason}")
        return
    print(f"  份额 {plan.shares}  权重 {[str(w) for w in plan.weights]}"
          f"  预期均值 {plan.expected_mean}  波动方差 {plan.variance}")
    nonzero = [t for t in plan.terms if t.value != 0]
    if nonzero:
        print("  非零贡献: " + ", ".join(
            f"{t.label}={t.value}" for t in nonzero))


def _signal_y(t: int) -> int:
    # 两个三角峰叠加的本地合成整数序列
    peak1 = max(0, 4 - abs(t - 3))
    peak2 = max(0, 3 - abs(t - 6))
    return 2 + peak1 + peak2


def _print_result(title: str, res) -> None:
    print(f"== {title} ==")
    print(json.dumps({
        "component_area": {k: round(v, 6)
                           for k, v in res.component_area.items()},
        "concentration": {k: round(v, 6)
                          for k, v in res.concentration.items()},
        "unallocated_area": round(res.unallocated_area, 6),
        "total_area": round(res.total_area, 6),
        "conservation_residual": f"{res.conservation_residual:.2e}",
    }, ensure_ascii=False, indent=2))
    assert abs(sum(res.component_area.values())
               + res.unallocated_area - res.total_area) <= res.tolerance
    print()


if __name__ == "__main__":
    main()
