"""命令行入口。

示例
----
::

  python -m chroma_quant.cli init --db demo.sqlite
  python -m chroma_quant.cli import-run --db demo.sqlite --run R \
      --samples '0:1 1:5 2:2 3:6'
  python -m chroma_quant.cli add-baseline --db demo.sqlite --run R \
      --anchors '0:3 3:1'
  python -m chroma_quant.cli publish --db demo.sqlite --run R
  python -m chroma_quant.cli query --db demo.sqlite --run R --a 0 --b 3
  python -m chroma_quant.cli area --db demo.sqlite --run R --a 0 --b 3
  python -m chroma_quant.cli add-method --db demo.sqlite --method M \
      --components 'A:2:(0,1;3,1) B:5:(2,1;3,2)'
  python -m chroma_quant.cli analyze --db demo.sqlite --method M --product PID
  python -m chroma_quant.cli explain --db demo.sqlite --method M --product PID
  python -m chroma_quant.cli sync --db demo.sqlite --selection sel.json
  python -m chroma_quant.cli blend --db demo.sqlite --sync SYNC_ID --header h.json
  python -m chroma_quant.cli blend-offline --sync-file s.json --header h.json
  python -m chroma_quant.cli validate
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction

from .blend import header_from_dict, optimize_blend, plan_to_dict
from .errors import ChromaError
from .meter_service import MeterService
from .rational import frac_str
from .repository import QuantRepository
from .signal_service import SignalService
from .sync import sync_stats_from_dict, sync_stats_to_dict
from . import validation as validation_mod


def _parse_pairs(text: str) -> list[tuple[Fraction, Fraction]]:
    """解析 "t:y t:y" 形式的有理点序列。"""
    out = []
    for tok in text.split():
        if not tok:
            continue
        t, y = tok.split(":", 1)
        out.append((Fraction(t), Fraction(y)))
    return out


def _parse_components(text: str) -> list[dict]:
    """解析 "名称:因子:(t,w;t,w) 名称:因子:(...)"。"""
    comps = []
    for chunk in text.split():
        name, factor, nodes_raw = chunk.split(":", 2)
        inner = nodes_raw.strip()
        if inner.startswith("(") and inner.endswith(")"):
            inner = inner[1:-1]
        nodes = []
        for pair in inner.split(";"):
            if pair:
                t, w = pair.split(",")
                nodes.append((Fraction(t), Fraction(w)))
        comps.append({"name": name, "factor": Fraction(factor),
                      "nodes": nodes})
    return comps


def _plf_to_list(plf) -> list[list[str]]:
    return [[frac_str(k.t), frac_str(k.y)] for k in plf.knots]


def cmd_init(args) -> None:
    svc = SignalService(args.db)
    svc.close()
    print(f"已初始化数据库 {args.db}")


def cmd_import_run(args) -> None:
    svc = SignalService(args.db)
    svc.create_run(args.run, _parse_pairs(args.samples), note=args.note)
    print(f"已导入 run={args.run}，{len(args.samples.split())} 个采样点")
    svc.close()


def cmd_add_baseline(args) -> None:
    svc = SignalService(args.db)
    if args.correct:
        v = svc.correct_baseline(args.run, _parse_pairs(args.anchors))
        print(f"已更正基线，新版本号 v{v}（旧版本保留）")
    else:
        v = svc.add_baseline(args.run, _parse_pairs(args.anchors))
        print(f"已写入基线版本 v{v}")
    svc.close()


def cmd_versions(args) -> None:
    svc = SignalService(args.db)
    print(json.dumps(svc.list_baseline_versions(args.run)))
    svc.close()


def cmd_publish(args) -> None:
    svc = SignalService(args.db)
    product = svc.publish_product(args.run, args.version)
    print(product.product_id)
    svc.close()


def cmd_query(args) -> None:
    svc = SignalService(args.db)
    q = svc.query(args.run, Fraction(args.a), Fraction(args.b),
                  args.version)
    payload = {
        "run_id": q["run_id"],
        "baseline_version": q["version"],
        "interval": [str(q["interval"][0]), str(q["interval"][1])],
        "coverage": [str(q["coverage"][0]), str(q["coverage"][1])],
        "raw_signal": _plf_to_list(q["raw_signal"]),
        "baseline": _plf_to_list(q["baseline"]),
        "net_response": _plf_to_list(q["net_response"]),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    svc.close()


def cmd_area(args) -> None:
    svc = SignalService(args.db)
    area = svc.interval_area(args.run, Fraction(args.a),
                             Fraction(args.b), args.version)
    print(frac_str(area), f"≈ {float(area):.12g}")
    svc.close()


def cmd_add_method(args) -> None:
    repo = QuantRepository(args.db)
    repo.create_method(args.method, _parse_components(args.components),
                       note=args.note)
    print(f"已创建方法 {args.method}")
    repo.close()


def cmd_import_product(args) -> None:
    repo = QuantRepository(args.db)
    with open(args.file, encoding="utf-8") as fh:
        pid = repo.import_product(json.load(fh))
    print(pid)
    repo.close()


def cmd_export_product(args) -> None:
    repo = QuantRepository(args.db)
    print(repo.export_product_json(args.product))
    repo.close()


def cmd_analyze(args) -> None:
    repo = QuantRepository(args.db)
    res = repo.run_analysis(args.method, args.product,
                            use_cache=not args.no_cache)
    payload = {
        "method_id": res.method_id,
        "product_id": res.product_id,
        "domain": [str(res.domain[0]), str(res.domain[1])],
        "component_area": res.component_area,
        "concentration": res.concentration,
        "unallocated_area": res.unallocated_area,
        "total_area": res.total_area,
        "conservation_residual": res.conservation_residual,
        "tolerance": res.tolerance,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    repo.close()


def cmd_explain(args) -> None:
    repo = QuantRepository(args.db)
    print(json.dumps(repo.explain(args.method, args.product),
                     ensure_ascii=False, indent=2))
    repo.close()


def cmd_validate(args) -> None:
    rc = validation_mod.main()
    raise SystemExit(rc)


# ----------------------------------------------------------------------
# 多路同步统计与受限合成
# ----------------------------------------------------------------------
def _load_json_file(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cmd_sync(args) -> None:
    """selection JSON: {"range": [a, b], "channels": [{"name",
    "product_id", "windows": [[lo, hi], ...],
    "clock": [[通道时间, 公共时间], ...]}, ...]}"""
    ms = MeterService(args.db)
    spec = _load_json_file(args.selection)
    stats = ms.compute_sync(spec["channels"], spec["range"][0],
                            spec["range"][1])
    print(json.dumps(sync_stats_to_dict(stats), ensure_ascii=False,
                     indent=2))
    ms.close()


def cmd_syncs(args) -> None:
    ms = MeterService(args.db)
    print(json.dumps(ms.list_syncs(), ensure_ascii=False, indent=2))
    ms.close()


def cmd_sync_show(args) -> None:
    ms = MeterService(args.db)
    print(ms.export_sync_json(args.sync))
    ms.close()


def cmd_import_sync(args) -> None:
    ms = MeterService(args.db)
    print(ms.import_sync(_load_json_file(args.file)))
    ms.close()


def cmd_blend(args) -> None:
    ms = MeterService(args.db)
    plan = ms.optimize(args.sync, header_from_dict(
        _load_json_file(args.header)))
    print(json.dumps(plan_to_dict(plan), ensure_ascii=False, indent=2))
    ms.close()


def cmd_blend_offline(args) -> None:
    """不接触任何数据库：统计产品与表头都来自 JSON 文件。"""
    stats = sync_stats_from_dict(_load_json_file(args.sync_file))
    plan = optimize_blend(stats, header_from_dict(
        _load_json_file(args.header)))
    print(json.dumps(plan_to_dict(plan), ensure_ascii=False, indent=2))


def cmd_plans(args) -> None:
    ms = MeterService(args.db)
    print(json.dumps(ms.list_plans(args.sync), ensure_ascii=False,
                     indent=2))
    ms.close()


def cmd_plan_show(args) -> None:
    ms = MeterService(args.db)
    print(json.dumps(ms.get_plan(args.plan), ensure_ascii=False, indent=2))
    ms.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chroma_quant",
        description="色谱基线产品与重叠峰组分定量仓")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init"); sp.set_defaults(func=cmd_init)
    sp.add_argument("--db", required=True)

    sp = sub.add_parser("import-run"); sp.set_defaults(func=cmd_import_run)
    sp.add_argument("--db", required=True)
    sp.add_argument("--run", required=True)
    sp.add_argument("--samples", required=True, help='"t:y t:y ..."')
    sp.add_argument("--note")

    sp = sub.add_parser("add-baseline"); sp.set_defaults(func=cmd_add_baseline)
    sp.add_argument("--db", required=True)
    sp.add_argument("--run", required=True)
    sp.add_argument("--anchors", required=True, help='"t:y t:y ..."')
    sp.add_argument("--version", type=int)
    sp.add_argument("--correct", action="store_true",
                    help="更正基线：追加新版本，旧版本保留")

    sp = sub.add_parser("versions"); sp.set_defaults(func=cmd_versions)
    sp.add_argument("--db", required=True); sp.add_argument("--run", required=True)

    sp = sub.add_parser("publish"); sp.set_defaults(func=cmd_publish)
    sp.add_argument("--db", required=True)
    sp.add_argument("--run", required=True)
    sp.add_argument("--version", type=int)

    sp = sub.add_parser("query"); sp.set_defaults(func=cmd_query)
    sp.add_argument("--db", required=True)
    sp.add_argument("--run", required=True)
    sp.add_argument("--a", required=True); sp.add_argument("--b", required=True)
    sp.add_argument("--version", type=int)

    sp = sub.add_parser("area"); sp.set_defaults(func=cmd_area)
    sp.add_argument("--db", required=True)
    sp.add_argument("--run", required=True)
    sp.add_argument("--a", required=True); sp.add_argument("--b", required=True)
    sp.add_argument("--version", type=int)

    sp = sub.add_parser("add-method"); sp.set_defaults(func=cmd_add_method)
    sp.add_argument("--db", required=True)
    sp.add_argument("--method", required=True)
    sp.add_argument("--components", required=True)
    sp.add_argument("--note")

    sp = sub.add_parser("import-product")
    sp.set_defaults(func=cmd_import_product)
    sp.add_argument("--db", required=True)
    sp.add_argument("--file", required=True)

    sp = sub.add_parser("export-product")
    sp.set_defaults(func=cmd_export_product)
    sp.add_argument("--db", required=True)
    sp.add_argument("--product", required=True)

    sp = sub.add_parser("analyze"); sp.set_defaults(func=cmd_analyze)
    sp.add_argument("--db", required=True)
    sp.add_argument("--method", required=True)
    sp.add_argument("--product", required=True)
    sp.add_argument("--no-cache", action="store_true")

    sp = sub.add_parser("explain"); sp.set_defaults(func=cmd_explain)
    sp.add_argument("--db", required=True)
    sp.add_argument("--method", required=True)
    sp.add_argument("--product", required=True)

    sp = sub.add_parser("validate"); sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("sync", help="计算多路同步统计产品")
    sp.set_defaults(func=cmd_sync)
    sp.add_argument("--db", required=True)
    sp.add_argument("--selection", required=True,
                    help='JSON: {"range":[a,b],"channels":[...]}')

    sp = sub.add_parser("syncs", help="列出同步统计产品")
    sp.set_defaults(func=cmd_syncs)
    sp.add_argument("--db", required=True)

    sp = sub.add_parser("sync-show"); sp.set_defaults(func=cmd_sync_show)
    sp.add_argument("--db", required=True)
    sp.add_argument("--sync", required=True)

    sp = sub.add_parser("import-sync"); sp.set_defaults(func=cmd_import_sync)
    sp.add_argument("--db", required=True)
    sp.add_argument("--file", required=True)

    sp = sub.add_parser("blend", help="在表头约束下求最小波动配比")
    sp.set_defaults(func=cmd_blend)
    sp.add_argument("--db", required=True)
    sp.add_argument("--sync", required=True)
    sp.add_argument("--header", required=True, help="合成表头 JSON 文件")

    sp = sub.add_parser("blend-offline",
                        help="不用数据库：统计产品与表头都来自文件")
    sp.set_defaults(func=cmd_blend_offline)
    sp.add_argument("--sync-file", required=True)
    sp.add_argument("--header", required=True)

    sp = sub.add_parser("plans"); sp.set_defaults(func=cmd_plans)
    sp.add_argument("--db", required=True)
    sp.add_argument("--sync")

    sp = sub.add_parser("plan-show"); sp.set_defaults(func=cmd_plan_show)
    sp.add_argument("--db", required=True)
    sp.add_argument("--plan", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ChromaError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
