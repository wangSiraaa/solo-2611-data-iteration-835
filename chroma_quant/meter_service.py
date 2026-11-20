"""合成表计服务：同步统计产品的计算、持久化、离线交换与受限合成。

职责边界
--------
* 只读使用信号仓的公开分段产品（经 :class:`SignalService` 的复读校验
  加载，源版本复查不变）；
* 同步统计一旦计算即以内容指纹为键不可变持久化，可导出 JSON 离线交换；
* 受限合成只读取已存的统计 JSON——**更换合成表头绝不重新读取折线**；
* 重新选取通道 = 新的 ``compute_sync`` 调用，旧统计产品永远保留可复读。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Sequence

from .blend import (BlendPlan, MeterHeader, header_from_dict, header_to_dict,
                    optimize_blend, plan_to_dict)
from .errors import NotFoundError
from .signal_service import SignalService
from .store import connect, init_db
from .sync import (SyncStats, compute_sync_stats, make_channel_spec,
                   make_clock, sync_stats_from_dict, sync_stats_to_dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MeterService:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: sqlite3.Connection = connect(db_path)
        init_db(self.conn)

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # 同步统计：计算 + 不可变持久化
    # ------------------------------------------------------------------
    def compute_sync(self, selection: Sequence[dict], a, b) -> SyncStats:
        """按通道选取计算同步统计产品并持久化（同内容命中已有记录）。

        selection 元素：``{"name", "product_id",
        "windows": [[lo, hi], ...], "clock": [[t, tau], ...]}``，
        其中有效窗与 clock 锚点的第一坐标都在通道时钟下。
        """
        svc = SignalService(self.db_path)
        specs = []
        try:
            for sel in selection:
                # get_product 带复读校验：用原采样与锚点重算净响应比对
                product = svc.get_product(sel["product_id"])
                specs.append(make_channel_spec(
                    sel["name"], product, sel["windows"],
                    make_clock(sel["clock"])))
        finally:
            svc.close()
        stats = compute_sync_stats(specs, a, b)
        self.conn.execute(
            "INSERT OR IGNORE INTO sync_products"
            "(sync_id, created_at, fingerprint, payload) VALUES (?,?,?,?)",
            (stats.sync_id, _now(), stats.fingerprint,
             json.dumps(sync_stats_to_dict(stats), ensure_ascii=False)))
        self.conn.commit()
        return stats

    def get_sync(self, sync_id: str) -> SyncStats:
        """按 id 取回统计产品（重组时复核指纹与派生量）。"""
        row = self.conn.execute(
            "SELECT payload FROM sync_products WHERE sync_id=?",
            (sync_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"未知同步统计产品: {sync_id}")
        stats = sync_stats_from_dict(json.loads(row["payload"]))
        if stats.sync_id != sync_id:
            raise NotFoundError(f"同步统计产品 id 不符: {sync_id}")
        return stats

    def list_syncs(self) -> list[str]:
        return [r["sync_id"] for r in self.conn.execute(
            "SELECT sync_id FROM sync_products ORDER BY created_at")]

    def export_sync_json(self, sync_id: str) -> str:
        return json.dumps(sync_stats_to_dict(self.get_sync(sync_id)),
                          ensure_ascii=False, indent=2)

    def import_sync(self, payload: dict) -> str:
        """导入外部同步统计产品（复核指纹后落库，可离线交换）。"""
        stats = sync_stats_from_dict(payload)
        self.conn.execute(
            "INSERT OR IGNORE INTO sync_products"
            "(sync_id, created_at, fingerprint, payload) VALUES (?,?,?,?)",
            (stats.sync_id, _now(), stats.fingerprint,
             json.dumps(sync_stats_to_dict(stats), ensure_ascii=False)))
        self.conn.commit()
        return stats.sync_id

    # ------------------------------------------------------------------
    # 受限合成：只读统计 JSON，绝不重读折线
    # ------------------------------------------------------------------
    def optimize(self, sync_id: str, header: MeterHeader | dict) -> BlendPlan:
        if isinstance(header, dict):
            header = header_from_dict(header)
        stats = self.get_sync(sync_id)  # 仅访问 sync_products 表
        plan = optimize_blend(stats, header)
        self.conn.execute(
            "INSERT OR IGNORE INTO blend_plans"
            "(plan_id, sync_id, header_json, created_at, payload)"
            " VALUES (?,?,?,?,?)",
            (plan.plan_id, sync_id,
             json.dumps(header_to_dict(header), ensure_ascii=False),
             _now(),
             json.dumps(plan_to_dict(plan), ensure_ascii=False)))
        self.conn.commit()
        return plan

    def get_plan(self, plan_id: str) -> dict:
        row = self.conn.execute(
            "SELECT payload FROM blend_plans WHERE plan_id=?",
            (plan_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"未知合成方案: {plan_id}")
        return json.loads(row["payload"])

    def list_plans(self, sync_id: str | None = None) -> list[str]:
        if sync_id is None:
            rows = self.conn.execute(
                "SELECT plan_id FROM blend_plans ORDER BY created_at"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT plan_id FROM blend_plans WHERE sync_id=?"
                " ORDER BY created_at", (sync_id,)).fetchall()
        return [r["plan_id"] for r in rows]
