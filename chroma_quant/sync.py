"""多路净响应的同步统计：公共时间轴上的均值与总体协方差。

动机
----
离线记录器的几路并行合成读数采样点错开、各自在不同时段失去有效记录，
且各路时钟到公共时间的映射各不相同。为「合成表计」准备可独立交换的
同步统计量时，必须只在**所有选中通道同时有效**的公共时段上累计矩：

* 每一路 = 净响应产品（通道时钟）+ 若干有效时间窗（通道时钟）
  + 严格递增、分段线性的时钟锚点映射（通道时钟 → 公共时间）。
  锚点覆盖范围之外不外推。
* 公共有效段 = 请求的公共时间范围 ∩ 各路有效窗（映到公共时间）的交集。
  单点交集不提供测长，直接丢弃。
* 在公共有效段上按公共时间长度累计 ``∫r_i`` 与 ``∫r_i·r_j``
  （通道两两读数乘积的积分）：均值 = ``∫r_i / L``，
  总体协方差 = ``∫r_i·r_j / L − 均值之积``。

**不允许**对每一对通道分别取交集再拼成矩阵：不同对使用不同的记录批，
拼出的「同步」协方差在实际中从未同时存在。本模块只接受整组通道的
共同有效段；整组没有正长度共同测度时抛 :class:`DomainError`，
由调用方重新选取通道。

所有矩以 ``Fraction`` 精确累计：在各路折点并集切成的事件格内，
每一路都是线性的，两两乘积是二次多项式，Simpson 公式对其精确。
统计产品可导出 JSON 离线交换，并按内容指纹核查（防篡改、可复读）。
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from . import limits
from .errors import DomainError, LimitError, ValidationError
from .quant import SegmentProduct
from .rational import frac_str, stable_id, to_fraction
from .segments import PLF, Point

SYNC_KIND = "chroma_quant.sync_stats"
SYNC_SCHEMA = 1


# ----------------------------------------------------------------------
# 时钟映射：通道时钟 → 公共时间（严格递增、分段线性、不外推）
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ClockMap:
    """锚点 ``(通道时间, 公共时间)`` 定义的分段线性严格递增映射。"""

    anchors: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.anchors) < 2:
            raise ValidationError("时钟映射至少需要两个锚点")
        for i, p in enumerate(self.anchors):
            if not isinstance(p.t, Fraction) or not isinstance(p.y, Fraction):
                raise ValidationError("时钟锚点必须是 Fraction")
            if i == 0:
                continue
            q = self.anchors[i - 1]
            if p.t <= q.t:
                raise ValidationError("时钟锚点的通道时间必须严格递增")
            if p.y <= q.y:
                raise ValidationError(
                    "时钟锚点的公共时间必须严格递增（映射严格递增）")

    @property
    def t0(self) -> Fraction:
        return self.anchors[0].t

    @property
    def t1(self) -> Fraction:
        return self.anchors[-1].t

    @property
    def tau0(self) -> Fraction:
        return self.anchors[0].y

    @property
    def tau1(self) -> Fraction:
        return self.anchors[-1].y

    def to_common(self, t) -> Fraction:
        """通道时间 → 公共时间；锚点覆盖外拒绝外推。"""
        t = to_fraction(t, "通道时间")
        if t < self.t0 or t > self.t1:
            raise DomainError(
                f"通道时间 {t} 超出时钟锚点覆盖 [{self.t0}, {self.t1}]，"
                "拒绝外推")
        return PLF(self.anchors)(t)

    def to_channel(self, tau) -> Fraction:
        """公共时间 → 通道时间（逆映射）；值域外拒绝外推。"""
        tau = to_fraction(tau, "公共时间")
        if tau < self.tau0 or tau > self.tau1:
            raise DomainError(
                f"公共时间 {tau} 超出时钟映射值域 "
                f"[{self.tau0}, {self.tau1}]，拒绝外推")
        inv = PLF(tuple(Point(p.y, p.t) for p in self.anchors))
        return inv(tau)


def make_clock(anchors: Sequence[tuple]) -> ClockMap:
    """由 ``(通道时间, 公共时间)`` 序列构造时钟映射。"""
    return ClockMap(tuple(
        Point(to_fraction(t, "通道时间"), to_fraction(tau, "公共时间"))
        for t, tau in anchors))


# ----------------------------------------------------------------------
# 通道：净响应产品 + 有效时间窗（通道时钟）+ 时钟映射
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelSpec:
    """一路合成读数的完整定义（有效窗给在通道时钟下）。"""

    name: str
    product: SegmentProduct
    windows: tuple[tuple[Fraction, Fraction], ...]
    clock: ClockMap

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("通道名称不能为空")
        if not self.windows:
            raise ValidationError(f"通道 {self.name} 至少需要一个有效时间窗")
        for lo, hi in self.windows:
            if not isinstance(lo, Fraction) or not isinstance(hi, Fraction):
                raise ValidationError("有效时间窗必须是 Fraction 对")
            if not lo < hi:
                raise ValidationError(
                    f"通道 {self.name} 的有效窗要求起点 < 终点")

    def _channel_domain(self) -> tuple[Fraction, Fraction]:
        """净响应域 ∩ 时钟映射域：之外既无读数也无时钟，不外推。"""
        net = self.product.net
        lo = max(net.t0, self.clock.t0)
        hi = min(net.t1, self.clock.t1)
        if lo >= hi:
            raise DomainError(
                f"通道 {self.name} 的净响应域与时钟映射域"
                "没有正长度交集")
        return lo, hi

    def mapped_net(self) -> PLF:
        """净响应复合逆时钟后、以公共时间为自变量的分段线性函数。

        事件格 = 净响应折点 ∪ 时钟锚点（裁剪到共同通道域），
        格内净响应与时钟同时线性，故复合函数在映过来的格点间线性。
        """
        net = self.product.net
        lo, hi = self._channel_domain()
        grid = {lo, hi}
        for k in net.knots:
            if lo < k.t < hi:
                grid.add(k.t)
        for p in self.clock.anchors:
            if lo < p.t < hi:
                grid.add(p.t)
        pts = [Point(self.clock.to_common(t), net(t)) for t in sorted(grid)]
        return PLF(tuple(pts))

    def common_windows(self) -> tuple[tuple[Fraction, Fraction], ...]:
        """有效窗裁剪到共同通道域后映到公共时间，并合并相邻/重叠段。"""
        lo_dom, hi_dom = self._channel_domain()
        mapped: list[tuple[Fraction, Fraction]] = []
        for a, b in self.windows:
            lo, hi = max(a, lo_dom), min(b, hi_dom)
            if lo < hi:  # 域外部分直接丢弃（未覆盖处不外推）
                mapped.append((self.clock.to_common(lo),
                               self.clock.to_common(hi)))
        return _normalize_windows(mapped)


def make_channel_spec(name: str, product: SegmentProduct,
                      windows: Sequence[tuple], clock: ClockMap
                      ) -> ChannelSpec:
    ws = tuple((to_fraction(lo, "有效窗起点"), to_fraction(hi, "有效窗终点"))
               for lo, hi in windows)
    return ChannelSpec(name=name, product=product, windows=ws, clock=clock)


# ----------------------------------------------------------------------
# 窗口代数（公共时间轴上）
# ----------------------------------------------------------------------
def _normalize_windows(
        wins: Sequence[tuple[Fraction, Fraction]]
) -> tuple[tuple[Fraction, Fraction], ...]:
    """排序并合并重叠或首尾相接的窗口。"""
    out: list[list[Fraction]] = []
    for lo, hi in sorted(wins):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return tuple((lo, hi) for lo, hi in out)


def _intersect_windows(
        w1: Sequence[tuple[Fraction, Fraction]],
        w2: Sequence[tuple[Fraction, Fraction]]
) -> list[tuple[Fraction, Fraction]]:
    """两组窗口的交集；只保留正长度段（单点交集不提供测长）。"""
    out: list[tuple[Fraction, Fraction]] = []
    for a_lo, a_hi in w1:
        for b_lo, b_hi in w2:
            lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
            if lo < hi:
                out.append((lo, hi))
    return out


# ----------------------------------------------------------------------
# 同步统计产品
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelProvenance:
    """一路通道在统计产品中的来源记录（离线核查用）。"""

    name: str
    product_id: str
    run_id: str
    baseline_version: int          # 各通道版本
    net_fingerprint: str
    clock_anchors: tuple[Point, ...]
    windows_channel: tuple[tuple[Fraction, Fraction], ...]  # 原始给定（通道时钟）
    windows_common: tuple[tuple[Fraction, Fraction], ...]   # 裁剪+映射后（公共时间）


@dataclass(frozen=True)
class SyncStats:
    """一组通道在共同有效段上的同步统计量（不可变、可离线交换）。"""

    sync_id: str
    requested_range: tuple[Fraction, Fraction]
    channels: tuple[ChannelProvenance, ...]
    common_segments: tuple[tuple[Fraction, Fraction], ...]
    total_length: Fraction
    first_moments: tuple[Fraction, ...]            # ∫r_i（公共有效段上）
    second_moments: tuple[tuple[Fraction, ...], ...]  # ∫r_i·r_j（对称阵）
    means: tuple[Fraction, ...]
    covariance: tuple[tuple[Fraction, ...], ...]   # 总体协方差（除以 L）
    fingerprint: str

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.channels)


def _content_parts(s: SyncStats) -> list:
    """指纹的规范内容：覆盖请求范围、共同段、各通道来源与全部矩。"""
    parts: list = [
        "range", frac_str(s.requested_range[0]),
        frac_str(s.requested_range[1]),
        "segments", [(frac_str(lo), frac_str(hi))
                     for lo, hi in s.common_segments],
        "L", frac_str(s.total_length),
        "m1", [frac_str(x) for x in s.first_moments],
        "m2", [[frac_str(x) for x in row] for row in s.second_moments],
    ]
    for c in s.channels:
        parts.append([
            "channel", c.name, c.product_id, c.run_id, c.baseline_version,
            c.net_fingerprint,
            [(frac_str(p.t), frac_str(p.y)) for p in c.clock_anchors],
            [(frac_str(lo), frac_str(hi)) for lo, hi in c.windows_channel],
            [(frac_str(lo), frac_str(hi)) for lo, hi in c.windows_common],
        ])
    return parts


def _fingerprint(s: SyncStats) -> str:
    return stable_id("sync_stats", _content_parts(s))


def _moments(mapped: Sequence[PLF],
             segments: Sequence[tuple[Fraction, Fraction]]
             ) -> tuple[list[Fraction], list[list[Fraction]]]:
    """在共同有效段上精确累计 ∫r_i 与 ∫r_i·r_j。

    事件格 = 各路折点在段内的并集；格内每一路线性，乘积为二次
    多项式，Simpson 公式（对三次及以下精确）给出精确有理积分。
    """
    n = len(mapped)
    m1 = [Fraction(0) for _ in range(n)]
    m2 = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for lo, hi in segments:
        grid = {lo, hi}
        for g in mapped:
            for k in g.knots:
                if lo < k.t < hi:
                    grid.add(k.t)
        xs = sorted(grid)
        for c_lo, c_hi in zip(xs, xs[1:]):
            h = c_hi - c_lo
            mid = (c_lo + c_hi) / 2
            v_lo = [g(c_lo) for g in mapped]
            v_mid = [g(mid) for g in mapped]
            v_hi = [g(c_hi) for g in mapped]
            for i in range(n):
                m1[i] += (v_lo[i] + v_hi[i]) * h / 2
                for j in range(i, n):
                    cell = h * (v_lo[i] * v_lo[j]
                                + 4 * v_mid[i] * v_mid[j]
                                + v_hi[i] * v_hi[j]) / 6
                    m2[i][j] += cell
                    if j != i:
                        m2[j][i] += cell
    return m1, m2


def compute_sync_stats(specs: Sequence[ChannelSpec], a, b) -> SyncStats:
    """对整组选中通道，在请求的公共时间范围 [a, b] 内计算同步统计。

    只用所有通道同时有效的时段；整组没有正长度共同测度时抛
    :class:`DomainError`（重新选取通道是调用方的职责）。
    """
    a, b = to_fraction(a, "a"), to_fraction(b, "b")
    if a >= b:
        raise ValidationError("请求的公共时间范围要求 a < b")
    if not specs:
        raise ValidationError("至少选择一路通道")
    if len(specs) > limits.MAX_SYNC_CHANNELS:
        raise LimitError(
            f"一次同步统计最多 {limits.MAX_SYNC_CHANNELS} 路通道，"
            f"收到 {len(specs)}")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValidationError("通道名称必须唯一（名称即身份）")

    mapped = [s.mapped_net() for s in specs]
    common: list[tuple[Fraction, Fraction]] = [(a, b)]
    for s in specs:
        common = _intersect_windows(common, s.common_windows())
    if not common:
        raise DomainError(
            f"所选 {len(specs)} 路通道在请求范围 [{a}, {b}] 内没有正长度的"
            "共同有效时段（单点交集不提供测长；对每一对分别取交集再拼矩阵"
            "会产生实际不存在的记录批，本系统拒绝）。请重新选取通道。")

    total = sum((hi - lo) for lo, hi in common)
    m1, m2 = _moments(mapped, common)
    means = tuple(m / total for m in m1)
    covariance = tuple(
        tuple(m2[i][j] / total - means[i] * means[j] for j in range(len(specs)))
        for i in range(len(specs)))

    channels = tuple(
        ChannelProvenance(
            name=s.name,
            product_id=s.product.product_id,
            run_id=s.product.run_id,
            baseline_version=s.product.baseline_version,
            net_fingerprint=s.product.fingerprint(),
            clock_anchors=s.clock.anchors,
            windows_channel=s.windows,
            windows_common=s.common_windows(),
        )
        for s in specs)

    stats = SyncStats(
        sync_id="",  # 先占位，指纹算好后填
        requested_range=(a, b),
        channels=channels,
        common_segments=tuple(common),
        total_length=total,
        first_moments=tuple(m1),
        second_moments=tuple(tuple(row) for row in m2),
        means=means,
        covariance=covariance,
        fingerprint="",
    )
    fp = _fingerprint(stats)
    return SyncStats(sync_id="SYNC-" + fp[:24],
                     requested_range=stats.requested_range,
                     channels=stats.channels,
                     common_segments=stats.common_segments,
                     total_length=stats.total_length,
                     first_moments=stats.first_moments,
                     second_moments=stats.second_moments,
                     means=stats.means,
                     covariance=stats.covariance,
                     fingerprint=fp)


# ----------------------------------------------------------------------
# 离线交换：JSON 字典进出，重组时复核指纹与派生量
# ----------------------------------------------------------------------
def sync_stats_to_dict(s: SyncStats) -> dict:
    return {
        "kind": SYNC_KIND,
        "schema": SYNC_SCHEMA,
        "sync_id": s.sync_id,
        "requested_range": [frac_str(s.requested_range[0]),
                            frac_str(s.requested_range[1])],
        "channels": [
            {
                "name": c.name,
                "product_id": c.product_id,
                "run_id": c.run_id,
                "baseline_version": c.baseline_version,
                "net_fingerprint": c.net_fingerprint,
                "clock_anchors": [[frac_str(p.t), frac_str(p.y)]
                                  for p in c.clock_anchors],
                "valid_windows_channel": [[frac_str(lo), frac_str(hi)]
                                          for lo, hi in c.windows_channel],
                "valid_windows_common": [[frac_str(lo), frac_str(hi)]
                                         for lo, hi in c.windows_common],
            }
            for c in s.channels
        ],
        "common_segments": [[frac_str(lo), frac_str(hi)]
                            for lo, hi in s.common_segments],
        "total_length": frac_str(s.total_length),
        "first_moments": [frac_str(x) for x in s.first_moments],
        "second_moments": [[frac_str(x) for x in row]
                           for row in s.second_moments],
        "means": [frac_str(x) for x in s.means],
        "covariance": [[frac_str(x) for x in row] for row in s.covariance],
        "fingerprint": s.fingerprint,
    }


def sync_stats_from_dict(d: dict) -> SyncStats:
    """从 JSON 字典重组统计产品，并复核内容指纹与派生量（核查）。"""
    if d.get("kind") != SYNC_KIND:
        raise ValidationError(f"不是同步统计产品: kind={d.get('kind')!r}")
    if int(d.get("schema", 0)) != SYNC_SCHEMA:
        raise ValidationError(f"不支持的同步统计 schema: {d.get('schema')!r}")

    channels = []
    for c in d["channels"]:
        channels.append(ChannelProvenance(
            name=c["name"],
            product_id=c["product_id"],
            run_id=c["run_id"],
            baseline_version=int(c["baseline_version"]),
            net_fingerprint=c["net_fingerprint"],
            clock_anchors=tuple(
                Point(to_fraction(t, "通道时间"), to_fraction(tau, "公共时间"))
                for t, tau in c["clock_anchors"]),
            windows_channel=tuple(
                (to_fraction(lo, "窗起点"), to_fraction(hi, "窗终点"))
                for lo, hi in c["valid_windows_channel"]),
            windows_common=tuple(
                (to_fraction(lo, "窗起点"), to_fraction(hi, "窗终点"))
                for lo, hi in c["valid_windows_common"]),
        ))
    stats = SyncStats(
        sync_id=str(d["sync_id"]),
        requested_range=(to_fraction(d["requested_range"][0], "a"),
                         to_fraction(d["requested_range"][1], "b")),
        channels=tuple(channels),
        common_segments=tuple(
            (to_fraction(lo, "段起点"), to_fraction(hi, "段终点"))
            for lo, hi in d["common_segments"]),
        total_length=to_fraction(d["total_length"], "总测长"),
        first_moments=tuple(to_fraction(x, "一阶矩")
                            for x in d["first_moments"]),
        second_moments=tuple(
            tuple(to_fraction(x, "二阶矩") for x in row)
            for row in d["second_moments"]),
        means=tuple(to_fraction(x, "均值") for x in d["means"]),
        covariance=tuple(
            tuple(to_fraction(x, "协方差") for x in row)
            for row in d["covariance"]),
        fingerprint=str(d["fingerprint"]),
    )
    # 核查 1：内容指纹必须一致（任何字段被改动都会改变指纹）
    if _fingerprint(stats) != stats.fingerprint:
        raise ValidationError("同步统计产品指纹校验失败：内容与指纹不符")
    if stats.sync_id != "SYNC-" + stats.fingerprint[:24]:
        raise ValidationError("同步统计产品 sync_id 与指纹不符")
    # 核查 2：均值/协方差必须能由矩与测长复算（派生量不得篡改）
    n = len(stats.channels)
    if len(stats.means) != n or len(stats.first_moments) != n:
        raise ValidationError("同步统计产品通道数与矩维度不符")
    L = stats.total_length
    if L <= 0:
        raise ValidationError("同步统计产品总测长必须为正")
    for i in range(n):
        if stats.means[i] != stats.first_moments[i] / L:
            raise ValidationError("同步统计产品均值无法由一阶矩复算")
        for j in range(n):
            expect = stats.second_moments[i][j] / L \
                - stats.means[i] * stats.means[j]
            if stats.covariance[i][j] != expect:
                raise ValidationError("同步统计产品协方差无法由矩复算")
            if stats.covariance[i][j] != stats.covariance[j][i]:
                raise ValidationError("同步统计产品协方差阵不对称")
    return stats
