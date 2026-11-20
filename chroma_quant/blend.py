"""受限读数合成：合成表头 + 整数份额下的最小波动配比。

合成表头（MeterHeader）独立于同步统计产品存在，定义：

* 固定通道次序下的整数份额，总份数 D（≤16）；
* 每通道份数下限、上限；
* 参与通道数上限（份额为正的通道数）；
* 每通道必须追加的独立读出噪声方差。

各通道独立噪声互不相关，其方差只叠加到同步协方差的对角线：
``Σ' = Σ + diag(σ²)``。权重 = 非负份数 / D，目标是最小化合成波动
方差 ``wᵀΣ'w``。并列时按固定通道次序的份额向量取字典序最小。

本模块是纯函数：只消费 :class:`~chroma_quant.sync.SyncStats`
（均值 + 协方差），**绝不读取折线**——更换表头不需要重新读取
任何产品分段。份额空间有界（≤6 路、D≤16），直接穷举所有合法
整数配置，结果精确（Fraction）、可复读。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from . import limits
from .errors import LimitError, ValidationError
from .rational import frac_str, stable_id, to_fraction
from .sync import SyncStats

HEADER_KIND = "chroma_quant.meter_header"
HEADER_SCHEMA = 1
PLAN_KIND = "chroma_quant.blend_plan"
PLAN_SCHEMA = 1


# ----------------------------------------------------------------------
# 合成表头
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MeterHeader:
    """合成表头：整数份额约束 + 独立读出噪声（与统计产品解耦）。"""

    header_id: str
    channels: tuple[str, ...]             # 固定通道次序
    total_shares: int                     # D
    min_shares: tuple[int, ...]
    max_shares: tuple[int, ...]
    max_participants: int                 # 份额为正的通道数上限
    noise_variances: tuple[Fraction, ...]  # 追加到对角的独立噪声方差

    def __post_init__(self) -> None:
        n = len(self.channels)
        if not self.header_id:
            raise ValidationError("合成表头 header_id 不能为空")
        if n == 0:
            raise ValidationError("合成表头至少包含一路通道")
        if n > limits.MAX_SYNC_CHANNELS:
            raise LimitError(
                f"合成表头最多 {limits.MAX_SYNC_CHANNELS} 路通道，收到 {n}")
        if len(set(self.channels)) != n or not all(self.channels):
            raise ValidationError("合成表头通道名称必须非空且唯一")
        if not isinstance(self.total_shares, int) \
                or isinstance(self.total_shares, bool):
            raise ValidationError("总份数 D 必须是整数")
        if self.total_shares < 1:
            raise ValidationError("总份数 D 必须为正整数")
        if self.total_shares > limits.MAX_TOTAL_SHARES:
            raise LimitError(
                f"总份数 D 至多 {limits.MAX_TOTAL_SHARES}，"
                f"收到 {self.total_shares}")
        for label, seq in (("min_shares", self.min_shares),
                           ("max_shares", self.max_shares),
                           ("noise_variances", self.noise_variances)):
            if len(seq) != n:
                raise ValidationError(f"{label} 长度必须等于通道数 {n}")
        for i in range(n):
            lo, hi = self.min_shares[i], self.max_shares[i]
            if not isinstance(lo, int) or not isinstance(hi, int) \
                    or isinstance(lo, bool) or isinstance(hi, bool):
                raise ValidationError("份数上下限必须是整数")
            if lo < 0:
                raise ValidationError("份数下限必须非负")
            if lo > hi:
                raise ValidationError(f"通道 {self.channels[i]} 份数下限超过上限")
            if hi > self.total_shares:
                raise ValidationError(
                    f"通道 {self.channels[i]} 份数上限超过总份数 D")
            if self.noise_variances[i] < 0:
                raise ValidationError("独立噪声方差必须非负")
        if not 1 <= self.max_participants <= n:
            raise ValidationError("参与通道数上限必须在 1 与通道数之间")


def make_header(header_id: str, channels: Sequence[str], total_shares: int,
                *, min_shares: Sequence[int] | None = None,
                max_shares: Sequence[int] | None = None,
                max_participants: int | None = None,
                noise_variances: Sequence | None = None) -> MeterHeader:
    """构造表头；缺省：下限 0、上限 D、参与数不限、无附加噪声。"""
    n = len(channels)
    return MeterHeader(
        header_id=header_id,
        channels=tuple(channels),
        total_shares=int(total_shares),
        min_shares=tuple(min_shares) if min_shares is not None else (0,) * n,
        max_shares=tuple(max_shares) if max_shares is not None
        else (int(total_shares),) * n,
        max_participants=(n if max_participants is None
                          else int(max_participants)),
        noise_variances=tuple(to_fraction(x, "噪声方差")
                              for x in noise_variances)
        if noise_variances is not None else (Fraction(0),) * n,
    )


def header_to_dict(h: MeterHeader) -> dict:
    return {
        "kind": HEADER_KIND,
        "schema": HEADER_SCHEMA,
        "header_id": h.header_id,
        "channels": list(h.channels),
        "total_shares": h.total_shares,
        "min_shares": list(h.min_shares),
        "max_shares": list(h.max_shares),
        "max_participants": h.max_participants,
        "noise_variances": [frac_str(x) for x in h.noise_variances],
    }


def header_from_dict(d: dict) -> MeterHeader:
    if d.get("kind") not in (None, HEADER_KIND):
        raise ValidationError(f"不是合成表头: kind={d.get('kind')!r}")
    return make_header(
        d["header_id"], d["channels"], int(d["total_shares"]),
        min_shares=d.get("min_shares"),
        max_shares=d.get("max_shares"),
        max_participants=d.get("max_participants"),
        noise_variances=d.get("noise_variances"),
    )


# ----------------------------------------------------------------------
# 合成方案
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BlendTerm:
    """方差的一个加项：对角项、交叉项或独立噪声项（求和即总方差）。"""

    label: str
    value: Fraction


@dataclass(frozen=True)
class BlendPlan:
    """一次受限合成的结果（不可变、可离线核查）。"""

    plan_id: str
    sync_id: str
    header: MeterHeader
    feasible: bool
    reason: str | None                # 不可配时的说明
    shares: tuple[int, ...] | None    # 最优整数份额
    weights: tuple[Fraction, ...] | None
    expected_mean: Fraction | None    # 预期均值 w·μ
    variance: Fraction | None         # 最小波动方差 wᵀΣ'w
    terms: tuple[BlendTerm, ...]      # 各对角/交叉/噪声项贡献
    configurations: int               # 枚举过的合法配置数（审计）


def _plan_id(sync_id: str, header: MeterHeader) -> str:
    canonical = json.dumps(header_to_dict(header), sort_keys=True,
                           ensure_ascii=False)
    return "PLAN-" + stable_id("blend_plan", sync_id, canonical)[:24]


def _enumerate_shares(header: MeterHeader):
    """递归穷举满足全部约束的整数份额向量（带可行性剪枝）。"""
    n = len(header.channels)
    D = header.total_shares
    lo, hi = header.min_shares, header.max_shares
    rest_lo = [0] * (n + 1)
    rest_hi = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        rest_lo[i] = rest_lo[i + 1] + lo[i]
        rest_hi[i] = rest_hi[i + 1] + hi[i]
    shares = [0] * n

    def rec(i: int, remaining: int, participants_left: int):
        if i == n:
            if remaining == 0:
                yield tuple(shares)
            return
        for s in range(lo[i], hi[i] + 1):
            if s > remaining:
                break
            rem = remaining - s
            if rem < rest_lo[i + 1]:
                break  # s 更大只会更不够
            if rem > rest_hi[i + 1]:
                continue
            used = 1 if s > 0 else 0
            if used > participants_left:
                continue
            if rem > 0 and participants_left - used == 0:
                continue  # 剩余份数无处安放
            shares[i] = s
            yield from rec(i + 1, rem, participants_left - used)
    yield from rec(0, D, header.max_participants)


def _infeasible_reason(header: MeterHeader) -> str:
    n = len(header.channels)
    D = header.total_shares
    reasons: list[str] = []
    sum_lo, sum_hi = sum(header.min_shares), sum(header.max_shares)
    if sum_lo > D:
        reasons.append(f"份数下限之和 {sum_lo} 超过总份数 D={D}")
    if sum_hi < D:
        reasons.append(f"份数上限之和 {sum_hi} 小于总份数 D={D}")
    forced = sum(1 for x in header.min_shares if x > 0)
    if forced > header.max_participants:
        reasons.append(
            f"下限为正的通道数 {forced} 超过参与通道数上限 "
            f"{header.max_participants}")
    topk = sum(sorted(header.max_shares, reverse=True)
               [:header.max_participants])
    if topk < D:
        reasons.append(
            f"参与通道数上限 {header.max_participants} 下，可用份数上限之和 "
            f"{topk} 小于总份数 D={D}")
    if not reasons:
        reasons.append(
            "在份数上下限与参与通道数上限下，不存在和为 D 的整数份额配置")
    return "；".join(reasons)


def optimize_blend(stats: SyncStats, header: MeterHeader) -> BlendPlan:
    """在表头约束下求波动方差最小的合法配置（纯函数，不读折线）。

    目标 ``wᵀ(Σ+diag(σ²))w``，权重 = 份数/D；并列取固定通道次序下
    份额向量字典序最小者。不可配时返回 ``feasible=False`` 与说明。
    """
    names = stats.channel_names
    if tuple(header.channels) != names:
        raise ValidationError(
            f"合成表头通道次序 {header.channels} 与同步统计产品 "
            f"{names} 不一致")
    n = len(names)
    D = header.total_shares
    # 独立噪声互不相关：方差只叠加到对角
    cov = [[stats.covariance[i][j]
            + (header.noise_variances[i] if i == j else Fraction(0))
            for j in range(n)] for i in range(n)]

    best_var: Fraction | None = None
    best_shares: tuple[int, ...] | None = None
    count = 0
    for shares in _enumerate_shares(header):
        count += 1
        w = [Fraction(s, D) for s in shares]
        var = sum(w[i] * w[j] * cov[i][j]
                  for i in range(n) for j in range(n))
        if best_var is None or var < best_var \
                or (var == best_var and shares < best_shares):
            best_var, best_shares = var, shares

    plan_id = _plan_id(stats.sync_id, header)
    if best_shares is None:
        return BlendPlan(
            plan_id=plan_id, sync_id=stats.sync_id, header=header,
            feasible=False, reason=_infeasible_reason(header),
            shares=None, weights=None, expected_mean=None, variance=None,
            terms=(), configurations=0)

    weights = tuple(Fraction(s, D) for s in best_shares)
    mean = sum(weights[i] * stats.means[i] for i in range(n))
    terms: list[BlendTerm] = []
    for i in range(n):
        terms.append(BlendTerm(
            f"{names[i]}*{names[i]}",
            weights[i] * weights[i] * stats.covariance[i][i]))
    for i in range(n):
        for j in range(i + 1, n):
            terms.append(BlendTerm(
                f"{names[i]}*{names[j]}",
                2 * weights[i] * weights[j] * stats.covariance[i][j]))
    for i in range(n):
        terms.append(BlendTerm(
            f"noise[{names[i]}]",
            weights[i] * weights[i] * header.noise_variances[i]))
    total = sum((t.value for t in terms), Fraction(0))
    if total != best_var:  # 防御：分解必须精确还原目标值
        raise ValidationError("方差分解与目标值不一致")
    return BlendPlan(
        plan_id=plan_id, sync_id=stats.sync_id, header=header,
        feasible=True, reason=None, shares=best_shares, weights=weights,
        expected_mean=mean, variance=best_var, terms=tuple(terms),
        configurations=count)


def plan_to_dict(p: BlendPlan) -> dict:
    return {
        "kind": PLAN_KIND,
        "schema": PLAN_SCHEMA,
        "plan_id": p.plan_id,
        "sync_id": p.sync_id,
        "header": header_to_dict(p.header),
        "feasible": p.feasible,
        "reason": p.reason,
        "shares": list(p.shares) if p.shares is not None else None,
        "weights": [frac_str(w) for w in p.weights]
        if p.weights is not None else None,
        "expected_mean": frac_str(p.expected_mean)
        if p.expected_mean is not None else None,
        "variance": frac_str(p.variance) if p.variance is not None else None,
        "terms": [{"label": t.label, "value": frac_str(t.value)}
                  for t in p.terms],
        "configurations_enumerated": p.configurations,
    }
