# 色谱基线产品与重叠峰组分定量仓

纯 Python 3.11 标准库实现（SQLite + `fractions.Fraction`，无第三方依赖），
只处理本地合成的有理数时间序列，**不涉及真实仪器或化学物质识别**。

## 解决的问题

同一条色谱中两个组分的峰会重叠。若对每个组分的归属区间分别积分，共同部分
会被重复取走，使总量超过真实观测。本系统把「归属」建模为**同一时刻的份额
归一**：W(t)=Σ 权重，份额 = 权重/W，重叠处按份额瓜分同一份净响应，从机制上
杜绝重复计数。

## 核心设计

### 1. 全程有理、分段线性（`segments.py`）

* 采样点 `(t,y)`、基线锚点、权重节点全部以 `Fraction` 精确存储；相邻点线性插值。
* 基线由 ≥2 个位于采样范围内、时间互异的锚点给出，锚点覆盖的闭范围即有效域；
  域外查询/锚点一律拒绝，**不外推**。
* 净响应 `max(signal-baseline, 0)`：在信号与基线折点并集上逐段线性求根，
  交点被**真实插入为折点**（不是采样点截零后连线），零段显式保留。
* `PLF.canonical()` 去除共线冗余折点；`fingerprint()` 对规范化后的函数内容
  取哈希——同一条函数无论用多少分段表示，指纹相同。

### 2. 定量引擎（`quant.py`）

* 每组分：固定正有理响应因子 + 一组非负权重节点；节点间线性、范围外恒 0。
* 事件网格 = 净响应折点 ∪ 权重节点 ∪ 各组分支撑边界（裁剪到有效域），
  因此**不依赖固定采样步长恰好命中窄峰**。
* 每个事件格内再求权重和 W 的内部零点并切成子格，统一处理：
  支撑边界的半开间断、内部零权重间隙、W 从零变正/归零。
* W=0 子格的净响应计入「未分配」；W>0 子格按份额积分。

### 3. 积分策略（`integration.py`）

份额相除后 `u·n/W` 是「二次多项式 / 一次多项式」，非多项式。主路径用闭式
解析解：多项式除法分解为 `αs + β + γ/W(s)`，积分含一个对数项，端点 W=0 时
`W·ln W → 0` 取极限。两条**代码完全独立**的核对路径：

* `adaptive_simpson_float`：float 递归自适应 Simpson；
* `composite_simpson_fraction`：纯 `Fraction` 复合 Simpson，全程无浮点。

总面积、未分配面积用精确有理梯形；守恒以精确值核对，残差超 1e-8 直接报错，
**禁止反扣任何组分凑守恒**。

### 4. 持久化与不可变历史（`store.py` / `signal_service.py` / `repository.py`）

* 分数以整数分子/分母存入 SQLite。
* 基线「更正」只追加新版本；产品与分析结果不可变，旧方法/旧产品/旧结果
  永远可复读。
* 分析缓存键 = 方法 × 净响应函数指纹：不同输入分段但同函数命中同一缓存。
* 信号服务只做基线扣除与区间面积，**不接受组分方法**；定量仓只消费公开分段
  产品，也能分析 `product_from_dict` 读入的外部等价产品（导入时复算校验）。
* 组分名称只是身份：结果按方法定义顺序返回，不做字典序重排。

### 5. 多路净响应的同步统计（`sync.py`）

为合成表计准备可独立交换的同步统计量。每一路通道 = 已有净响应产品
（通道时钟）+ 若干有效时间窗（通道时钟）+ 严格递增、分段线性的时钟
锚点映射（通道时钟 → 公共时间），锚点覆盖之外**不外推**。

* 公共有效段 = 请求的公共时间范围 ∩ 各路有效窗（映到公共时间）的交集；
  **单点交集不提供测长**，直接丢弃。
* 在共同有效段上按公共时间长度精确累计 `∫r_i` 与 `∫r_i·r_j`（通道两两
  读数乘积的积分）：均值 = `∫r_i / L`，总体协方差 =
  `∫r_i·r_j / L − 均值之积`。事件格（各路折点并集）内乘积是二次多项式，
  Simpson 公式精确，全程 `Fraction`。
* **不允许**对每一对通道分别取交集再拼矩阵——那会产生实际不存在的
  记录批。整组通道没有正长度共同测度时抛 `DomainError`，由调用方
  重新选取通道。
* 统计产品列出共同有效段、各通道版本（run / 基线版本 / 净响应指纹）、
  时钟锚点、有效窗与全部矩；以内容指纹为 id，可导出 JSON 离线交换，
  重组时复核指纹与派生量（均值、协方差必须能由矩复算）。

### 6. 受限读数合成（`blend.py` / `meter_service.py`）

合成表头独立于统计产品：固定通道次序下的整数份额（总份数 D ≤ 16）、
每通道份数上下限、参与通道数上限、每通道必须追加的独立读出噪声方差。
各通道独立噪声互不相关，只把方差叠加到同步协方差的对角线：
`Σ' = Σ + diag(σ²)`。以非负份数 / D 为权重，穷举全部合法整数配置
（≤6 路、D≤16），求波动方差 `wᵀΣ'w` 最小者；并列按固定通道次序的
份额向量取字典序最小。输出最优份额、预期均值、各对角/交叉/噪声项贡献
（求和即总方差）；不可配时给出明确说明。

`MeterService` 只读使用信号仓产品（源版本复读校验不变），统计产品与
合成方案不可变落库；**更换表头只读取已存统计 JSON，绝不重新读取
折线**；重新选取通道 = 新的 `compute_sync`，旧统计产品永远保留可复读。

## 规模上限（`limits.py`）

每条信号 500 采样点、20 基线锚点；每方法 10 组分、每组分 30 权重节点；
50 次分析（缓存命中不计入）；同步统计一次至多 6 路通道，合成表头
总份数 D 至多 16。

## 入口

```bash
# 完整自动化验证（135 项断言，含手算例与独立积分核对）
python -m chroma_quant.validation

# 可复跑端到端演示
python -m chroma_quant.demo

# CLI（在仓库根目录执行，或设置 PYTHONPATH）
python -m chroma_quant.cli validate
python -m chroma_quant.cli init --db demo.sqlite
python -m chroma_quant.cli import-run --db demo.sqlite --run R --samples '0:1 1:5 2:2'
python -m chroma_quant.cli add-baseline --db demo.sqlite --run R --anchors '0:3 2:1'
python -m chroma_quant.cli publish --db demo.sqlite --run R
python -m chroma_quant.cli query  --db demo.sqlite --run R --a 0 --b 2
python -m chroma_quant.cli area   --db demo.sqlite --run R --a 0 --b 2
python -m chroma_quant.cli add-method --db demo.sqlite --method M \
    --components 'alpha:2:(0,1;2,0) beta:5:(1,1;2,2)'
python -m chroma_quant.cli analyze --db demo.sqlite --method M --product <PID>
python -m chroma_quant.cli explain --db demo.sqlite --method M --product <PID>
python -m chroma_quant.cli export-product --db demo.sqlite --product <PID> > p.json
python -m chroma_quant.cli import-product --db other.sqlite --file p.json

# 多路同步统计与受限合成
python -m chroma_quant.cli sync --db demo.sqlite --selection sel.json
#   sel.json: {"range": ["0/1","2/1"], "channels": [{"name": "A",
#     "product_id": "<PID>", "windows": [["0/1","2/1"]],
#     "clock": [["0/1","0/1"],["2/1","2/1"]]}, ...]}
python -m chroma_quant.cli syncs --db demo.sqlite
python -m chroma_quant.cli sync-show --db demo.sqlite --sync <SYNC_ID> > s.json
python -m chroma_quant.cli import-sync --db other.sqlite --file s.json
python -m chroma_quant.cli blend --db demo.sqlite --sync <SYNC_ID> --header h.json
#   h.json: {"header_id": "H", "channels": ["A","B"], "total_shares": 4,
#     "min_shares": [0,0], "max_shares": [4,4], "max_participants": 2,
#     "noise_variances": ["0/1","0/1"]}   # 后四项可缺省
python -m chroma_quant.cli blend-offline --sync-file s.json --header h.json
python -m chroma_quant.cli plans --db demo.sqlite --sync <SYNC_ID>
python -m chroma_quant.cli plan-show --db demo.sqlite --plan <PLAN_ID>
```

## 验证案例（`validation.py`）

| 案例 | 覆盖点 |
|---|---|
| C1 | 可手算常数份额（纯有理、零残差）、名称身份顺序 |
| C2 | 基线跨零、交点不落采样点的真实切段、子采样点插值 |
| C3 | 交点恰好落在采样点 |
| C4 | 宽度 1/3 的窄归属窗，不依赖步长命中；两条独立积分器核对 |
| C5 | 重叠峰不重复取走、零权重间隙→未分配、精确守恒 |
| C6 | 不同输入分段同函数→同指纹、导入去重、缓存命中、坏产品拒绝 |
| C7 | 更正基线锚点后结果改变；旧方法/产品/结果仍可复读 |
| C8 | 有效域外拒绝外推、超范围/单锚点拒绝 |
| C9 | 500/20/10/30/50 全部规模上限 |
| C10 | 严格递增、权重非负、正因子、名称唯一等输入校验 |
| C11 | 净响应与权重折点交错的通用案例，三条积分路径两两一致 |
| S1 | 题面手算例：A=t、B=2−t、D=4 → 2:2 方差 0；A≥3 份 → 3:1 方差 1/12；指纹核查与篡改拒绝 |
| S2 | 固定原表头、B 改恒定 1 → 全部选 B（方差 0） |
| S3 | 非线性时钟段（斜率变化的锚点映射）、窗经时钟映射、覆盖外不外推、独立数值积分复核 |
| S4 | 三路仅两两有交叠→整组拒绝、重新选路可行；单点交集不提供测长 |
| S5 | 独立噪声叠加对角迫使分散、参与通道数上限、并列取字典序最小 |
| S6 | 下限和/上限和/参与数导致的不可配说明；D≤16、≤6 路上限 |
| S7 | 统计产品离线交换（无折线的新库复算一致）、源版本复查、不可变历史、多表头同统计 |
| S8 | 时钟单调性、有效窗、重名、表头通道次序、噪声非负等输入校验 |
| S9 | 请求范围裁剪、窗外拒绝、相接窗合并与多段测长 |
