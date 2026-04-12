# Gorge Chase PPO 下一步修改指导方案

## 一、背景与目标

当前你希望参考一套“高分思路”的 curriculum 配置：

### 1. 高分思路中的阶段划分

#### warmup_stable（到 episode 150）
- 宝箱数：9 ~ 10
- buff 数：2
- 第二只怪出现时间：220 ~ 300
- 怪物加速时间：360 ~ 460
- 最大步数：2000

#### mid_pressure（到 episode 500）
- 宝箱数：8 ~ 10
- buff 数：1 ~ 2
- 第二只怪出现时间：160 ~ 280
- 怪物加速时间：240 ~ 420
- 最大步数：2000

#### late_speedup_survival（到 episode 900）
- 宝箱数：7 ~ 10
- buff 数：1 ~ 2
- 第二只怪出现时间：120 ~ 220
- 怪物加速时间：180 ~ 320
- 最大步数：2000

#### hard_generalization
- 宝箱数：6 ~ 10
- buff 数：0 ~ 2
- 第二只怪出现时间：120 ~ 320
- 怪物加速时间：140 ~ 420
- 最大步数：2000

这套设计的目标很明确：

- 前期先让模型在资源更多、压力更低的环境里学会稳定推进
- 后期逐步降低宝箱和 buff 下限
- 提前第二只怪出现时间
- 提前怪物加速时间
- 让模型从“会拿分”过渡到“高压下也能活”

但结合当前代码实现和前面分析出的实际问题，**这套 curriculum 不能直接原样上线**。正确做法是：**先修训练链路，再上更平滑的 curriculum v1，最后再逐步逼近这套高分参数。**

---

## 二、为什么当前不能直接照搬这套高分 curriculum

### 1. 当前训练链路还不够稳

从现有代码看，当前训练系统有几个关键特征：

- **样本按整局收集，episode 结束后才送 learner**
- **每局开始都会 `load_model("latest")`**
- **PPO learner 每次只做 1 次 forward + 1 次 backward + 1 次 optimizer.step()**
- **GAE 代码存在 done 截断缺失问题**
- **critic 当前没有 value / return normalization**

这意味着如果直接把课程学习改得更激进，会把原本链路中的不稳定因素进一步放大。

---

### 2. 高分思路里的 warmup 太短、跳变太猛

这套高分参数存在几个明显风险：

#### （1）warmup 太短
`warmup_stable` 只到 episode 150。

对从零开始训练的 PPO 来说，这通常只够学会一些粗浅行为，比如：

- 少撞墙
- 稍微远离怪物
- 偶尔捡宝箱

但很难真正学稳这些关键策略：

- 安全拿箱
- 怪逼近时主动拉空间
- 给高压阶段提前留安全路线
- 合理保留闪现

#### （2）阶段变化同时动了太多参数
高分思路不是只调一个变量，而是同时在改：

- 宝箱数
- buff 数
- 第二只怪出现时间
- 怪物加速时间
- 最大步数

这会导致阶段之间不像“平滑过渡”，而更像“换题”。

#### （3）阶段内部范围也比较宽
例如 `hard_generalization`：

- 第二只怪：120 ~ 320
- 怪物加速：140 ~ 420

这非常适合最后做泛化，但不适合太早上线。

#### （4）max_step = 2000 会放大当前 workflow 的缺点
当前代码是整局收样本后才送 learner。

如果直接把 `max_step` 提到 2000，就会带来：

- 样本进入 learner 的延迟更大
- GPU 更容易呈现“等很久，算一下，再等很久”
- 样本更陈旧
- PPO 更新更容易受到 stale data 影响

---

## 三、下一步的总原则

下一步不要直接改成“高分思路原版”，而应该遵循下面这个顺序：

### 总体路线
1. **先修训练链路**
2. **再上平滑版 curriculum v1**
3. **最后再逐步逼近高分思路参数**

换句话说：

> 先解决“训练是否稳定”，再解决“课程是否够强”。

---

## 四、第一优先级：先修底层训练问题

这部分优先级高于 reward、特征和 curriculum。

### 1. 修复 GAE 的 done 截断问题

这是当前最优先要修的点。

你当前 `_calc_gae` 的问题是：

- `delta` 里没有乘 `(1 - done)`
- `gae` 递推里也没有乘 `(1 - done)`
- `sample_process()` 里直接把下一帧 value 赋给当前帧 next_value，没有在 episode 结束时截断

这会导致：

- 跨局 bootstrap
- reward_sum 污染
- critic 学到错误目标
- critic 一旦脏掉，actor 也会跟着偏

### 应该怎么改

建议逻辑改成：

- 如果当前样本 `done = 1`，则 `next_value = 0`
- `delta = reward + gamma * next_value * (1 - done) - value`
- `gae = delta + gamma * lambda * (1 - done) * gae`

这是必须优先修复的。

---

### 2. 暂时不要把 max_step 直接提到 2000

在你还没有改 flush 机制之前，建议：

- 暂时保持 `max_step = 1000`
- 不要先把高分思路里的 2000 直接搬过来

因为当前还是整局送样本，直接拉长 episode 只会：

- 让 learner 等得更久
- 让样本更旧
- 让 GPU 利用率更碎
- 放大 stale data 问题

**只有在你把“整局 flush”改成“每 N 步 flush 一次”之后，才建议再试 1500 或 2000。**

---

### 3. 把“每局 load latest”改成低频同步

当前 workflow 在每局 reset 后都会调用：

- `load_model("latest")`

这会带来：

- 频繁的模型加载开销
- 采样节奏被切碎
- actor 侧推理与同步过于频繁
- 训练/采样闭环不平滑

### 建议改法
改成以下任一种低频同步方式：

- 每 **10 ~ 20 个 episode** 同步一次
- 每 **100 ~ 200 个 train_global_step** 同步一次
- 每 **30 ~ 60 秒** 同步一次

重点是：

> 不要每局同步。

---

### 4. 给 critic 增加 value / return normalization

当前算法里已经做了：

- advantage normalization

但没有做：

- value normalization
- reward_sum normalization
- running mean/std

而你当前 reward 是很多项加总得到的，所以在 curriculum 变化时，reward 分布和 return 分布都可能跟着变。

### 建议做法
至少做其中一种：

#### 方案 A：对 `reward_sum` 做 running mean/std normalization
这是更标准的做法。

#### 方案 B：先做 reward scaling
例如把 reward 整体乘一个常数，例如：

- `0.2`
- `0.5`

这虽然简单，但也能先帮助 critic 稳定下来。

---

## 五、第二优先级：优化采样与 learner 的供数节奏

### 1. 不要继续坚持“整局结束才送样本”

当前的 workflow 是：

- 一整局结束
- 再把 `collector` 送去 `sample_process()`
- 再发给 learner

这会导致：

- GPU 利用率忽高忽低
- 样本到达 learner 延迟大
- max_step 越长，这个问题越严重

### 建议改法
改成：

- 每 **128 步** flush 一次
- 或每 **256 步** flush 一次

这样可以做到：

- 样本更快进入 learner
- 更新更平滑
- 样本更“新鲜”
- GPU 更容易持续工作

### 2. actor 尽量减少占 GPU
如果 actor 和 learner 共用 GPU，那么建议：

- actor 侧尽量使用 CPU 推理
- GPU 尽量专门留给 learner

这样可以减少：

- 小推理任务打断 GPU
- 训练任务被零碎切分
- 利用率“打一枪空很久”的现象

---

## 六、第三优先级：上一个更平滑的 Curriculum v1

这一版 curriculum 的目标是：

- 保留“前期资源丰富、后期压力变大”的大方向
- 但不要像原高分思路那样跳得太快
- 并且先维持 `max_step = 1000`

### Phase 1：warmup_stable
**建议持续到 600 ~ 800 个训练 episode**

- 宝箱数：`9 ~ 10`
- buff 数：`2`
- 第二只怪出现：`260 ~ 320`
- 怪物加速时间：`420 ~ 520`
- max_step：`1000`

#### 目标
先让模型稳定学会：

- 低风险拿箱
- 不撞墙
- 怪逼近时会拉开
- 真危险时才交闪现

这个阶段不只是“活下来”，而是“学会安全推进”。

---

### Phase 2：mid_pressure
**建议持续到 1800 ~ 2500**

- 宝箱数：`8 ~ 10`
- buff 数：`1 ~ 2`
- 第二只怪出现：`220 ~ 300`
- 怪物加速时间：`320 ~ 460`
- max_step：`1000`

#### 目标
开始让模型适应中等压力，重点观察：

- treasure_pickup 是否还能保持
- step_score 是否开始崩
- 撞墙和卡住是否明显增多

---

### Phase 3：late_speedup_survival
**建议持续到 4000 ~ 5000**

- 宝箱数：`7 ~ 10`
- buff 数：`1 ~ 2`
- 第二只怪出现：`170 ~ 240`
- 怪物加速时间：`240 ~ 360`
- max_step：`1000`

#### 目标
重点训练“进入高压后还能不能活”。

这一阶段开始：

- 让模型真正接触较早 speedup
- 让空间选择和保命变得更重要
- 但仍然不要把参数一下收紧到 hardest 分布

---

### Phase 4：hard_generalization
**5000 之后**

- 宝箱数：`6 ~ 10`
- buff 数：`0 ~ 2`
- 第二只怪出现：`120 ~ 320`
- 怪物加速时间：`160 ~ 420`
- max_step：`1000`

如果后续你已经完成了：

- 每 N 步 flush
- learner 供数稳定

再考虑把 `max_step` 提高到：

- `1500`
- 再到 `2000`

#### 目标
这时候才开始做真正泛化训练。

---

## 七、阶段推进方式：不要只按 episode 硬切

### 当前更好的原则
不要用：

- 150 / 500 / 900 到点就切

而是改成：

> **最少局数 + 表现门槛** 一起决定是否升级。

当前代码里其实已经有这套思路的基础：

- `curriculum_score_ema`
- `curriculum_step_score_ema`
- `curriculum_post_speedup_steps_ema`
- `curriculum_treasure_pickup_ema`
- `curriculum_speedup_reached_ema`

所以推荐你真正用起来。

---

### 1. Warmup → Mid 的升级条件
至少满足：

- 最少局数：`600`
- `treasure_pickup_ema >= 0.35`
- step_score EMA 已经比较稳定
- blocked_moves / wall_hits 明显下降

---

### 2. Mid → Late 的升级条件
至少满足：

- 最少局数：`1200`
- `speedup_reached_ema >= 0.20`
- `treasure_pickup_ema >= 0.25`
- step_score 没有明显塌陷

---

### 3. Late → Hard 的升级条件
至少满足：

- 最少局数：`1800`
- `speedup_reached_ema >= 0.50`
- `post_speedup_steps_ema >= 15 ~ 25`
- official_score 和 step_score 开始恢复增长

---

## 八、如果你是多 actor 分布式，建议切到全局 curriculum 计数

当前代码里 curriculum 更偏向：

- 用 local_train_episode_cnt 推进

如果你后面是多 actor 分布式训练，这样容易出现：

- 有的 actor 还在 warmup
- 有的 actor 已经到 late
- 整个系统采样分布不一致

### 建议
如果确认是多 actor 分布式训练，建议：

- 打开 `CURRICULUM_USE_GLOBAL_EPISODE`
- 用全局 episode 推进 curriculum

这样课程推进会更一致。

---

## 九、验证方式也要改：不要只评“当前阶段”

当前 validation 更像是：

- 在当前 stage 下做验证

这会有一个问题：

- 看起来 val 在涨
- 但不一定代表真实泛化在涨

### 建议固定三套评测环境

#### Eval-Easy
- 宝箱：`9 ~ 10`
- buff：`2`
- 第二只怪：`260 ~ 320`
- 加速：`420 ~ 520`
- max_step：`1000`

#### Eval-Target
- 使用你最终正式想打的目标分布

#### Eval-Hard
- 宝箱：`6 ~ 10`
- buff：`0 ~ 2`
- 第二只怪：`120 ~ 320`
- 加速：`160 ~ 420`
- max_step：`1000`

### 这样做的好处
你就能区分：

- 是真的变强了
- 还是只是在当前 curriculum 分布里“做题做熟了”

---

## 十、reward 这一轮先别大改，遵循“少改动原则”

当前 reward 已经不是静态死权重，而是一整套课程化 schedule。

因此这一轮不建议你立刻同时做下面这些重构：

- 把所有 shaping reward 全部退火到 0
- 全面重写 flash 逻辑
- 把所有惩罚都改成硬阈值触发

### 当前更稳妥的做法
这一轮只做两件事：

1. 修 GAE 和 value 稳定性
2. 平滑课程学习的环境参数

### 什么时候再动 reward
只有在你后面仍然明显观察到这些行为时，再回头改 reward：

- 原地抽搐
- 既贪宝箱又怕死
- 怪逼近时来回横跳
- 闪现使用极度矛盾
- 高压阶段策略冲突很明显

---

## 十一、按优先级给出下一步执行顺序

### 第 1 优先级：立即修改
- 修复 GAE done 截断
- 继续保持 `max_step = 1000`
- 取消“每局 load latest”，改为低频同步

### 第 2 优先级：接着修改
- 给 critic 增加 value / return normalization
- 改成每 `128` 或 `256` 步 flush 一次样本
- 如果 actor 和 learner 共用 GPU，尽量让 actor 改为 CPU 推理

### 第 3 优先级：上线 Curriculum v1
- 不要直接用 `150 / 500 / 900`
- 使用更长 warmup、更平滑中间阶段
- 先主要平滑 `monster_interval` 和 `monster_speedup`
- 宝箱 / buff 先只做小幅变化

### 第 4 优先级：最后再逼近高分思路原版
- 训练链路稳定之后
- 再逐步缩短 warmup
- 再尝试把 `max_step` 提到 1500 / 2000
- 再放宽 hard 阶段内部随机范围

---

## 十二、最终建议（一句话版）

你现在最需要避免的是：

> **在“GAE 还漏、样本还是整局送、每局还同步最新模型”的前提下，直接上一个更激进、更长局、更高压的 curriculum。**

那样即使训练崩了，也很难分清：

- 是 curriculum 本身太激进
- 还是训练链路在放大问题

更合理的路线应该是：

> **先修训练链路，再上平滑版 curriculum v1，最后再向高分思路的参数逐步逼近。**

---

## 十三、可直接执行的短版清单

### 现在立刻做
- [ ] 修 GAE done 截断
- [ ] 暂不把 max_step 改到 2000
- [ ] 取消每局 `load_model("latest")`
- [ ] 给 critic 加 normalization

### 接下来做
- [ ] 改为每 128 / 256 步 flush
- [ ] actor 尽量少占 GPU
- [ ] 上平滑版 curriculum v1

### 后面再做
- [ ] 用“局数下限 + 表现门槛”推进阶段
- [ ] 固定 easy / target / hard 三套验证环境
- [ ] 再考虑向原高分 curriculum 收紧

