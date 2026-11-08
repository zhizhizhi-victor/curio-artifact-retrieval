# 拾遗 Curio

面向中国古代文物的中英文**文本到图像检索系统**。默认使用领域适配图文模型，当前生产属性头排序权重为0；馆藏描述用于展示、训练弱监督和单独的元数据辅助对照。默认在线图像排序不读取馆藏文本匹配分数。

**组员先读：[组内讨论板 DISCUSSION.md](DISCUSSION.md)**，包含当前思路、算法与模型、实测效果、局限、待讨论议题及留言模板。团队协作流程见 [docs/COLLABORATION.md](docs/COLLABORATION.md)，贡献和拉取请求检查见 [CONTRIBUTING.md](CONTRIBUTING.md)。

**完整资源下载：[GitHub Release：10,200张CC0图片库、底模、双语属性对比LoRA权重、向量索引与实验记录](https://github.com/zhizhizhi-victor/curio-artifact-retrieval/releases/tag/curio-resources-v3-10k)**。资源作为同仓库 Release 附件共享，避免将数 GiB 二进制文件写入 Git 历史。安装、分包说明和许可见 [RESOURCES.md](docs/RESOURCES.md)，固定哈希见 [resources_manifest.json](docs/resources_manifest.json)。

```bash
# 克隆后先恢复完整运行资源（含 10,200 图库、SigLIP2/DINO 底模、LoRA 与索引）；仅需 Python 3.11+，支持断点续传和 SHA256 校验
python scripts/download_resources.py
# 若需要额外下载 5,508 张新增采集博物馆原图，改用 --all
```

> **核心升级（v3-10k 多条件纯视觉检索与规模扩库，详见 [docs/COMPOSITIONAL_10K_REPORT.md](docs/COMPOSITIONAL_10K_REPORT.md)）**：
> 1. **10,200 件跨四馆 CC0 中国文物扩库**（`collection_10k.jsonl`：Met 3,904、MIA 2,955、Cleveland 1,810、AIC 1,531），全部经 SHA256 / 256-bit aHash / 64-bit pHash 与参考图隔离校验，且保持原 4,692 件冻结快照 `collection_v2.jsonl` 哈希零改动。
> 2. **免重训多条件组合纯视觉检索（`subquery_and` / `compositional`）**：主体锚定子句拆解 + 跨语言可见形态桥接 + 候选池 Soft-AND 短板惩罚 + F1 可靠度校准对数概率探针门控，将盲测集双语均值 `nDCG@10` 从 `0.8615` 提升至 `0.9017`（中文 `+0.0740`, $p=0.0060$）。
> 3. **双语属性对比空间池化头 LoRA（`siglip_lora_comp`）**：微调 `SigLIP-2` 空间多头注意力池化层 `vision_model.head` + NegCLIP 式单属性反事实硬负例，在 `visual_test` 创下英文 `nDCG@10 = 0.9345`，并在 4,692 全库 160 题评测上将英文 / 中文 `nDCG@10` 提升至 **`0.4469 (+0.0399) / 0.4628 (+0.0444)`**（`P@5` 达 `56.0% / 56.8%`）。

本地基准包含 **4,692 件冻结 CC0 图像**（Met 2,954、Cleveland 1,738）与 **10,200 件扩展 CC0 图库**、预训练编码器、训练和评测脚本、FastAPI 后端和 React 界面。正式结果见 [docs/COMPOSITIONAL_10K_REPORT.md](docs/COMPOSITIONAL_10K_REPORT.md) 与 [docs/RESULTS.md](docs/RESULTS.md)；原始日志和模型权重保存在本地 `artifacts/` 与 `logs/`。

**本轮新增**：真实馆外参考图→本地图图匹配、DINOv2、器型/前景颜色/纹理消融、DINO局部patch研究模式、可信性诊断修正，以及101张新图组成的可选 **4,793图扩展图库**。参考图有官方来源、许可、哈希及全库同物/近重复排除；在线按缓存参考图选择，未伪称实时LLM联网。8类意图在全4,692图召回后，对244张不同图片、257条查询/图片对做了AI像素盲判，结果包含负例，专门分支未证明普遍更优。详细见 [新增路线与实测](docs/REFERENCE_RETRIEVAL_RESULTS.md)、[来源与扩展](docs/REFERENCE_SOURCES.md)、[当前状态](docs/CURRENT_STATE.md)。

## 直接运行

macOS / Linux 一键启动（自动启用 Apple Silicon `mps` / CUDA，并挂载 10,200 张扩展图库）：

```bash
# 推荐：加载新微调双语属性对比 LoRA + 10,200 张扩展图库 + 多条件组合检索
bash scripts/run.sh --comp-lora

# 或：加载原冻结 LoRA + 免重训多条件组合检索
bash scripts/run.sh --compositional
```

Windows PowerShell 启动：

```powershell
.\scripts\run.ps1
```

浏览器打开 `http://127.0.0.1:8000`。启动器核验冻结数据、图像索引与分类头的哈希，默认读取 `artifacts/visual_selection.json` 中只由视觉 dev 选择的生产配置，并启用离线模式；缺失资源会报出具体原因。`-CheckOnly` 只检查资源。手动比较冻结 CLIP 可用 `-Encoder clip -Index artifacts/clip`；不带分类头的 `image_only` 会如实标出回退，原始 CLIP 对比可在界面选择 baseline。

研究模式端点 `POST /api/search/level3` 先做纯视觉召回，再对器型、材质和年代使用馆藏字段做证据重排，逐条返回 catalogue evidence。它用于研究三级查询（例如“明代青花瓷瓶”），不改变默认 `image_only`，也不把年代或真实性当作图像结论。

界面新增“文字寻找参考图”和“上传图片检索”。前者调用 `POST /api/search/reference`：例如“康熙梅瓶”，先显示馆外参考照片与来源年代，再用照片比图库像素。后者向 `/api/search/image`发送图片原始字节。可选SigLIP、DINO、局部patch和传统专门特征；拒绝分割和未覆盖的查询会明确显示。文本检索可选“扩展图库”，使用 `/api/search/expanded`；原v2指标不适用于新图库。

首次安装源码环境：

```powershell
uv sync --extra dev
pnpm --dir web install --frozen-lockfile
pnpm --dir web build
.\scripts\run.ps1 -CheckOnly
```

Python 3.11–3.12；开发设备为 Windows、RTX 3070 8GB、32GB RAM。`uv.lock` 固定 Python 依赖，`web/pnpm-lock.yaml` 固定前端依赖。PyTorch 默认 CUDA 12.8；无 CUDA 时编码器使用 CPU，速度需另测。模型命令可显式 `--device cpu`。

## 三层视觉任务

- **基本类别**：CLIP B/32 与 SigLIP2 base 的图文检索；SigLIP 图像特征上的器型/材质分类头。材质标签来自馆方，是弱监督，视觉预测不能证明真伪或实际材质。
- **颜色、器形与纹饰**：AI 只看盲图制作原始240 train / 60 dev / 60 test的外观caption及属性标签，关系分组后保留218张训练图；训练独立 sigmoid 属性头和图像塔 LoRA。黑白照片颜色未知、模糊细纹饰 null，不把未知当负例。
- **专业表述**：项目手工规则把青花、粉彩、描金、镂空等展开成普通外观词，记录展开依据。乾隆等年代、转心结构、未知专名不声称已由像素核验；“流心”不偷偷纠正为另一个词。

算法假设、实验隔离与预处理决策见 [VISION_ROUTE.md](docs/VISION_ROUTE.md)、[MODELS.md](docs/MODELS.md) 和 [SPECIALIST_QUERY.md](docs/SPECIALIST_QUERY.md)。参考图已通过实际馆方API采集；线上使用可追溯缓存。当前没有外部LLM或大模型核验器，规则不冒称模型推理。

## 检索方法

| 方法 | 排序使用的信息 | 用途 |
|---|---|---|
| `baseline` | 原文文本向量 × 图像向量 | 原始零样本基线 |
| `translated` | 规则展开文本 × 图像向量 | 中英查询处理对照 |
| `image_only`（默认） | 重写后的图文相似度 + 可选属性探针 | 生产视觉检索 |
| `subquery_and` | 主体锚定子句拆解 + 视觉形态桥接 + Soft-AND 合取 | 免探针纯图像多条件组合检索 |
| `compositional` | `subquery_and` + F1 可靠度校准对数概率探针门控 | 完整纯图像多条件组合检索 |
| 参考图 SigLIP / DINO | 外部参考图 × 本地图像向量 | 图图匹配 |
| 形状/颜色/纹理及融合 | 前景遮罩与图像特征 | 专门视觉研究分支，有覆盖限制 |
| DINO patch | 全图召回100 + 前景局部对应/软轮廓 | 独立弱标签实验，未证明稳定提升 |
| `hybrid` | 图像向量、BM25、馆藏槽位的 RRF | 元数据辅助对照 |
| `visual` | hybrid + 图像分类概率 | 旧辅助方法对照 |
| `lexical` | 馆藏文字 BM25 | 元数据诊断 |

`image_only` 不调用 BM25 或元数据冲突评分；更换图库文字不改变其排名，有回归测试。没有图像索引时返回 unavailable，缺失分类头时明确标明 dense_rewritten 回退。分类头与当前图像特征必须严格匹配。文本入口的器型/材料/年代/来源筛选来自馆藏字段，严格执行；空结果不会自动放宽。参考图/上传入口明确禁用这些筛选，图库排名只使用图像通道。

相似度、融合排序分数和分类概率分别展示；它们都不是“人工确认”。否定仅对已覆盖的图像属性使用 1−P，缺失属性保持未知。数量和空间关系不伪称已验证。

## 数据与审计

训练和服务使用冻结主表：`data/snapshots/collection_v2.jsonl`。SHA256：

```text
4dbb5e4045bbab4fd245f7fce2d2fc363aabfe41a45d31b91703ddaac5ebee68
```

对象 split：train 3,474 / dev 568 / test 650。原始 JSON、原图、处理图和缩略图保留；原图短边≥256、长边≥512，处理图最长1024，缩略图最长320，不放大原图。`sha256` 是原图哈希，`image_sha256` 与 `thumb_sha256` 分别校验实际衍生文件。`phash` 历史字段实际为16×16 aHash，只按完全相同哈希分组，不声称已做Hamming距离近重复聚类。

来源许可证据、采集边界和失败记录见 [DATA.md](docs/DATA.md)。已有数据无需再下载。采集复现会访问公开馆方接口（网络可能限流）：

```powershell
.\.venv\Scripts\python.exe scripts/data_ingest.py --target 5000 --met-candidates 9000 --cleveland 2500 --workers 8
.\.venv\Scripts\python.exe scripts/rebuild_from_raw.py
.\.venv\Scripts\python.exe scripts/data_audit.py --data data/snapshots/collection_v2.jsonl
```

5,000是采集目标，不是虚报实际数量。不要在训练期间覆盖冻结快照或修改图像；新增数据应另建版本并重建索引。Met受限时实际通过Chrome访问官方新搜索接口取得候选列表，保存在 `data/raw/met_browser_*.json`，不是仅凭先前数据源评估的估计。

## 训练和评测复现

以下步骤会真实使用GPU、保存checkpoint和每epoch日志。先运行基础对照（两个预训练模型、馆藏caption LoRA随机/hard各6epochs、基础分类头），随后运行视觉主实验：

```powershell
.\.venv\Scripts\python.exe scripts/run_experiments.py --data data/snapshots/collection_v2.jsonl --phase dev --epochs 6 --batch-size 16 --resume
.\.venv\Scripts\python.exe scripts/run_visual_experiments.py --phase dev --epochs 12 --resume
```

视觉原始标注与查询已保存。重新合并时：

```powershell
.\.venv\Scripts\python.exe scripts/merge_visual_labels.py
.\.venv\Scripts\python.exe scripts/merge_visual_labels.py --queries eval/visual_dev
.\.venv\Scripts\python.exe scripts/merge_visual_labels.py --queries eval/visual_test
```

LoRA只更新图像attention的q/v低秩参数，文本塔冻结，多正例InfoNCE加teacher特征锚定。视觉caption训练使用218图，随机/hard各12epochs；epoch只按60图dev配对R@5/MRR选择。hard batch来自冻结teacher图像近邻，不读取test。分类正则化与重排权重只按dev选择。冻结选择后才执行test：

```powershell
.\.venv\Scripts\python.exe scripts/run_experiments.py --phase test --epochs 6 --batch-size 16 --resume
.\.venv\Scripts\python.exe scripts/run_visual_experiments.py --phase test --epochs 12 --resume
.\.venv\Scripts\python.exe scripts/summarize_results.py
.\.venv\Scripts\python.exe scripts/summarize_visual_results.py --require-complete --failure-cards
```

`--resume` 仅跳过同一数据和配置下的完成步骤。改变实验数据或参数应使用新输出目录/状态，不应改test后重新选择。索引构建和单次训练的底层命令见 [MODELS.md](docs/MODELS.md) 与CLI `--help`。

## 如何解读效果

1. **主要视觉审计**：dev/test各独立60图、各24条中英意图，逐query穷举0–3级相关性。AI直接看图判断，不由元数据词重合或检索排名生成。它是小图库AI审计，不是4,692图全库的真人gold。
2. **视觉caption配对**：留出的60图与AI外观caption做图文配对，报告R@1/5/10和MRR。caption文字风格与标注者差异仍会影响结果。
3. **辅助元数据对照**：完整650个test对象配对；80个双语silver意图在全库检索，衡量馆藏字段符合程度。高分不能证明视觉理解更强。

自由查询报告nDCG@10、P@5、AP@10、Recall@20、MRR@10及95% bootstrap区间。二值相关阈值grade≥2；nDCG增益2^grade−1；AP@10分母是完整已知相关项数。中英表述按语言分别统计，不当成两倍独立意图。预声明方法比较保留正负差值，按每组“5指标×2语言”做Holm校正。失败样例固定按最差查询选出，不能只展示成功样例。

当前没有真人标注，不能报告假双标κ。真人复核工具入口：

```powershell
.\.venv\Scripts\python.exe -m curio.evaluate.annotation --help
```

盲标池、独立标注者、修订链、冲突导出及一致性接口见 [API.md](docs/API.md)。组员人工复核是下一阶段可使用的工作，不被冒记为已完成。

## 测试与结构

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
pnpm --dir web build
.\scripts\run.ps1 -CheckOnly
# 先启动8000后执行真实顺序/4并发请求
.\.venv\Scripts\python.exe scripts/system_check.py --count 200 --workers 4
```

```text
src/curio/     query、图文编码/训练、index、retrieve、evaluate、serve
scripts/       采集、盲图卡、实验编排、汇总和系统验收
web/           React/TypeScript前端
 data/         原始来源、图像、台账、处理表与冻结快照（不进Git）
eval/          盲图标签、查询、显式qrels和标注协议产物
artifacts/     模型适配器、索引、分类头、指标与哈希（不进Git）
docs/          开发路线、模型/数据说明、结果、失败分析、AI披露
```

模型和图片通过稳定item ID访问，API不暴露本机文件路径。系统默认只监听127.0.0.1；用户反馈追加保存到本地，不自动训练。架构见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)，开发连续性与剩余事项见 [DEVELOPMENT.md](docs/DEVELOPMENT.md)。

这是开发和测试交付，最终课程报告、PPT、组员贡献归属与同侪评议尚不在本阶段。AI参与范围和实际核验方式见 [ai_log.md](docs/ai_log.md)。
