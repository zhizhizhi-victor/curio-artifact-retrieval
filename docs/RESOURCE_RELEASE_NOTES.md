完整资源包（`curio-resources-v3-10k`）：包含 Curio v2 冻结基准、**10,200 张跨四馆 CC0 中国文物扩展图库**、**新微调双语属性对比 LoRA（`siglip_lora_comp`）** 以及全部向量索引与实验评测产物。

- **10,200 张 CC0 文物图库**（`collection_10k.jsonl`：Met 3,904、MIA 2,955、Cleveland 1,810、AIC 1,531）+ 4,692 张冻结基准快照（`collection_v2.jsonl`，SHA256 严格未变）+ 101 张 v3 扩展图，包含全部处理图与缩略图；另有可选的 5,508 张新增采集原图包（`originals-10k`）。
- **45 张外部参考照片**、来源 API 原始记录、许可台账、去重隔离审计（`data/expansion_10k/report.json`）。
- **固定 revision 的 SigLIP2 与 DINOv2 底模**，原领域 LoRA + **新微调双语属性对比空间池化头 LoRA（`siglip_lora_comp`）**、五类视觉线性探针、4,692 与 10,200 双规模向量索引（`siglip_10k`、`siglip_10k_comp_lora`）。
- **完整训练与评测记录**：`compositional` / `subquery_and` 消融评测、配对置换检验、10,200 库规模基准与微调历史日志。

每个 ZIP 可独立解压到项目根目录；推荐使用下载脚本自动选择、断点续传、SHA256 校验并恢复路径：

```bash
python scripts/download_resources.py       # 默认运行资源（含 10,200 图库、底模、LoRA 与全部索引）
python scripts/download_resources.py --all # 额外包含 5,508 张新增采集原图
```

恢复资源后安装依赖即可直接启动 Web 服务：

```bash
uv sync --extra dev
npm --prefix web install && npm --prefix web run build
bash scripts/run.sh --comp-lora
```

每个附件均附带独立 SHA256 校验值，`curio-files.jsonl` 提供逐文件哈希清单；附件不含任何凭据或私有会话信息。
