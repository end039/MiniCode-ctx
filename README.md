<div align="center">

# 容器化终端编程 Agent — MiniCode (fork)

### 在零依赖终端编程 Agent 上自研 **上下文工程 / 多 Agent / 后台记忆 / Docker 沙箱 / SWE-bench 评测**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Dependencies: 0 (core)](https://img.shields.io/badge/core%20deps-0-f97316?style=flat-square)](pyproject.toml)
[![Fork of MiniCode-Python](https://img.shields.io/badge/fork-MiniCode--Python-6366F1?style=flat-square)](https://github.com/QUSETIONS/MiniCode-Python)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e?style=flat-square)](LICENSE)

</div>

> fork 自零依赖终端编程 Agent [MiniCode-Python](https://github.com/QUSETIONS/MiniCode-Python)，在其 agent loop 之上实现了 **上下文工程、探索子 Agent、后台记忆、Docker 沙箱执行、SWE-bench 评测** 五大能力，并打通「**容器内推理 → 官方评分**」闭环。

---

## ✨ 在上游基础上新增的能力（本项目重点）

### 1. 上下文工程：两层轮内压缩 + 落盘
- **两层压缩**（`minicode/context_compactor.py`）：micro-compact（清理旧工具输出，不调 LLM）+ auto-compact（LLM 摘要折叠历史、保留最近 token）。
- **真实用量驱动**：用 provider 返回的 `usage`（而非字符估算）计算占用、分级阈值，在**每次模型调用前**触发，长任务上下文稳定压在窗口内（默认 256k，`MINI_CODE_CONTEXT_WINDOW` 可调）。
- **大工具结果落盘**（`minicode/tool_result_store.py`）：超阈值的工具输出写盘，上下文只留预览 + 路径。

### 2. 多 Agent：探索/检索子 Agent（上下文隔离）
- `dispatch_agent` 工具（`minicode/subagent.py` + `minicode/tools/dispatch_agent.py`）：主 Agent 把"全局 where/how"类检索**委派**给**只读**子 Agent，后者在隔离上下文里 grep/read 后**只回一段结论**——主线上下文从"十几个文件原文"降到一段摘要。
- 意图识别走**工具描述、模型驱动**（对齐 Claude Code 的 Task），非关键词匹配；触顶强制收尾，避免返回占位串。

### 3. 后台记忆：低信号、廉价 LLM 蒸馏
- `minicode/background_memory.py`：独立 daemon 线程，仅消费「用户问题 + 每轮总结 + 工具名」（隐藏代码/工具大文本），提炼持久事实写入**分层记忆文件**并跨会话注入系统提示；退出同步兜底防丢。`MINI_CODE_BACKGROUND_MEMORY=0` 关闭。

### 4. Docker 沙箱执行（让 Agent 在容器里干活）
- `minicode/exec_backend.py` + `ToolContext.container`：`read/write/edit/grep/run_command` 等工具**透明地在 Docker 容器内执行**（`docker exec` / `cp`），仓库留容器内、**不做 volume 挂载**。
- **host 执行路径零回归**（默认行为字节不变），容器本身即沙箱边界。

### 5. SWE-bench Lite 评测闭环
- `benchmarks/swe_bench_runner.py`：逐实例起官方镜像、**容器内**跑 Agent、提取 `git diff` 产出 `predictions.jsonl`；支持断点续跑、并行、镜像/磁盘治理。
- **评分交给官方** `swebench.harness.run_evaluation`（应用补丁 + gold 测试、独立容器判分），不自己复刻。


---

## 📊 SWE-bench Lite 评测结果（django 子集）

> **27 / 30 resolved**，模型 `deepseek-v4-pro`，官方 `run_evaluation` 判分，0 空补丁、0 harness 错误。

⚠️ 这 30 例是**按"最易"挑选**的 django 实例（均 `FAIL_TO_PASS=1`、gold patch 最短），所以该数字**偏乐观，不代表全 SWE-bench Lite 榜单分**。本项目的价值在于**端到端打通了工业级评测管线**（容器隔离执行、官方评测对接、断点续跑、OOM / 网络排障），而非这个百分比本身。复现步骤见 [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)。

---


### 用 DeepSeek 驱动
`~/.mini-code/settings.json`（或同名环境变量）：
```json
{
  "model": "deepseek-v4-pro",
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
    "ANTHROPIC_API_KEY": "sk-..."
  }
}
```

### 跑 SWE-bench Lite（django 子集）
```bash
pip install swebench datasets
source deepseek.env                                  # 导出 ANTHROPIC_BASE_URL/MODEL/API_KEY
python benchmarks/pick_django.py 30                  # 选 30 个最易的 django 实例
python benchmarks/swe_bench_runner.py --parallel 1 --keep-images   # 容器内推理 → predictions.jsonl
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path benchmarks/predictions.jsonl \
  --run_id minicode-django30 --cache_level instance --max_workers 4
```

---

## 🏗️ 关键模块（新增 / 改动）

| 文件 | 职责 |
|---|---|
| `minicode/context_compactor.py` | 两层轮内上下文压缩（micro / auto），真实用量驱动 |
| `minicode/tool_result_store.py` | 大工具结果落盘，上下文留预览 + 路径 |
| `minicode/subagent.py` · `tools/dispatch_agent.py` | 探索/检索子 Agent（只读、上下文隔离、容器传播） |
| `minicode/background_memory.py` | 后台廉价 LLM 蒸馏持久记忆 |
| `minicode/exec_backend.py` | 容器感知执行后端（host / container） |
| `minicode/agent_loop.py` | 接入压缩 / 真实用量 / 思考 round-trip / `tool_container` |
| `minicode/anthropic_adapter.py` | 解析 `usage`、思考块回传 |
| `minicode/tty_app.py` | 用量 meter、压缩提示、`/compact`、`/resume`、子 Agent 计数、命令门控 |
| `benchmarks/` | `pick_django.py` 选样 · `swe_bench_runner.py` 推理 · `RESULTS.md` 结果 |

---

## 🧩 基础能力（继承自上游 MiniCode-Python）

零依赖（核心仅标准库）的双语终端编程助手：备用屏幕 TUI、多轮 agent loop、30+ 内置工具（文件 I/O、代码搜索、shell、git、测试、代码智能等）、权限系统、会话持久化、分层记忆、MCP 集成、斜杠命令。详见上游仓库文档。

---

## 🧪 测试

本仓新增的能力配有 `unittest` 用例（无网络、可离线跑）：

```bash
python tests/test_context_compactor.py
python tests/test_background_memory.py
python tests/test_subagent.py
python tests/test_exec_backend.py        # fake-docker，无需守护进程
python tests/test_tui_features.py
```

> 部分上游测试为 pytest 风格，需 `pip install pytest` 后 `pytest` 运行。

---

## 🙏 致谢 / 上游

- **[MiniCode-Python](https://github.com/QUSETIONS/MiniCode-Python)**（[@QUSETIONS](https://github.com/QUSETIONS)）— 本项目的 fork 基座（Python 实现）。
- **[MiniCode](https://github.com/LiuMengxuan04/MiniCode)**（[@LiuMengxuan04](https://github.com/LiuMengxuan04)）— 最初的 TypeScript 实现。


本仓为学习用途的二次开发，新增能力上下文工程、多 Agent、后台记忆、Docker 沙箱、SWE-bench 评测。

## 📄 License

MIT（沿用上游）— 见 [LICENSE](LICENSE)。
