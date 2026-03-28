# 🐟 WeiboFish: 政务新媒体多智能体与大模型全景仿真沙盘

**WeiboFish** 是一款专为**公共管理与社会科学实证研究**设计的政务新媒体舆情仿真沙盘。  
本项目在开发过程中受到了 MiroFish 项目关于多智能体仿真逻辑的启发。WeiboFish 在此基础上，针对政务新媒体场景进行了实证数据校准、引入了 FAISS 稠密向量检索机制（RAG），并对 Agent 推演与智库研判功能进行了深度迭代。特此向原作者与研究团队表示感谢。

---

## ✨ 核心亮点 (Features)

1. **🔬 定量理论基座 (PyTorch)**：内置基于真实数据的多元回归模型参数，通过双向 LSTM 和 RoBERTa 实时提取通报文本的可读性与情绪强度，测算先验的“理论互动概率”。
2. **📚 历史案卷对标 (RAG + FAISS)**：系统可在毫秒级从 **32 万条**历史政务微博中检索相似案例，并提取真实互动量（转/赞/评）作为大模型决策依据。
3. **🤖 微观群体行为漏斗 (MAS)**：多样化网民智能体参与推演，形成点赞、转发、评论与互动裂变，展示群体心理演化。
4. **🧠 智库级内参专报 (DeepSeek-Reasoner)**：推演结束后自动生成结构化政策内参，融合定量预测、历史案例和推演结果。

---

## 🧭 重构后项目架构说明

### 1. 目录结构（核心部分）

```text
WeiboFish/
├─ app.py                          # Streamlit 入口（页面编排层）
├─ build_vector_db.py              # 构建 FAISS 向量库
├─ generate_1000_personas.py       # 批量生成智能体画像
├─ fix_data.py                     # 数据修复脚本
├─ data/                           # 业务数据、画像、向量索引
├─ logs/                           # 运行日志
├─ src/
│  ├─ config/
│  │  ├─ load_params.py            # 回归参数加载与计算
│  │  └─ reaction_params.json      # 回归参数配置
│  ├─ features/models/
│  │  └─ text_analyzer.py          # 文本特征抽取（RoBERTa + CORAL + BiLSTM）
│  ├─ sim/
│  │  ├─ memory.py                 # 历史记忆检索（RAG）
│  │  ├─ dynamics.py               # 控制台仿真主流程
│  │  └─ personas.py               # 画像生成逻辑（单机版本）
│  └─ webapp/                      # 新增：前端业务分层模块
│     ├─ __init__.py               # 统一导出
│     ├─ data_loader.py            # 数据读取与缓存加载
│     ├─ agent.py                  # Streamlit Agent 行为定义
│     ├─ sampling.py               # Agent 抽样策略
│     ├─ dashboard.py              # 看板图表绘制
│     ├─ comment_renderer.py       # 评论区 HTML 构建
│     ├─ feedback.py               # 反馈表单模块
│     ├─ misc.py                   # 通用工具（ID提取、Word导出、字体等）
│     └─ usage.py                  # 调用日志统计
└─ README.md
```

### 2. 分层职责

- **入口编排层 (`app.py`)**  
  只负责页面交互、流程调度和模块组合。
- **模型与算法层 (`src/features`, `src/config`, `src/sim`)**  
  负责文本分析、回归打分、历史检索与仿真机制。
- **Web 业务层 (`src/webapp`)**  
  将 UI 相关逻辑模块化（抽样、渲染、反馈、工具函数），降低 `app.py` 复杂度。
- **数据与离线脚本层 (`data/`, `build_vector_db.py` 等)**  
  负责索引构建、画像生成与数据准备。

### 3. 运行调用链（简化）

1. `streamlit run app.py`
2. `app.py` 调用 `load_agenda_data()` / `load_ai_engines()` 完成资源加载
3. `text_analyzer.py` + `load_params.py` 计算传播潜力指标
4. `memory.py` 检索历史相似案例（RAG）
5. `sampling.py` 抽样 Agent，`agent.py` 驱动互动行为
6. `dashboard.py` + `comment_renderer.py` 渲染结果
7. `DeepSeek-Reasoner` 输出内参报告，`misc.py` 导出 Word

---

## 🚀 快速开始 (Quick Start)

### 1. 环境依赖与安装

将本项目克隆到本地后，建议使用 Conda 环境：

```bash
cd WeiboFish
pip install -r requirements.txt
```

> 提示：为加速向量计算，建议安装支持 CUDA 的 PyTorch 版本。

### 2. 配置大模型 API 密钥

本项目当前通过页面侧边栏输入 DeepSeek API Key（推荐）。

- 启动后在左侧 `DeepSeek API Key` 输入框填入密钥。
- 代码中也保留了 `MY_API_KEY` 变量位，便于二次开发时改造成环境变量/统一配置。

### 3. 数据准备与向量记忆库构建

为了让智能体拥有历史记忆，请先本地构建 FAISS 向量库。

**获取数据**：可将数据命名为 `文章列表汇总.xlsx` 或 `.csv` 放入 `data/` 目录；也可使用提供的 32 万条开源政务微博数据集。

> 下载链接: https://pan.baidu.com/s/1HTMfa85D14u-X-94lmfzEQ?pwd=6r92  
> 提取码: `6r92`

**构建索引**：

```bash
python build_vector_db.py
```

运行成功后，`data/` 下将生成：

- `weibo_memory.index`
- `weibo_memory_meta.pkl`

### 4. 启动沙盘

```bash
streamlit run app.py
```

浏览器打开：`http://localhost:8501`

---

## 🖥️ 沙盘推演流程指南

启动系统后，在左侧边栏输入待测试的政务通报元信息和文本，点击“启动全景仿真推演”，系统依次执行：

1. **第一阶段：传播潜力定量测算**  
   提取文本特征（可读性、情绪强度、媒介丰富度等），给出基础传播评价。
2. **第二阶段：历史相似案卷检索 (RAG)**  
   检索高相似历史事件并展示真实网民反馈数据。
3. **第三阶段：多智能体推演实况 (MAS)**  
   观察不同人设 Agent 的互动行为（点赞、评论、转发、盖楼）。
4. **第四阶段：政务智库研判专报**  
   生成诊断报告并给出可执行的文本优化建议。

---

## 🖥️ 测试地址

本项目提供公开测试环境（免 API 输入，仅用于体验）。

- 测试链接：http://39.97.49.16:8501
- 建议浏览器：最新版 Chrome / Edge

---

## 🤝 参与贡献

欢迎对**计算社会科学、计算传播学、公共管理智能化**感兴趣的同学提交 Issue 或 Pull Request。  
如果项目对你的论文或研究有帮助，欢迎点个 ⭐ Star。
