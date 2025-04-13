# lighteval-llm

使用 LightEval 对 LLM 进行评估。

目前已囊括的数据集有：

- MATH 500
- GPQA

## 🛠️ 安装

项目使用 uv 作为项目管理工具，请确保安装了 uv。

```bash
cd src

uv sync
```

## ⚡ 运行评估脚本

使用 LightEval 编写的评估代码放在 `src/eval` 目录下，需要使用 lighteval 命令来启动。相关启动命令已编写成 shell 脚本，放在 `eval-scripts` 目录下。

以运行 MATH 500 为例：

1. 修改 `eval-scripts/eval-math500.sh` 脚本中的模型位置及结果输出目录。
2. 运行 shell 脚本：

```bash
# 在项目根目录下执行
bash eval-scripts/eval-math500.sh
```
