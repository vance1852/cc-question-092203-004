# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、场地边界和间距约束、遗传算法与粒子群优化、经济性分析以及无界面图表输出。

## 安装

建议使用 Python 3.10 或更新版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 可以使用 `.venv\\Scripts\\Activate.ps1` 激活环境。

## 快速验证

```bash
python quick_test.py
```

快速验证会覆盖模型、约束、年发电量、优化、经济性和图表生成，并在 `test_output/` 写入临时图片。该目录不会纳入版本控制。

经济性计算的单元测试（平价情景零实际折现率、负/多根 IRR、退役费现值、盈利/亏损/临界场景）：

```bash
python -m unittest tests.test_economics -v
```

## 经济口径说明

NPV、LCOE、IRR 共用同一组寿命期现金流：初始投资发生在 `t=0`，运维与发电收益按年发生，退役拆除费用在寿命期末 `t=n` 一次性计入，再按费雪方程得到的实际折现率折现。名义折现率等于通胀率（平价情景，实际折现率为 0）时计算连续不中断；IRR 允许为负，无根、多根、无有效发电量（LCOE 为 `inf`）均有明确的返回状态，见 `results.json` 中 `irr_status` / `lcoe_status`。可通过 `--discount-rate` 与 `--inflation-rate` 分别设置名义折现率与通胀率。

## 完整分析

```bash
python -m wind_farm_opt --help
python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50 --output-dir output
```

也可以先生成配置文件，再通过 `--config` 运行：

```bash
python -m wind_farm_opt --generate-config my_config.json
python -m wind_farm_opt --config my_config.json
```

所有运行结果默认写入 `output/`，可以用 `--no-plots` 跳过图表生成。命令行使用无界面绘图后端，适合容器和服务器环境。
