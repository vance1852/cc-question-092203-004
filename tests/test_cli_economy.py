"""CLI 端到端测试：经济性分析摘要与结果文件必须使用同一组数值。

覆盖正常折现率与平价情景（名义折现率 == 通胀率）两种配置。
"""

import json

import pytest

from wind_farm_opt.config import create_sample_config
from wind_farm_opt.cli import WindFarmOptimizerCLI


def run_cli(tmp_path, discount_rate=0.06, inflation_rate=0.025):
    config = create_sample_config()
    config.visualization.save_dir = str(tmp_path)
    config.visualization.save_plots = False
    config.economic.discount_rate = discount_rate
    config.economic.inflation_rate = inflation_rate

    cli = WindFarmOptimizerCLI(config)
    cli.run_full_analysis(
        run_baseline=True,
        run_opt=False,
        run_econ=True,
        run_sweep=False,
        run_viz=False,
        save=True,
    )
    return cli


@pytest.mark.parametrize(
    "discount_rate,inflation_rate",
    [
        (0.06, 0.025),   # 常规情景
        (0.025, 0.025),  # 平价情景：实际折现率为 0，过去在此除零中止
    ],
)
def test_economic_results_consistent_between_summary_and_file(tmp_path, discount_rate, inflation_rate):
    cli = run_cli(tmp_path, discount_rate, inflation_rate)
    er = cli.economic_result
    assert er is not None

    results_path = tmp_path / "results.json"
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 必须是合法 JSON（无 Infinity/NaN）
    econ = data["economic"]

    # 结果文件与分析结果使用同一组数值
    assert econ["discount_rate_nominal"] == pytest.approx(discount_rate)
    assert econ["inflation_rate"] == pytest.approx(inflation_rate)
    assert econ["real_discount_rate"] == pytest.approx(er.real_discount_rate)
    assert econ["lifetime_years"] == pytest.approx(er.lifetime)
    assert econ["total_installed_capacity_mw"] == pytest.approx(er.total_installed_capacity)
    assert econ["net_aep_gwh"] == pytest.approx(er.net_aep)
    assert econ["total_capital_cost_yiyuan"] == pytest.approx(er.total_capital_cost / 1e4)
    assert econ["decommissioning_cost_yiyuan"] == pytest.approx(er.decommissioning_cost / 1e4)
    assert econ["annual_om_cost_wanyuan"] == pytest.approx(er.total_om_cost_annual)
    assert econ["annual_revenue_wanyuan"] == pytest.approx(er.annual_revenue)
    assert econ["lcoe_yuan_per_kwh"] == pytest.approx(er.lcoe)
    assert econ["npv_yiyuan"] == pytest.approx(er.npv / 1e4)
    assert econ["irr_pct"] == pytest.approx(er.irr)
    assert econ["irr_status"] == er.irr_status
    assert econ["irr_all_roots_pct"] == pytest.approx(list(er.irr_roots))
    assert econ["payback_years"] == pytest.approx(er.payback_period)

    # 成本分项与摘要口径一致：分项合计 = 初始投资 + 期末退役费用
    breakdown = econ["cost_breakdown_wanyuan"]
    assert breakdown == pytest.approx(er.cost_breakdown)
    assert "退役拆除(期末)" in breakdown
    assert breakdown["退役拆除(期末)"] == pytest.approx(er.decommissioning_cost)
    assert sum(breakdown.values()) == pytest.approx(
        er.total_capital_cost + er.decommissioning_cost
    )

    # 退役费用确实进入造价模型声明的现金流
    assert er.decommissioning_cost > 0


def test_parity_scenario_real_rate_zero(tmp_path):
    """平价情景：实际折现率为 0 时分析不再中止，且结果有限。"""
    cli = run_cli(tmp_path, discount_rate=0.025, inflation_rate=0.025)
    er = cli.economic_result

    assert er.real_discount_rate == pytest.approx(0.0)
    assert er.npv is not None
    assert er.lcoe is not None

    with open(tmp_path / "results.json", "r", encoding="utf-8") as f:
        econ = json.load(f)["economic"]
    assert econ["real_discount_rate"] == pytest.approx(0.0)
    assert econ["npv_yiyuan"] == pytest.approx(er.npv / 1e4)


def test_sweep_does_not_corrupt_pipeline(tmp_path):
    """台数扫描后流程不中止，结果文件（含经济性与扫描数据）正常写出。"""
    config = create_sample_config()
    config.visualization.save_dir = str(tmp_path)
    config.visualization.save_plots = False

    cli = WindFarmOptimizerCLI(config)
    cli._min_turbines = 6
    cli._max_turbines = 8
    cli.run_full_analysis(
        run_baseline=True,
        run_opt=False,
        run_econ=True,
        run_sweep=True,
        run_viz=True,   # 扫描后可视化不得再因状态污染而崩溃
        save=True,
    )

    # 扫描后风机台数相关状态已恢复
    assert len(cli.turbines) == config.n_turbines
    assert len(cli.rotor_diameters) == config.n_turbines

    with open(tmp_path / "results.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "economic" in data
    assert "turbine_sweep" in data
    sweep = data["turbine_sweep"]
    assert len(sweep["n_turbines"]) == len(sweep["lcoe_yuan_per_kwh"]) > 0
    # 扫描曲线中的 LCOE 均为有限值（该场景发电量均有效）
    assert all(v == v and v != float("inf") for v in sweep["lcoe_yuan_per_kwh"])
