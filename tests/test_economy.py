"""经济性计算测试。

覆盖：折现工具在零利率处的连续性、平价情景（名义折现率=通胀率）、
盈利/亏损/临界三类典型场景、负IRR求解、多根与无根语义、
退役费用进入现金流的一致口径、无有效发电量时的返回语义，
以及分析摘要与结果文件使用同一组数值。
"""

import json
import math

import numpy as np
import pytest

from wind_farm_opt.economy.costs import (
    EconomicAnalyzer,
    EconomicResult,
    FarmCostModel,
    LifetimeCashFlow,
    annuity_factor,
    discount_factor,
    get_default_farm_cost,
    get_default_turbine_cost,
    solve_irr,
)

# 测试基准: 12 台 V126-3.45MW
N_TURBINES = 12
RATED_MW = 3.45
CAPACITY_MW = N_TURBINES * RATED_MW  # 41.4 MW
AEP_GWH = 150.0
LIFETIME = 25.0

# 由造价模型推出的常量（万元）
CAPITAL = CAPACITY_MW * (580.0 + 70.0 + 300.0) + 5000.0 + 2000.0  # 46330
OM_ANNUAL = CAPACITY_MW * 15.0  # 621
DECOM = CAPACITY_MW * 100.0  # 4140


def make_analyzer(price=0.45, discount=0.06, inflation=0.025, decom_per_mw=100.0):
    turbine_cost = get_default_turbine_cost("V126-3.45MW")
    farm_cost = FarmCostModel(
        discount_rate=discount,
        inflation_rate=inflation,
        decommissioning_cost_per_MW=decom_per_mw,
    )
    return EconomicAnalyzer(turbine_cost, farm_cost, electricity_price=price)


def naive_annuity(rate, n):
    """朴素年金公式（不防除零），用于交叉验证。"""
    return (1.0 - (1.0 + rate) ** (-n)) / rate


# ---------------------------------------------------------------------------
# 折现工具：零利率处的连续性
# ---------------------------------------------------------------------------

class TestDiscountingContinuity:
    def test_annuity_at_zero_rate_equals_periods(self):
        assert annuity_factor(0.0, LIFETIME) == pytest.approx(LIFETIME)
        assert discount_factor(0.0, LIFETIME) == pytest.approx(1.0)

    def test_annuity_continuous_around_zero(self):
        # r -> 0 时年金系数趋于年数，偏差本身为 O(r)
        for r in (1e-9, -1e-9, 1e-12, -1e-12, 0.0):
            assert annuity_factor(r, LIFETIME) == pytest.approx(LIFETIME, rel=1e-6)
        # 级数分支（|r| < 1e-8）等于解析级数值
        assert annuity_factor(1e-9, LIFETIME) == pytest.approx(
            LIFETIME * (1.0 - 0.5e-9 * (LIFETIME + 1.0)), rel=1e-12
        )
        # 通项分支在稍大利率处与朴素公式一致
        # （朴素公式在 |r| ~ 1e-9 时自身已因相消失准，不在此比较）
        for r in (1e-7, 1e-6, 1e-5):
            assert annuity_factor(r, LIFETIME) == pytest.approx(
                naive_annuity(r, LIFETIME), rel=1e-7
            )

    def test_annuity_matches_naive_formula(self):
        for r in (0.05, -0.05, 0.5, -0.5, 0.03414634146341464):
            assert annuity_factor(r, LIFETIME) == pytest.approx(
                naive_annuity(r, LIFETIME), rel=1e-12
            )
            assert discount_factor(r, LIFETIME) == pytest.approx((1 + r) ** (-LIFETIME))

    def test_annuity_negative_rate_exceeds_periods(self):
        # 负利率下年金系数大于年数（旧实现错误地恒等于年数）
        assert annuity_factor(-0.05, LIFETIME) > LIFETIME

    def test_annuity_accepts_arrays(self):
        rates = np.array([-0.05, 0.0, 0.05])
        out = annuity_factor(rates, LIFETIME)
        assert isinstance(out, np.ndarray)
        assert out[1] == pytest.approx(LIFETIME)
        assert out[0] == pytest.approx(naive_annuity(-0.05, LIFETIME))

    def test_zero_periods(self):
        assert annuity_factor(0.05, 0.0) == 0.0


# ---------------------------------------------------------------------------
# 平价情景：名义折现率 == 通胀率（实际折现率为 0）
# ---------------------------------------------------------------------------

class TestParityScenario:
    def test_no_crash_and_closed_form_values(self):
        analyzer = make_analyzer(discount=0.025, inflation=0.025)
        assert analyzer.real_discount_rate == pytest.approx(0.0)

        result = analyzer.analyze(N_TURBINES, RATED_MW, AEP_GWH)

        revenue = AEP_GWH * 1e6 * 0.45 / 1e4
        net_annual = revenue - OM_ANNUAL
        # r_real = 0 时: NPV = -C0 + A*L - D
        assert result.npv == pytest.approx(-CAPITAL + net_annual * LIFETIME - DECOM)
        # LCOE = (C0 + OM*L + D) / (E*L)
        expected_lcoe = (CAPITAL + OM_ANNUAL * LIFETIME + DECOM) * 1e4 / (AEP_GWH * 1e6 * LIFETIME)
        assert result.lcoe == pytest.approx(expected_lcoe)
        assert math.isfinite(result.npv) and math.isfinite(result.lcoe)

    def test_continuous_around_parity(self):
        """实际折现率在 0 附近扰动时，NPV/LCOE 保持连续。"""
        base = make_analyzer(discount=0.025, inflation=0.025).analyze(N_TURBINES, RATED_MW, AEP_GWH)
        for delta in (1e-6, -1e-6, 1e-9, -1e-9):
            nearby = make_analyzer(discount=0.025 + delta, inflation=0.025).analyze(
                N_TURBINES, RATED_MW, AEP_GWH
            )
            assert nearby.npv == pytest.approx(base.npv, rel=1e-4)
            assert nearby.lcoe == pytest.approx(base.lcoe, rel=1e-4)


# ---------------------------------------------------------------------------
# 典型盈利情景
# ---------------------------------------------------------------------------

class TestProfitableScenario:
    def test_metrics_and_root(self):
        analyzer = make_analyzer(price=0.45)
        result = analyzer.analyze(N_TURBINES, RATED_MW, AEP_GWH)

        assert result.npv > 0
        assert result.lcoe < 0.45
        assert result.irr_status == "unique"
        assert result.irr > result.real_discount_rate * 100
        assert result.payback_period == pytest.approx(CAPITAL / (result.annual_revenue - OM_ANNUAL))

        # 根处 NPV 必须为 0（含退役费用的同一现金流）
        cash_flow = analyzer.build_cash_flow(
            CAPITAL, result.annual_revenue, OM_ANNUAL, DECOM
        )
        assert cash_flow.npv(result.irr / 100.0) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 典型亏损情景：负内部收益率必须被求出，而非返回无解
# ---------------------------------------------------------------------------

class TestLossScenario:
    def test_negative_irr_unique(self):
        # 退役费置零 → 常规现金流，存在唯一负 IRR
        analyzer = make_analyzer(price=0.15, decom_per_mw=0.0)
        result = analyzer.analyze(N_TURBINES, RATED_MW, AEP_GWH)

        assert result.npv < 0
        assert result.irr_status == "unique"
        assert result.irr is not None and result.irr < 0.0

        cash_flow = analyzer.build_cash_flow(CAPITAL, result.annual_revenue, OM_ANNUAL, 0.0)
        assert cash_flow.npv(result.irr / 100.0) == pytest.approx(0.0, abs=1e-6)

    def test_negative_irr_with_decommissioning_multiple_roots(self):
        # 低电价 + 期末大额退役支出 → 现金流两次变号，存在两个负根
        analyzer = make_analyzer(price=0.15)
        result = analyzer.analyze(N_TURBINES, RATED_MW, AEP_GWH)

        assert result.irr_status == "multiple"
        assert len(result.irr_roots) == 2
        assert all(r < 0 for r in result.irr_roots)
        # 选定根为绝对值最小者
        assert result.irr == pytest.approx(
            min(result.irr_roots, key=abs), rel=1e-12
        )
        cash_flow = analyzer.build_cash_flow(CAPITAL, result.annual_revenue, OM_ANNUAL, DECOM)
        for r in result.irr_roots:
            assert cash_flow.npv(r / 100.0) == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 临界情景：电价恰使 NPV = 0 → IRR == 实际折现率，LCOE == 电价
# ---------------------------------------------------------------------------

class TestBreakEvenScenario:
    def test_break_even_price(self):
        analyzer = make_analyzer()
        r_real = analyzer.real_discount_rate
        ann = naive_annuity(r_real, LIFETIME)
        disc = (1 + r_real) ** (-LIFETIME)

        net_required = (CAPITAL + DECOM * disc) / ann
        revenue_required = net_required + OM_ANNUAL
        price_star = revenue_required * 1e4 / (AEP_GWH * 1e6)

        result = make_analyzer(price=price_star).analyze(N_TURBINES, RATED_MW, AEP_GWH)

        assert result.npv == pytest.approx(0.0, abs=1e-6)
        assert result.irr == pytest.approx(r_real * 100.0, abs=1e-3)
        assert result.lcoe == pytest.approx(price_star, rel=1e-9)


# ---------------------------------------------------------------------------
# 退役费用：期末一次性支出，按一致口径进入 NPV 与 LCOE
# ---------------------------------------------------------------------------

class TestDecommissioning:
    def test_decommissioning_cost_value(self):
        analyzer = make_analyzer()
        assert analyzer.compute_decommissioning_cost(N_TURBINES, RATED_MW) == pytest.approx(DECOM)

    def test_breakdown_contains_decommissioning_and_sums(self):
        result = make_analyzer().analyze(N_TURBINES, RATED_MW, AEP_GWH)
        assert result.cost_breakdown["退役拆除(期末)"] == pytest.approx(DECOM)
        assert sum(result.cost_breakdown.values()) == pytest.approx(
            result.total_capital_cost + result.decommissioning_cost
        )

    def test_npv_includes_discounted_decommissioning(self):
        with_d = make_analyzer(decom_per_mw=100.0).analyze(N_TURBINES, RATED_MW, AEP_GWH)
        without_d = make_analyzer(decom_per_mw=0.0).analyze(N_TURBINES, RATED_MW, AEP_GWH)

        r_real = with_d.real_discount_rate
        disc = (1 + r_real) ** (-LIFETIME)
        assert with_d.npv == pytest.approx(without_d.npv - DECOM * disc, rel=1e-12)

    def test_lcoe_includes_discounted_decommissioning(self):
        with_d = make_analyzer(decom_per_mw=100.0).analyze(N_TURBINES, RATED_MW, AEP_GWH)
        without_d = make_analyzer(decom_per_mw=0.0).analyze(N_TURBINES, RATED_MW, AEP_GWH)

        r_real = with_d.real_discount_rate
        ann = naive_annuity(r_real, LIFETIME)
        disc = (1 + r_real) ** (-LIFETIME)
        expected_delta = DECOM * disc * 1e4 / (AEP_GWH * 1e6 * ann)
        assert with_d.lcoe - without_d.lcoe == pytest.approx(expected_delta, rel=1e-9)

    def test_irr_solved_on_cash_flow_with_decommissioning(self):
        analyzer = make_analyzer()
        result = analyzer.analyze(N_TURBINES, RATED_MW, AEP_GWH)
        cash_flow = analyzer.build_cash_flow(
            CAPITAL, result.annual_revenue, OM_ANNUAL, result.decommissioning_cost
        )
        assert cash_flow.npv(result.irr / 100.0) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 返回语义：无根 / 多根 / 无有效发电量
# ---------------------------------------------------------------------------

class TestIRRSemantics:
    def test_no_root_when_all_flows_negative(self):
        cf = LifetimeCashFlow(initial_investment=100.0, annual_net=-5.0, decommissioning=10.0, lifetime=25.0)
        res = solve_irr(cf)
        assert res.status == "no_root" and res.irr is None and res.roots == ()

    def test_no_root_when_all_flows_positive(self):
        cf = LifetimeCashFlow(initial_investment=0.0, annual_net=5.0, decommissioning=0.0, lifetime=25.0)
        res = solve_irr(cf)
        assert res.status == "no_root" and res.irr is None

    def test_no_root_when_all_zero(self):
        cf = LifetimeCashFlow(initial_investment=0.0, annual_net=0.0, decommissioning=0.0, lifetime=25.0)
        res = solve_irr(cf)
        assert res.status == "no_root" and res.irr is None

    def test_unique_root_conventional_cash_flow(self):
        cf = LifetimeCashFlow(initial_investment=100.0, annual_net=10.0, decommissioning=0.0, lifetime=25.0)
        res = solve_irr(cf)
        assert res.status == "unique"
        assert 0.08 < res.irr < 0.09
        assert cf.npv(res.irr) == pytest.approx(0.0, abs=1e-8)

    def test_multiple_roots_picks_smallest_magnitude(self):
        # -100 + 20/年×25 - 300(期末): NPV(0) > 0，两端为负 → 两个根
        cf = LifetimeCashFlow(initial_investment=100.0, annual_net=20.0, decommissioning=300.0, lifetime=25.0)
        res = solve_irr(cf)
        assert res.status == "multiple"
        assert len(res.roots) == 2
        assert res.irr == min(res.roots, key=abs)
        for r in res.roots:
            assert cf.npv(r) == pytest.approx(0.0, abs=1e-8)

    def test_invalid_cash_flow_rejected(self):
        with pytest.raises(ValueError):
            LifetimeCashFlow(initial_investment=100.0, annual_net=1.0, decommissioning=0.0, lifetime=0.0)
        with pytest.raises(ValueError):
            LifetimeCashFlow(initial_investment=-1.0, annual_net=1.0, decommissioning=0.0, lifetime=25.0)


class TestNoValidGeneration:
    def test_zero_aep_semantics(self):
        result = make_analyzer().analyze(N_TURBINES, RATED_MW, 0.0)

        assert result.lcoe is None
        assert result.irr is None
        assert result.irr_status == "no_root"
        assert result.payback_period is None

        # NPV 仍为有限值: -(C0 + OM*ann + D*disc)
        r_real = result.real_discount_rate
        ann = naive_annuity(r_real, LIFETIME)
        disc = (1 + r_real) ** (-LIFETIME)
        assert result.npv == pytest.approx(-(CAPITAL + OM_ANNUAL * ann + DECOM * disc))
        assert math.isfinite(result.npv)

    def test_negative_aep_semantics(self):
        result = make_analyzer().analyze(N_TURBINES, RATED_MW, -10.0)
        assert result.lcoe is None
        assert result.irr is None
        assert result.irr_status == "no_root"

    def test_result_is_json_serializable(self):
        result = make_analyzer().analyze(N_TURBINES, RATED_MW, 0.0)
        payload = {
            "lcoe": result.lcoe,
            "npv": result.npv,
            "irr": result.irr,
            "irr_status": result.irr_status,
        }
        # 不允许出现 Infinity/NaN 等非法 JSON 值
        text = json.dumps(payload, allow_nan=False)
        assert json.loads(text)["lcoe"] is None


# ---------------------------------------------------------------------------
# analyze 结果内部一致性 + 接口兼容性
# ---------------------------------------------------------------------------

class TestResultConsistency:
    def test_fields_mutually_consistent(self):
        result = make_analyzer().analyze(N_TURBINES, RATED_MW, AEP_GWH)

        r_real = result.real_discount_rate
        ann = naive_annuity(r_real, LIFETIME)
        disc = (1 + r_real) ** (-LIFETIME)
        net_annual = result.annual_revenue - result.total_om_cost_annual

        # 用结果字段重算 NPV / LCOE，必须一致
        npv = -result.total_capital_cost + net_annual * ann - result.decommissioning_cost * disc
        assert result.npv == pytest.approx(npv)

        cost_pv = result.total_capital_cost + result.total_om_cost_annual * ann + result.decommissioning_cost * disc
        lcoe = cost_pv * 1e4 / (result.net_aep * 1e6 * ann)
        assert result.lcoe == pytest.approx(lcoe)

        assert result.lifetime == pytest.approx(LIFETIME)
        assert result.real_discount_rate == pytest.approx(1.06 / 1.025 - 1.0)

    def test_backward_compatible_calls(self):
        analyzer = make_analyzer()
        # 旧式调用（不含退役费用关键字）仍可用
        lcoe = analyzer.compute_lcoe(CAPITAL, OM_ANNUAL, AEP_GWH)
        npv = analyzer.compute_npv(CAPITAL, 6750.0, OM_ANNUAL)
        irr = analyzer.compute_irr(CAPITAL, 6750.0, OM_ANNUAL)
        assert lcoe is not None and math.isfinite(lcoe)
        assert math.isfinite(npv)
        assert irr is not None

        # 退役费用作为可选参数传入
        npv_d = analyzer.compute_npv(CAPITAL, 6750.0, OM_ANNUAL, decommissioning_cost=DECOM)
        assert npv_d < npv

    def test_result_defaults_keep_old_construction(self):
        res = EconomicResult(
            total_installed_capacity=41.4,
            net_aep=150.0,
            annual_revenue=6750.0,
            lcoe=0.23,
            total_capital_cost=CAPITAL,
            total_om_cost_annual=OM_ANNUAL,
            npv=1.0,
            irr=12.0,
            payback_period=7.0,
            cost_breakdown={},
        )
        assert res.decommissioning_cost == 0.0
        assert res.irr_status == "no_root"

    def test_invalid_rates_rejected(self):
        turbine_cost = get_default_turbine_cost("V126-3.45MW")
        with pytest.raises(ValueError):
            EconomicAnalyzer(turbine_cost, FarmCostModel(discount_rate=-1.0))
        with pytest.raises(ValueError):
            EconomicAnalyzer(turbine_cost, FarmCostModel(inflation_rate=-1.5))
