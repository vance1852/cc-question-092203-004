"""经济计算模块测试。

覆盖寿命期现金流重整后的全部关键语义：

* 实际折现率为零（平价情景）或接近零时 NPV/LCOE 连续、不除零；
* IRR 允许为负，临界零根、无根、多根、无投资的返回语义明确；
* 退役费用在寿命期末一次性进入 NPV 与 LCOE，口径一致；
* 无有效发电量时 LCOE 为 inf；
* 盈利、亏损、临界三类典型场景结果可复核；
* 摘要、成本分项、逐年现金流与结果文件使用同一组数值。
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from wind_farm_opt.config import WindFarmConfig
from wind_farm_opt.economy.costs import (
    EconomicAnalyzer,
    FarmCostModel,
    LifecycleEconomics,
    TurbineCostModel,
    annuity_factor,
    discount_factor,
    get_default_farm_cost,
    get_default_turbine_cost,
    real_discount_rate,
    solve_irr,
)

TURBINE = "V126-3.45MW"
N_TURBINES = 12
RATED_MW = 3.45
LIFETIME = 25.0
CAPACITY_MW = N_TURBINES * RATED_MW  # 41.4 MW


def make_analyzer(nominal: float = 0.06, inflation: float = 0.025,
                  price: float = 0.45) -> EconomicAnalyzer:
    return EconomicAnalyzer(
        get_default_turbine_cost(TURBINE),
        FarmCostModel(discount_rate=nominal, inflation_rate=inflation),
        electricity_price=price,
    )


class TestDiscountingContinuity(unittest.TestCase):
    """年金/折现因子及指标在实际折现率为零处的连续性。"""

    def test_real_rate_fisher(self):
        self.assertAlmostEqual(real_discount_rate(0.025, 0.025), 0.0)
        self.assertAlmostEqual(
            real_discount_rate(0.06, 0.025), 1.06 / 1.025 - 1.0, places=12
        )

    def test_annuity_factor_continuous_at_zero(self):
        for n in (1.0, 10.0, 25.0):
            self.assertAlmostEqual(annuity_factor(0.0, n), n, places=12)
            for r in (-1e-12, -1e-9, 1e-9, 1e-12):
                # 数值稳定的参考形式（避免直接公式在小 r 处相消）
                ref = -math.expm1(-n * math.log1p(r)) / r
                self.assertAlmostEqual(annuity_factor(r, n), ref, places=8)
        with self.assertRaises(ValueError):
            annuity_factor(0.0, 0.0)

    def test_discount_factor_continuous_at_zero(self):
        for r in (-1e-12, 0.0, 1e-12):
            self.assertAlmostEqual(discount_factor(r, 25.0), (1.0 + r) ** -25.0,
                                   places=10)

    def test_parity_npv_is_undiscounted_sum(self):
        a = make_analyzer(nominal=0.025, inflation=0.025)
        npv = a.compute_npv(30000.0, 2000.0, 500.0,
                            decommissioning_cost=414.0)
        # 实际折现率 0：NPV = -I + (Rev-OM)*n - D
        self.assertAlmostEqual(npv, -30000.0 + 1500.0 * LIFETIME - 414.0,
                               places=8)

    def test_parity_lcoe_finite(self):
        a = make_analyzer(nominal=0.025, inflation=0.025)
        lcoe = a.compute_lcoe(30000.0, 500.0, 100.0,
                              decommissioning_cost=414.0)
        expected = (30000.0 + 500.0 * LIFETIME + 414.0) * 1e4 / (100e6 * LIFETIME)
        self.assertTrue(math.isfinite(lcoe))
        self.assertAlmostEqual(lcoe, expected, places=10)

    def test_near_zero_matches_zero(self):
        """零点泰勒展开与普通公式在阈值两侧给出一致结果。"""
        a0 = make_analyzer(nominal=0.025, inflation=0.025)
        a_near = make_analyzer(nominal=0.025 + 1e-9, inflation=0.025)
        kw = dict(total_capital_cost=30000.0, annual_revenue=2000.0,
                  annual_om_cost=500.0, decommissioning_cost=414.0)
        self.assertAlmostEqual(
            a0.compute_npv(**kw), a_near.compute_npv(**kw), places=2
        )

    def test_no_crash_for_negative_real_rate(self):
        """名义折现率低于通胀率（负实际利率）也应正常计算。"""
        a = make_analyzer(nominal=0.01, inflation=0.03)
        npv = a.compute_npv(30000.0, 2000.0, 500.0, decommissioning_cost=414.0)
        lcoe = a.compute_lcoe(30000.0, 500.0, 100.0, decommissioning_cost=414.0)
        self.assertTrue(math.isfinite(npv))
        self.assertTrue(math.isfinite(lcoe))


class TestDecommissioning(unittest.TestCase):
    """退役费用必须进入寿命期末现金流，并以一致口径影响 NPV/LCOE。"""

    def setUp(self):
        self.a = make_analyzer()
        self.capital, _ = self.a.compute_capital_cost(N_TURBINES, RATED_MW)
        self.om = self.a.compute_annual_om_cost(N_TURBINES, RATED_MW)
        self.decom = self.a.compute_decommissioning_cost(CAPACITY_MW)

    def test_decommissioning_amount(self):
        # 41.4 MW * 100 万元/MW
        self.assertAlmostEqual(self.decom, CAPACITY_MW * 100.0, places=8)
        self.assertGreater(self.decom, 0.0)

    def test_decommissioning_reduces_npv_by_discounted_amount(self):
        npv_no = self.a.compute_npv(self.capital, 5000.0, self.om,
                                    decommissioning_cost=0.0)
        npv_yes = self.a.compute_npv(self.capital, 5000.0, self.om,
                                     decommissioning_cost=self.decom)
        r = real_discount_rate(0.06, 0.025)
        self.assertAlmostEqual(npv_no - npv_yes,
                               self.decom * discount_factor(r, LIFETIME),
                               places=6)

    def test_decommissioning_increases_lcoe(self):
        lcoe_no = self.a.compute_lcoe(self.capital, self.om, 150.0,
                                      decommissioning_cost=0.0)
        lcoe_yes = self.a.compute_lcoe(self.capital, self.om, 150.0,
                                       decommissioning_cost=self.decom)
        r = real_discount_rate(0.06, 0.025)
        energy_pv = 150e6 * annuity_factor(r, LIFETIME)
        self.assertAlmostEqual(
            (lcoe_yes - lcoe_no) * energy_pv / 1e4,
            self.decom * discount_factor(r, LIFETIME),
            places=6,
        )

    def test_schedule_places_decom_at_final_year_only(self):
        revenue = self.a.compute_annual_revenue(150.0)
        lc = self.a.build_lifecycle(
            self.capital, self.om, revenue, 150e6,
            decommissioning_cost=self.decom,
        )
        schedule = self.a.build_cash_flow_schedule(lc)
        n = int(LIFETIME)
        decom = schedule["decommissioning_cost"]
        self.assertEqual(decom[n], self.decom)
        self.assertEqual(sum(decom[:n]), 0.0)
        # 逐年折现净流之和即 NPV（同一口径）
        self.assertAlmostEqual(sum(schedule["net_present_value"]), lc.npv,
                               places=6)
        # 资本开支仅在 t=0
        self.assertEqual(schedule["capital_cost"][0], self.capital)
        self.assertEqual(sum(schedule["capital_cost"][1:]), 0.0)

    def test_lifecycle_cost_identity(self):
        """寿命期成本现值 = CapEx + 运维现值 + 退役现值。"""
        lc: LifecycleEconomics = self.a.build_lifecycle(
            self.capital, self.om, 5000.0, 150e6,
            decommissioning_cost=self.decom,
        )
        self.assertAlmostEqual(
            lc.lifecycle_cost_pv,
            lc.capital_cost + lc.om_pv + lc.decommissioning_pv,
            places=6,
        )
        self.assertAlmostEqual(
            lc.npv,
            lc.revenue_pv - lc.om_pv - lc.capital_cost - lc.decommissioning_pv,
            places=6,
        )


class TestIRR(unittest.TestCase):
    """允许为负的 IRR 求解与返回语义。"""

    def test_positive_irr_known_case(self):
        # -1000 + 100*A(r,25)=0，无退役
        res = solve_irr(1000.0, 100.0, 25.0)
        self.assertEqual(res.status, "unique")
        r = res.irr_pct / 100.0
        self.assertGreater(r, 0.0)
        self.assertAlmostEqual(
            -1000.0 + 100.0 * annuity_factor(r, 25.0), 0.0, places=6
        )

    def test_negative_irr_is_found(self):
        # 未折现总和也收不回投资：根存在于 (-1, 0)
        res = solve_irr(12000.0, 400.0, 25.0)
        self.assertEqual(res.status, "unique")
        self.assertIsNotNone(res.irr_pct)
        self.assertLess(res.irr_pct, 0.0)
        r = res.irr_pct / 100.0
        self.assertAlmostEqual(
            -12000.0 + 400.0 * annuity_factor(r, 25.0), 0.0, places=6
        )

    def test_deeply_negative_irr(self):
        res = solve_irr(1e9, 400.0, 25.0)
        self.assertEqual(res.status, "unique")
        self.assertLess(res.irr_pct, -40.0)
        self.assertGreater(res.irr_pct, -100.0)

    def test_zero_irr_boundary(self):
        # 10000 = 400*25 恰好在 r=0 处为零
        res = solve_irr(10000.0, 400.0, 25.0)
        self.assertEqual(res.status, "unique")
        self.assertAlmostEqual(res.irr_pct, 0.0, places=8)

    def test_no_root_when_never_recovers(self):
        res = solve_irr(1000.0, -10.0, 25.0)
        self.assertEqual(res.status, "no_root")
        self.assertIsNone(res.irr_pct)
        self.assertEqual(res.all_irr_pct, [])

    def test_no_investment(self):
        res = solve_irr(0.0, 100.0, 25.0)
        self.assertEqual(res.status, "no_investment")
        self.assertIsNone(res.irr_pct)

    def test_multiple_roots_from_large_decommissioning(self):
        # I=1000, c=150, n=20, D=1500：r=0 处 NPV=500>0，两端为负
        res = solve_irr(1000.0, 150.0, 20.0, decommissioning_cost=1500.0)
        self.assertEqual(res.status, "multiple")
        self.assertIsNone(res.irr_pct)
        self.assertEqual(len(res.all_irr_pct), 2)
        self.assertLess(res.all_irr_pct[0], 0.0)
        self.assertGreater(res.all_irr_pct[1], 0.0)
        for p in res.all_irr_pct:
            r = p / 100.0
            residual = (
                -1000.0
                + 150.0 * annuity_factor(r, 20.0)
                - 1500.0 * discount_factor(r, 20.0)
            )
            self.assertAlmostEqual(residual, 0.0, places=5)

    def test_nominal_irr_reported_for_unique_root(self):
        a = make_analyzer(nominal=0.06, inflation=0.025)
        res = a.compute_irr(1000.0, 100.0, 0.0)
        self.assertEqual(res.status, "unique")
        # 名义 = (1+实际)(1+通胀) - 1
        self.assertAlmostEqual(
            res.irr_nominal_pct,
            ((1 + res.irr_pct / 100.0) * 1.025 - 1.0) * 100.0,
            places=8,
        )


class TestLCOEAndScenarios(unittest.TestCase):
    """度电成本语义与盈利/亏损/临界场景。"""

    def setUp(self):
        self.a = make_analyzer()
        self.capital, breakdown = self.a.compute_capital_cost(N_TURBINES, RATED_MW)
        self.breakdown = breakdown
        self.om = self.a.compute_annual_om_cost(N_TURBINES, RATED_MW)
        self.decom = self.a.compute_decommissioning_cost(CAPACITY_MW)

    def test_no_energy_returns_inf(self):
        for aep in (0.0, -10.0):
            lcoe = self.a.compute_lcoe(self.capital, self.om, aep,
                                       decommissioning_cost=self.decom)
            self.assertEqual(lcoe, float(np.inf))
            res = self.a.analyze(N_TURBINES, RATED_MW, aep)
            self.assertEqual(res.lcoe, float(np.inf))
            self.assertEqual(res.lcoe_status, "no_energy")
            # 无发电量时 NPV 仍是有限的（只有成本流出）
            self.assertTrue(math.isfinite(res.npv))
            self.assertLess(res.npv, 0.0)

    def test_profitable_scenario(self):
        """典型盈利项目：NPV>0、IRR>实际折现率、LCOE<电价、可回收。"""
        res = self.a.analyze(N_TURBINES, RATED_MW, 200.0)
        self.assertEqual(res.lcoe_status, "ok")
        self.assertGreater(res.npv, 0.0)
        self.assertEqual(res.irr_result.status, "unique")
        self.assertIsNotNone(res.irr)
        self.assertGreater(res.irr, 3.4146)  # 实际折现率
        self.assertLess(res.lcoe, 0.45)
        self.assertIsNotNone(res.payback_period)
        self.assertLess(res.payback_period, LIFETIME)

    def test_loss_scenario_negative_irr(self):
        """典型亏损项目（发电量极低、退役占比大）：NPV<0，IRR 为负或多根。"""
        res = self.a.analyze(N_TURBINES, RATED_MW, 20.0)
        self.assertLess(res.npv, 0.0)
        self.assertGreater(res.lcoe, 0.45)
        self.assertIsNone(res.payback_period)
        roots = res.irr_result.all_irr_pct
        self.assertTrue(all(p < 0.0 for p in roots))
        for p in roots:
            r = p / 100.0
            revenue = self.a.compute_annual_revenue(20.0)
            residual = (
                -self.capital
                + (revenue - self.om) * annuity_factor(r, LIFETIME)
                - self.decom * discount_factor(r, LIFETIME)
            )
            self.assertAlmostEqual(residual, 0.0, places=4)

    def test_critical_scenario_lcoe_equals_price(self):
        """临界项目：电价=LCOE 时 NPV=0，高 IRR 根恰为实际折现率。"""
        r_real = real_discount_rate(0.06, 0.025)
        af = annuity_factor(r_real, LIFETIME)
        df = discount_factor(r_real, LIFETIME)
        annual_net_break = (self.capital + self.decom * df) / af
        aep = (annual_net_break + self.om) * 1e4 / (0.45 * 1e6)

        res = self.a.analyze(N_TURBINES, RATED_MW, aep)
        self.assertAlmostEqual(res.lcoe, 0.45, places=9)
        self.assertAlmostEqual(res.npv, 0.0, places=4)
        # 最高根必为实际折现率（含退役费时另有一个深负根）
        self.assertAlmostEqual(max(res.irr_result.all_irr_pct),
                               r_real * 100.0, places=5)

    def test_breakdown_sums_to_capital(self):
        self.assertAlmostEqual(
            sum(self.breakdown.values()), self.capital, places=6
        )

    def test_result_uses_single_set_of_numbers(self):
        """analyze 返回值内部自洽：分项、现值、LCOE、NPV 同源。"""
        res = self.a.analyze(N_TURBINES, RATED_MW, 150.0)
        self.assertAlmostEqual(
            sum(res.cost_breakdown.values()), res.total_capital_cost, places=6
        )
        # LCOE * 折现电量 == 寿命期成本现值（单位换算）
        energy_pv = res.net_aep * 1e6 * annuity_factor(
            real_discount_rate(0.06, 0.025), LIFETIME
        )
        self.assertAlmostEqual(
            res.lcoe * energy_pv / 1e4, res.lifecycle_cost_pv, places=4
        )
        # NPV = 收益现值 - 寿命期成本现值
        revenue_pv = res.annual_revenue * annuity_factor(
            real_discount_rate(0.06, 0.025), LIFETIME
        )
        self.assertAlmostEqual(
            res.npv, revenue_pv - res.lifecycle_cost_pv, places=4
        )
        # 退役现值介于 0 与名义额之间（正实际折现率）
        self.assertGreater(res.decommissioning_cost_pv, 0.0)
        self.assertLess(res.decommissioning_cost_pv, res.decommissioning_cost)


class TestConfigRoundTrip(unittest.TestCase):
    def test_inflation_rate_round_trip(self):
        cfg = WindFarmConfig()
        cfg.economic.inflation_rate = 0.03
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.json")
            cfg.to_json(path)
            loaded = WindFarmConfig.from_json(path)
        self.assertEqual(loaded.economic.inflation_rate, 0.03)
        self.assertEqual(loaded.economic.discount_rate, 0.06)


class TestCLIResultsFile(unittest.TestCase):
    """端到端：平价情景下摘要数值与 results.json 一致且不除零。"""

    def test_parity_run_writes_consistent_json(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable, "-m", "wind_farm_opt",
                "--no-optimization", "--no-plots",
                "--discount-rate", "0.025",
                "--inflation-rate", "0.025",
                "--output-dir", tmp,
            ]
            proc = subprocess.run(
                cmd, cwd=repo_root, capture_output=True, text=True, timeout=300
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            with open(os.path.join(tmp, "results.json"), encoding="utf-8") as f:
                data = json.load(f)

        econ = data["economic"]
        self.assertAlmostEqual(econ["discount_rate_real"], 0.0, places=12)
        self.assertTrue(math.isfinite(econ["lcoe_yuan_per_kwh"]))
        self.assertEqual(econ["lcoe_status"], "ok")
        self.assertIsNotNone(econ["npv_wanyuan"])
        self.assertGreater(econ["decommissioning_cost_wanyuan"], 0.0)
        self.assertGreater(econ["decommissioning_cost_pv_wanyuan"], 0.0)
        # 平价下退役现值等于名义额
        self.assertAlmostEqual(
            econ["decommissioning_cost_pv_wanyuan"],
            econ["decommissioning_cost_wanyuan"],
            places=6,
        )
        # 成本分项之和与总投资一致（结果文件内部自洽）
        self.assertAlmostEqual(
            sum(econ["cost_breakdown_wanyuan"].values()),
            econ["total_capital_cost_wanyuan"],
            places=4,
        )
        self.assertIn(econ["irr_status"],
                      {"unique", "multiple", "no_root", "no_investment"})


if __name__ == "__main__":
    unittest.main()
