"""风电场经济性评估。

寿命期现金流与度电成本(LCOE)模型。

口径约定（NPV、LCOE、IRR 共用同一组现金流与折现假设）：

- 现金流按不变价（实际）口径计列：初始投资发生在 t=0；
  年净现金流（年发电收益 - 年运维费用）发生在 t=1..lifetime；
  退役拆除费用作为一次性支出发生在寿命期末 t=lifetime。
- 折现采用实际折现率 ``r_real = (1 + 名义折现率) / (1 + 通货膨胀率) - 1``。
  当名义折现率与通胀率相等（平价情景）时 r_real = 0，年金系数退化为
  年数本身，计算在 r_real -> 0 处保持连续，不会发生除零。
- 内部收益率(IRR)在 r > -1 域内对同一寿命期现金流求解 NPV(r) = 0，
  允许负收益率；无根、多根时的返回语义见 :func:`solve_irr` 与
  :class:`IRRResult`。
- 无有效发电量（净AEP <= 0）时：LCOE 无定义，返回 None；NPV 仍为
  有限值；现金流无符号变化，IRR 状态为 "no_root"。
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

__all__ = [
    "TurbineCostModel",
    "FarmCostModel",
    "EconomicResult",
    "LifetimeCashFlow",
    "IRRResult",
    "EconomicAnalyzer",
    "annuity_factor",
    "discount_factor",
    "solve_irr",
    "get_default_turbine_cost",
    "get_default_farm_cost",
]


# ---------------------------------------------------------------------------
# 数值稳定的折现工具
# ---------------------------------------------------------------------------

def annuity_factor(
    rate: Union[float, np.ndarray],
    n_periods: float,
) -> Union[float, np.ndarray]:
    """年金现值系数 ``(1 - (1+r)^-n) / r``。

    在 ``r = 0`` 处连续（极限为 ``n_periods``），并支持负利率
    （``r > -1``）。使用 ``log1p``/``expm1`` 保证小利率下的精度。

    Parameters
    ----------
    rate : float 或 np.ndarray
        每期折现率（必须 > -1）
    n_periods : float
        期数（年），<= 0 时返回 0

    Returns
    -------
    float 或 np.ndarray
        年金现值系数
    """
    r = np.asarray(rate, dtype=float)
    if n_periods <= 0:
        out = np.zeros_like(r)
        return float(out) if np.isscalar(rate) else out
    with np.errstate(divide="ignore", invalid="ignore"):
        general = -np.expm1(-n_periods * np.log1p(r)) / r
    # r -> 0 时的级数展开: n * (1 - r*(n+1)/2 + O(r^2))
    limit = n_periods * (1.0 - 0.5 * r * (n_periods + 1.0))
    out = np.where(np.abs(r) < 1e-8, limit, general)
    return float(out) if np.isscalar(rate) else out


def discount_factor(
    rate: Union[float, np.ndarray],
    n_periods: float,
) -> Union[float, np.ndarray]:
    """一次性期末现金流折现系数 ``(1+r)^-n``，在 ``r = 0`` 处连续。"""
    r = np.asarray(rate, dtype=float)
    out = np.exp(-n_periods * np.log1p(r))
    return float(out) if np.isscalar(rate) else out


# ---------------------------------------------------------------------------
# 造价模型
# ---------------------------------------------------------------------------

@dataclass
class TurbineCostModel:
    """风机造价模型。

    Parameters
    ----------
    turbine_model : str
        风机型号名称
    capital_cost_per_MW : float
        单位容量造价 (万元/MW)
    installation_cost_per_MW : float
        安装费用 (万元/MW)
    o_and_m_cost_per_MW_per_year : float
        年运维费用 (万元/MW/year)
    design_lifetime : float
        设计寿命 (年)
    """

    turbine_model: str
    capital_cost_per_MW: float
    installation_cost_per_MW: float
    o_and_m_cost_per_MW_per_year: float
    design_lifetime: float = 25.0


@dataclass
class FarmCostModel:
    """风电场整体造价模型。

    Parameters
    ----------
    site_development_cost : float
        场地开发费用 (万元)
    grid_connection_cost_per_MW : float
        并网费用 (万元/MW)
    access_road_cost : float
        道路建设费用 (万元)
    decommissioning_cost_per_MW : float
        退役拆除费用 (万元/MW)，在寿命期末一次性支出
    discount_rate : float
        名义折现率 (0-1)
    inflation_rate : float
        通货膨胀率 (0-1)
    """

    site_development_cost: float = 5000.0
    grid_connection_cost_per_MW: float = 300.0
    access_road_cost: float = 2000.0
    decommissioning_cost_per_MW: float = 100.0
    discount_rate: float = 0.06
    inflation_rate: float = 0.025


# ---------------------------------------------------------------------------
# 寿命期现金流与 IRR 求解
# ---------------------------------------------------------------------------

@dataclass
class LifetimeCashFlow:
    """寿命期现金流（不变价口径，单位：万元）。

    Parameters
    ----------
    initial_investment : float
        t=0 初始投资（正数表示支出）
    annual_net : float
        t=1..lifetime 每年净现金流（年收益 - 年运维，可为负）
    decommissioning : float
        t=lifetime 期末一次性退役拆除支出（正数表示支出）
    lifetime : float
        寿命期（年），必须为正
    """

    initial_investment: float
    annual_net: float
    decommissioning: float
    lifetime: float

    def __post_init__(self) -> None:
        if self.lifetime <= 0:
            raise ValueError(f"寿命期必须为正数，当前为 {self.lifetime}")
        if self.initial_investment < 0:
            raise ValueError(f"初始投资不能为负，当前为 {self.initial_investment}")
        if self.decommissioning < 0:
            raise ValueError(f"退役费用不能为负，当前为 {self.decommissioning}")

    def npv(self, rate: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """按折现率 ``rate`` 计算净现值（万元）。

        ``rate > -1``；在 ``rate -> 0`` 处连续。支持标量或数组输入。
        """
        return (
            -self.initial_investment
            + self.annual_net * annuity_factor(rate, self.lifetime)
            - self.decommissioning * discount_factor(rate, self.lifetime)
        )


@dataclass
class IRRResult:
    """IRR 求解结果。

    Attributes
    ----------
    status : str
        求解状态：

        - ``"unique"``   : 存在唯一实根，``irr`` 为该根；
        - ``"multiple"`` : 存在多个实根（现金流多次变号，例如期末退役
          支出较大时），``irr`` 取绝对值最小者，``roots`` 列出全部实根；
        - ``"no_root"``  : 无实根（现金流无符号变化，或 NPV 在求解域内
          恒不为零），``irr`` 为 None。
    irr : Optional[float]
        选定的内部收益率（小数，非百分数）；无根时为 None。
    roots : tuple[float, ...]
        求解域内找到的全部实根（小数，升序）。
    """

    status: str
    irr: Optional[float]
    roots: tuple = ()


# IRR 求解域: r ∈ (_IRR_RATE_MIN, _IRR_RATE_MAX)，即 1+r ∈ [1e-4, 1e8]
_IRR_RATE_MIN = -0.9999
_IRR_RATE_MAX = 1e8 - 1.0
_IRR_GRID_SIZE = 30000


def _bisect_root(func, lo: float, hi: float, max_iter: int = 200) -> float:
    """在 [lo, hi] 内用二分法求 func 的根（端点须异号）。"""
    f_lo = func(lo)
    f_hi = func(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    neg_lo = f_lo < 0.0
    mid = lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if f_mid == 0.0 or (hi - lo) <= 1e-12 * max(1.0, abs(mid)):
            return mid
        if (f_mid < 0.0) == neg_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _dedupe_roots(roots: list, tol: float = 1e-9) -> list:
    """合并数值上重复（相对误差小于 tol）的根。"""
    if not roots:
        return []
    ordered = sorted(roots)
    merged = [ordered[0]]
    for r in ordered[1:]:
        if abs(r - merged[-1]) <= tol * max(1.0, abs(r), abs(merged[-1])):
            merged[-1] = 0.5 * (merged[-1] + r)
        else:
            merged.append(r)
    return merged


def solve_irr(cash_flow: LifetimeCashFlow) -> IRRResult:
    """求解寿命期现金流的内部收益率(IRR)。

    在 r ∈ (-0.9999, 1e8) 内对 NPV(r) 做网格扫描定位符号变化，
    再对每个含根区间二分求精。允许负收益率。

    Parameters
    ----------
    cash_flow : LifetimeCashFlow
        寿命期现金流

    Returns
    -------
    IRRResult
        求解状态与根，见 :class:`IRRResult`。
    """
    if (
        cash_flow.initial_investment == 0.0
        and cash_flow.annual_net == 0.0
        and cash_flow.decommissioning == 0.0
    ):
        # 现金流恒为零，IRR 无定义
        return IRRResult(status="no_root", irr=None, roots=())

    rates = np.geomspace(1.0 + _IRR_RATE_MIN, 1.0 + _IRR_RATE_MAX, _IRR_GRID_SIZE) - 1.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        values = np.asarray(cash_flow.npv(rates), dtype=float)

    roots: list = []
    for k in range(len(rates) - 1):
        v0, v1 = values[k], values[k + 1]
        if np.isnan(v0) or np.isnan(v1):
            continue
        if v0 == 0.0:
            roots.append(float(rates[k]))
        elif (v0 < 0.0) != (v1 < 0.0):
            roots.append(
                _bisect_root(cash_flow.npv, float(rates[k]), float(rates[k + 1]))
            )
    if not np.isnan(values[-1]) and values[-1] == 0.0:
        roots.append(float(rates[-1]))

    roots = _dedupe_roots(roots)
    if not roots:
        return IRRResult(status="no_root", irr=None, roots=())

    chosen = min(roots, key=abs)
    status = "unique" if len(roots) == 1 else "multiple"
    return IRRResult(status=status, irr=chosen, roots=tuple(roots))


# ---------------------------------------------------------------------------
# 评估结果
# ---------------------------------------------------------------------------

@dataclass
class EconomicResult:
    """经济性评估结果。

    Parameters
    ----------
    total_installed_capacity : float
        总装机容量 (MW)
    net_aep : float
        净年发电量 (GWh/year)
    annual_revenue : float
        年收益 (万元/year)
    lcoe : Optional[float]
        度电成本 (元/kWh)；无有效发电量（净AEP <= 0）时为 None
    total_capital_cost : float
        总初始投资 (万元)
    total_om_cost_annual : float
        年运维费用 (万元/year)
    npv : Optional[float]
        净现值 (万元)，按实际折现率折现，含期末退役支出
    irr : Optional[float]
        内部收益率 (%)；无实根时为 None
    payback_period : Optional[float]
        静态投资回收期 (年)，若年净现金流非正则返回 None
    cost_breakdown : dict[str, float]
        成本分项 (万元)，含初始投资分项与期末退役拆除费用
    decommissioning_cost : float
        期末退役拆除费用 (万元)
    real_discount_rate : float
        实际折现率 (小数)
    lifetime : float
        寿命期 (年)
    irr_status : str
        IRR 求解状态: "unique" / "multiple" / "no_root"
    irr_roots : tuple[float, ...]
        全部 IRR 实根 (%)，升序；多根时 irr 取绝对值最小者
    """

    total_installed_capacity: float
    net_aep: float
    annual_revenue: float
    lcoe: Optional[float]
    total_capital_cost: float
    total_om_cost_annual: float
    npv: Optional[float]
    irr: Optional[float]
    payback_period: Optional[float]
    cost_breakdown: dict
    decommissioning_cost: float = 0.0
    real_discount_rate: float = 0.0
    lifetime: float = 0.0
    irr_status: str = "no_root"
    irr_roots: tuple = ()


# ---------------------------------------------------------------------------
# 经济性分析器
# ---------------------------------------------------------------------------

class EconomicAnalyzer:
    """风电场经济性分析器。"""

    def __init__(
        self,
        turbine_cost: TurbineCostModel,
        farm_cost: FarmCostModel,
        electricity_price: float = 0.45,
    ) -> None:
        """
        Parameters
        ----------
        turbine_cost : TurbineCostModel
            风机造价模型
        farm_cost : FarmCostModel
            风电场造价模型
        electricity_price : float
            上网电价 (元/kWh)
        """
        if farm_cost.discount_rate <= -1.0:
            raise ValueError(f"折现率必须大于 -100%，当前为 {farm_cost.discount_rate}")
        if farm_cost.inflation_rate <= -1.0:
            raise ValueError(f"通货膨胀率必须大于 -100%，当前为 {farm_cost.inflation_rate}")
        self.turbine_cost = turbine_cost
        self.farm_cost = farm_cost
        self.electricity_price = electricity_price

    @property
    def real_discount_rate(self) -> float:
        """实际折现率 = (1 + 名义折现率) / (1 + 通货膨胀率) - 1。

        名义折现率等于通胀率（平价情景）时为 0，相关计算在该点连续。
        """
        return (1.0 + self.farm_cost.discount_rate) / (1.0 + self.farm_cost.inflation_rate) - 1.0

    def compute_capital_cost(
        self,
        n_turbines: int,
        rated_power_per_turbine_MW: float,
    ) -> tuple:
        """计算初始投资。

        Parameters
        ----------
        n_turbines : int
            风机台数
        rated_power_per_turbine_MW : float
            单台风机额定功率 (MW)

        Returns
        -------
        tuple[float, dict[str, float]]
            - 总初始投资 (万元)
            - 成本分项明细
        """
        total_capacity = n_turbines * rated_power_per_turbine_MW

        turbine_capital = n_turbines * rated_power_per_turbine_MW * self.turbine_cost.capital_cost_per_MW
        turbine_installation = n_turbines * rated_power_per_turbine_MW * self.turbine_cost.installation_cost_per_MW
        grid_connection = total_capacity * self.farm_cost.grid_connection_cost_per_MW
        site_dev = self.farm_cost.site_development_cost
        access_road = self.farm_cost.access_road_cost

        total = (
            turbine_capital
            + turbine_installation
            + grid_connection
            + site_dev
            + access_road
        )

        breakdown = {
            "风机设备": turbine_capital,
            "风机安装": turbine_installation,
            "并网工程": grid_connection,
            "场地开发": site_dev,
            "道路建设": access_road,
        }

        return total, breakdown

    def compute_annual_om_cost(
        self,
        n_turbines: int,
        rated_power_per_turbine_MW: float,
    ) -> float:
        """计算年运维费用。

        Parameters
        ----------
        n_turbines : int
            风机台数
        rated_power_per_turbine_MW : float
            单台风机额定功率 (MW)

        Returns
        -------
        float
            年运维费用 (万元/year)
        """
        total_capacity = n_turbines * rated_power_per_turbine_MW
        return total_capacity * self.turbine_cost.o_and_m_cost_per_MW_per_year

    def compute_annual_revenue(self, net_aep_GWh: float) -> float:
        """计算年发电收益。

        Parameters
        ----------
        net_aep_GWh : float
            净年发电量 (GWh/year)

        Returns
        -------
        float
            年收益 (万元/year)
        """
        return net_aep_GWh * 1e6 * self.electricity_price / 1e4

    def compute_decommissioning_cost(
        self,
        n_turbines: int,
        rated_power_per_turbine_MW: float,
    ) -> float:
        """计算退役拆除费用（寿命期末一次性支出）。

        Parameters
        ----------
        n_turbines : int
            风机台数
        rated_power_per_turbine_MW : float
            单台风机额定功率 (MW)

        Returns
        -------
        float
            退役拆除费用 (万元)，发生在 t = 寿命期末
        """
        total_capacity = n_turbines * rated_power_per_turbine_MW
        return total_capacity * self.farm_cost.decommissioning_cost_per_MW

    def build_cash_flow(
        self,
        total_capital_cost: float,
        annual_revenue: float,
        annual_om_cost: float,
        decommissioning_cost: float = 0.0,
        lifetime: Optional[float] = None,
    ) -> LifetimeCashFlow:
        """构建寿命期现金流（NPV/LCOE/IRR 的统一口径）。

        Parameters
        ----------
        total_capital_cost : float
            初始投资 (万元)，t=0 支出
        annual_revenue : float
            年收益 (万元/year)
        annual_om_cost : float
            年运维费用 (万元/year)
        decommissioning_cost : float
            退役拆除费用 (万元)，t=lifetime 期末支出
        lifetime : Optional[float]
            寿命期 (年)，默认使用风机设计寿命

        Returns
        -------
        LifetimeCashFlow
            寿命期现金流
        """
        if lifetime is None:
            lifetime = self.turbine_cost.design_lifetime
        return LifetimeCashFlow(
            initial_investment=float(total_capital_cost),
            annual_net=float(annual_revenue - annual_om_cost),
            decommissioning=float(decommissioning_cost),
            lifetime=float(lifetime),
        )

    def compute_lcoe(
        self,
        total_capital_cost: float,
        total_om_cost_annual: float,
        net_aep_GWh: float,
        lifetime: Optional[float] = None,
        decommissioning_cost: float = 0.0,
    ) -> Optional[float]:
        """计算度电成本(LCOE)。

        LCOE = 总费用现值 / 总发电量现值，其中总费用现值包含初始投资、
        运维费用现值以及期末退役拆除费用现值。

        Parameters
        ----------
        total_capital_cost : float
            初始投资 (万元)
        total_om_cost_annual : float
            年运维费用 (万元/year)
        net_aep_GWh : float
            年净发电量 (GWh/year)
        lifetime : Optional[float]
            寿命期 (年)，默认使用风机设计寿命
        decommissioning_cost : float
            退役拆除费用 (万元)，寿命期末一次性支出

        Returns
        -------
        Optional[float]
            LCOE (元/kWh)；无有效发电量（net_aep_GWh <= 0）时返回 None
        """
        if net_aep_GWh <= 0:
            return None
        if lifetime is None:
            lifetime = self.turbine_cost.design_lifetime

        r_real = self.real_discount_rate
        annuity = annuity_factor(r_real, lifetime)
        decom_discount = discount_factor(r_real, lifetime)

        total_cost_pv = (
            total_capital_cost
            + total_om_cost_annual * annuity
            + decommissioning_cost * decom_discount
        )
        total_energy_pv = net_aep_GWh * 1e6 * annuity

        lcoe_yuan_per_kwh = (total_cost_pv * 1e4) / total_energy_pv

        return float(lcoe_yuan_per_kwh)

    def compute_npv(
        self,
        total_capital_cost: float,
        annual_revenue: float,
        annual_om_cost: float,
        lifetime: Optional[float] = None,
        decommissioning_cost: float = 0.0,
    ) -> float:
        """计算净现值(NPV)，含期末退役拆除费用现值。

        Parameters
        ----------
        total_capital_cost : float
            初始投资 (万元)
        annual_revenue : float
            年收益 (万元/year)
        annual_om_cost : float
            年运维费用 (万元/year)
        lifetime : Optional[float]
            寿命期 (年)
        decommissioning_cost : float
            退役拆除费用 (万元)，寿命期末一次性支出

        Returns
        -------
        float
            NPV (万元)
        """
        cash_flow = self.build_cash_flow(
            total_capital_cost=total_capital_cost,
            annual_revenue=annual_revenue,
            annual_om_cost=annual_om_cost,
            decommissioning_cost=decommissioning_cost,
            lifetime=lifetime,
        )
        return float(cash_flow.npv(self.real_discount_rate))

    def compute_payback_period(
        self,
        total_capital_cost: float,
        annual_revenue: float,
        annual_om_cost: float,
    ) -> Optional[float]:
        """计算静态投资回收期（不折现、不含期末退役费用）。

        Parameters
        ----------
        total_capital_cost : float
            初始投资 (万元)
        annual_revenue : float
            年收益 (万元/year)
        annual_om_cost : float
            年运维费用 (万元/year)

        Returns
        -------
        Optional[float]
            投资回收期 (年)，若年净现金流非正则返回 None
        """
        net_annual = annual_revenue - annual_om_cost
        if net_annual <= 0:
            return None
        return float(total_capital_cost / net_annual)

    def compute_irr_detail(
        self,
        total_capital_cost: float,
        annual_revenue: float,
        annual_om_cost: float,
        lifetime: Optional[float] = None,
        decommissioning_cost: float = 0.0,
    ) -> IRRResult:
        """求解内部收益率(IRR)的完整结果（含求解状态与全部实根）。

        对包含期末退役支出的寿命期现金流求解 NPV(r) = 0，
        允许负收益率。返回语义见 :class:`IRRResult`。
        """
        cash_flow = self.build_cash_flow(
            total_capital_cost=total_capital_cost,
            annual_revenue=annual_revenue,
            annual_om_cost=annual_om_cost,
            decommissioning_cost=decommissioning_cost,
            lifetime=lifetime,
        )
        return solve_irr(cash_flow)

    def compute_irr(
        self,
        total_capital_cost: float,
        annual_revenue: float,
        annual_om_cost: float,
        lifetime: Optional[float] = None,
        decommissioning_cost: float = 0.0,
    ) -> Optional[float]:
        """计算内部收益率(IRR)。

        对包含期末退役支出的寿命期现金流求解 NPV(r) = 0，
        允许负收益率；多根时取绝对值最小者。

        Parameters
        ----------
        total_capital_cost : float
            初始投资 (万元)
        annual_revenue : float
            年收益 (万元/year)
        annual_om_cost : float
            年运维费用 (万元/year)
        lifetime : Optional[float]
            寿命期 (年)
        decommissioning_cost : float
            退役拆除费用 (万元)，寿命期末一次性支出

        Returns
        -------
        Optional[float]
            IRR (%)，若无实根则返回 None
        """
        detail = self.compute_irr_detail(
            total_capital_cost=total_capital_cost,
            annual_revenue=annual_revenue,
            annual_om_cost=annual_om_cost,
            lifetime=lifetime,
            decommissioning_cost=decommissioning_cost,
        )
        return None if detail.irr is None else float(detail.irr * 100.0)

    def analyze(
        self,
        n_turbines: int,
        rated_power_per_turbine_MW: float,
        net_aep_GWh: float,
    ) -> EconomicResult:
        """进行完整的经济性分析。

        所有指标（NPV、LCOE、IRR）基于同一寿命期现金流计算：
        t=0 初始投资，t=1..lifetime 年净现金流，t=lifetime 期末退役支出。

        Parameters
        ----------
        n_turbines : int
            风机台数
        rated_power_per_turbine_MW : float
            单台风机额定功率 (MW)
        net_aep_GWh : float
            净年发电量 (GWh/year)；<= 0 时 LCOE 为 None，IRR 状态为
            "no_root"，NPV 仍为有限值

        Returns
        -------
        EconomicResult
            经济性分析结果
        """
        total_capacity = n_turbines * rated_power_per_turbine_MW

        total_capital_cost, capital_breakdown = self.compute_capital_cost(
            n_turbines, rated_power_per_turbine_MW
        )
        annual_om_cost = self.compute_annual_om_cost(
            n_turbines, rated_power_per_turbine_MW
        )
        annual_revenue = self.compute_annual_revenue(net_aep_GWh)
        decommissioning_cost = self.compute_decommissioning_cost(
            n_turbines, rated_power_per_turbine_MW
        )
        lifetime = float(self.turbine_cost.design_lifetime)

        cash_flow = self.build_cash_flow(
            total_capital_cost=total_capital_cost,
            annual_revenue=annual_revenue,
            annual_om_cost=annual_om_cost,
            decommissioning_cost=decommissioning_cost,
            lifetime=lifetime,
        )

        real_rate = self.real_discount_rate
        npv = float(cash_flow.npv(real_rate))
        lcoe = self.compute_lcoe(
            total_capital_cost,
            annual_om_cost,
            net_aep_GWh,
            lifetime=lifetime,
            decommissioning_cost=decommissioning_cost,
        )
        irr_detail = solve_irr(cash_flow)
        payback = self.compute_payback_period(
            total_capital_cost, annual_revenue, annual_om_cost
        )

        cost_breakdown = dict(capital_breakdown)
        cost_breakdown["退役拆除(期末)"] = float(decommissioning_cost)

        return EconomicResult(
            total_installed_capacity=float(total_capacity),
            net_aep=float(net_aep_GWh),
            annual_revenue=float(annual_revenue),
            lcoe=lcoe,
            total_capital_cost=float(total_capital_cost),
            total_om_cost_annual=float(annual_om_cost),
            npv=npv,
            irr=None if irr_detail.irr is None else float(irr_detail.irr * 100.0),
            payback_period=payback,
            cost_breakdown=cost_breakdown,
            decommissioning_cost=float(decommissioning_cost),
            real_discount_rate=float(real_rate),
            lifetime=lifetime,
            irr_status=irr_detail.status,
            irr_roots=tuple(float(r * 100.0) for r in irr_detail.roots),
        )


def get_default_turbine_cost(model: str = "V164-9.5MW") -> TurbineCostModel:
    """获取默认风机造价模型。

    Parameters
    ----------
    model : str
        风机型号

    Returns
    -------
    TurbineCostModel
        风机造价模型
    """
    if model == "V164-9.5MW":
        return TurbineCostModel(
            turbine_model="V164-9.5MW",
            capital_cost_per_MW=650.0,
            installation_cost_per_MW=80.0,
            o_and_m_cost_per_MW_per_year=18.0,
            design_lifetime=25.0,
        )
    elif model == "V126-3.45MW":
        return TurbineCostModel(
            turbine_model="V126-3.45MW",
            capital_cost_per_MW=580.0,
            installation_cost_per_MW=70.0,
            o_and_m_cost_per_MW_per_year=15.0,
            design_lifetime=25.0,
        )
    else:
        raise ValueError(f"未知的风机型号: {model}")


def get_default_farm_cost() -> FarmCostModel:
    """获取默认风电场造价模型。"""
    return FarmCostModel(
        site_development_cost=5000.0,
        grid_connection_cost_per_MW=300.0,
        access_road_cost=2000.0,
        decommissioning_cost_per_MW=100.0,
        discount_rate=0.06,
        inflation_rate=0.025,
    )
