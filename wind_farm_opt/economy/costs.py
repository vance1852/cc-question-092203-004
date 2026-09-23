"""风电场经济性评估。

寿命期现金流模型（NPV / LCOE / IRR 共用同一口径）
--------------------------------------------------
设寿命期为 ``n`` 年，所有金额均以基年不变价（实际口径）表示：

* ``t = 0``：初始投资（资本开支）流出；
* ``t = 1..n``：每年运维费用流出、发电收益流入，年发电量记为 ``E``；
* ``t = n``  ：退役拆除费用流出（期末一次性发生）。

实际折现率由费雪方程得到::

    r_real = (1 + r_nominal) / (1 + inflation) - 1

当名义折现率等于通胀率时，``r_real = 0``（平价情景）。年金因子与折现
因子在 ``r_real = 0`` 处通过 ``log1p/expm1`` 及泰勒展开保持连续，不
再出现除零。

NPV 与 LCOE 使用完全相同的折现现金流::

    NPV  = -CapEx + (Rev - OM) * A(r, n) - Decom / (1+r)^n
    LCOE = [CapEx + OM*A(r,n) + Decom/(1+r)^n] / [E * A(r, n)]

其中 ``A(r, n)`` 为年金现值因子。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# 视为零实际利率的阈值：|r| 小于该值时使用泰勒展开，保证零点连续
_RATE_EPS = 1e-10

# IRR 数值搜索域下界（r > -1），取 1 + r = 1e-12
_IRR_S_LOW = np.log(1e-12)
_IRR_GRID_POINTS = 2001
_IRR_BISECT_ITERS = 200


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
        退役拆除费用 (万元/MW)，于寿命期末一次性发生
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


def real_discount_rate(nominal_rate: float, inflation_rate: float) -> float:
    """由名义折现率与通胀率计算实际折现率（费雪方程）。"""
    if inflation_rate <= -1.0:
        raise ValueError(f"通胀率必须大于 -100%，当前: {inflation_rate}")
    return float((1.0 + nominal_rate) / (1.0 + inflation_rate) - 1.0)


def annuity_factor(rate: float, years: float) -> float:
    """年金现值因子 ``A(r,n) = sum_{t=1..n} (1+r)^-t``。

    在 ``r = 0`` 处连续（极限为 ``n``），在零附近使用二阶泰勒展开，
    其余位置使用数值稳定的 ``-expm1(-n*log1p(r))/r``。
    """
    if years <= 0:
        raise ValueError(f"寿命期必须为正数，当前: {years}")
    if abs(rate) < _RATE_EPS:
        # A(r) = n - n(n+1)/2 r + n(n+1)(n+2)/6 r^2 + O(r^3)
        return float(
            years
            - years * (years + 1.0) / 2.0 * rate
            + years * (years + 1.0) * (years + 2.0) / 6.0 * rate * rate
        )
    if rate <= -1.0:
        raise ValueError(f"折现率必须大于 -100%，当前: {rate}")
    return float(-np.expm1(-years * np.log1p(rate)) / rate)


def discount_factor(rate: float, year: float) -> float:
    """折现因子 ``(1+r)^-t``，在 ``r = 0`` 处连续为 1。"""
    if rate <= -1.0:
        raise ValueError(f"折现率必须大于 -100%，当前: {rate}")
    if abs(rate) < _RATE_EPS:
        # exp(-t*log1p(r)) 的一阶展开，保证与年金因子同一精度
        return float(1.0 - year * rate)
    return float(np.exp(-year * np.log1p(rate)))


@dataclass
class LifecycleEconomics:
    """一组寿命期现金流的折现结果（NPV 与 LCOE 的唯一数据来源）。

    所有金额单位为万元（另有能量单位注明），均为基年不变价。

    Parameters
    ----------
    lifetime_years : float
        寿命期 (年)
    nominal_discount_rate : float
        名义折现率
    inflation_rate : float
        通胀率
    real_discount_rate : float
        实际折现率
    capital_cost : float
        t=0 初始投资
    decommissioning_cost : float
        t=n 退役费用（名义发生额）
    annual_revenue : float
        年发电收益
    annual_om_cost : float
        年运维费用
    annual_net_cash_flow : float
        年运营净现金流 (收益 - 运维)
    annual_energy_kwh : float
        年发电量 (kWh)
    annuity_factor : float
        年金现值因子
    decommissioning_pv : float
        退役费用现值
    om_pv : float
        运维费用现值
    revenue_pv : float
        发电收益现值
    energy_pv_kwh : float
        折现发电量 (kWh)
    lifecycle_cost_pv : float
        寿命期成本现值 = 初始投资 + 运维现值 + 退役现值
    npv : float
        净现值
    """

    lifetime_years: float
    nominal_discount_rate: float
    inflation_rate: float
    real_discount_rate: float
    capital_cost: float
    decommissioning_cost: float
    annual_revenue: float
    annual_om_cost: float
    annual_net_cash_flow: float
    annual_energy_kwh: float
    annuity_factor: float
    decommissioning_pv: float
    om_pv: float
    revenue_pv: float
    energy_pv_kwh: float
    lifecycle_cost_pv: float
    npv: float


@dataclass
class IRRResult:
    """内部收益率求解结果。

    Parameters
    ----------
    status : str
        求解状态，取值：

        * ``"unique"``：找到唯一实根，``irr_pct`` 即该内部收益率；
        * ``"multiple"``：存在多个实根（典型为大额退役费导致符号多次
          变化），``all_irr_pct`` 给出全部根，``irr_pct`` 为 ``None``，
          单一 IRR 不适用，NPV 在最低/最高根之间为正；
        * ``"no_root"``：定义域 ``r > -1`` 内无根（项目在任何可行折
          现率下 NPV 均为负，或无投资）；
        * ``"no_investment"``：初始投资不大于 0，IRR 无定义。
    irr_pct : Optional[float]
        唯一内部收益率（实际口径，%）；多根或无根时为 ``None``
    all_irr_pct : list[float]
        搜索到的全部实根（实际口径，%），升序排列
    irr_nominal_pct : Optional[float]
        唯一根对应的名义内部收益率（%），多根/无根时为 ``None``
    message : str
        状态说明
    """

    status: str
    irr_pct: Optional[float] = None
    all_irr_pct: list[float] = field(default_factory=list)
    irr_nominal_pct: Optional[float] = None
    message: str = ""


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
    lcoe : float
        度电成本 (元/kWh)；无有效发电量时为 ``numpy.inf``，
        此时 ``lcoe_status == "no_energy"``
    lcoe_status : str
        ``"ok"`` 或 ``"no_energy"``（年发电量不大于 0）
    total_capital_cost : float
        总初始投资 (万元)
    total_om_cost_annual : float
        年运维费用 (万元/year)
    decommissioning_cost : float
        寿命期末退役费用 (万元，名义发生额)
    decommissioning_cost_pv : float
        退役费用现值 (万元)
    lifecycle_cost_pv : float
        寿命期成本现值 (万元)
    npv : Optional[float]
        净现值 (万元)
    irr : Optional[float]
        唯一内部收益率（实际口径，%）；无根或多根时为 ``None``，
        详见 ``irr_result``
    irr_result : Optional[IRRResult]
        IRR 求解的完整结果（含多根、状态说明）
    payback_period : Optional[float]
        静态投资回收期 (年)；年净现金流不为正或寿命期内无法收回时
        为 ``None``
    cost_breakdown : dict[str, float]
        初始投资成本分项 (万元)，各项之和等于 ``total_capital_cost``
    """

    total_installed_capacity: float
    net_aep: float
    annual_revenue: float
    lcoe: float
    total_capital_cost: float
    total_om_cost_annual: float
    npv: Optional[float]
    irr: Optional[float]
    payback_period: Optional[float]
    cost_breakdown: dict[str, float]
    lcoe_status: str = "ok"
    decommissioning_cost: float = 0.0
    decommissioning_cost_pv: float = 0.0
    lifecycle_cost_pv: float = 0.0
    irr_result: Optional[IRRResult] = None


def _npv_at_rate(
    rate: float,
    capital_cost: float,
    annual_net_cash_flow: float,
    decommissioning_cost: float,
    lifetime: float,
) -> float:
    """按给定实际折现率计算 NPV（标量），定义域为 ``r > -1``。

    采用因式分解形式避免 ``r -> -1`` 时两个无穷大相减::

        q = 1+r, S = 1 + q + ... + q^(n-1)
        A(r, n) = q^-n * S
        NPV = (c*S - D) * q^-n - I

    与 :func:`annuity_factor` 相同的零点连续处理，因此负利率可以
    正确求值，且在浮点极限处得到符号正确的 ±inf 而非 nan。
    """
    r = float(rate)
    n = float(lifetime)
    c = float(annual_net_cash_flow)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if abs(r) < _RATE_EPS:
            # S(r) = n + n(n-1)/2 r + O(r^2)；q^-n ≈ 1 - nr
            s_geo = n + n * (n - 1.0) / 2.0 * r
            q_inv_n = 1.0 - n * r
            bracket = c * s_geo - decommissioning_cost
            tail = bracket * q_inv_n
            if not np.isfinite(tail) and abs(bracket) > 0.0:
                tail = np.sign(bracket) * np.inf
            return float(-capital_cost + tail)

        log_q = np.log1p(r)
        # S = (1 - q^n) / (1 - q) = expm1(n lnq) / expm1(lnq)
        s_geo = np.expm1(n * log_q) / np.expm1(log_q)
        bracket = c * s_geo - decommissioning_cost
        log_tail = np.log(abs(bracket)) - n * log_q if bracket != 0.0 else -np.inf
        if log_tail > 709.0:
            tail = np.sign(bracket) * np.inf if bracket != 0.0 else 0.0
        else:
            tail = np.sign(bracket) * np.exp(log_tail) if bracket != 0.0 else 0.0
        return float(-capital_cost + tail)


def solve_irr(
    capital_cost: float,
    annual_net_cash_flow: float,
    lifetime: float,
    decommissioning_cost: float = 0.0,
) -> IRRResult:
    """求解寿命期现金流的内部收益率（实际口径，允许为负）。

    现金流为 ``t=0`` 投资流出、``t=1..n`` 年等额净流入、``t=n`` 退役
    费流出。在 ``s = ln(1+r)`` 空间做全域网格扫描 + 符号区间二分，
    因此：

    * 亏损项目若在 ``-1 < r < 0`` 存在根，可以正确返回负 IRR；
    * 退役费过大造成符号多次变化时可识别出多个根；
    * 任何可行折现率下 NPV 均为负时明确返回 ``"no_root"``。

    Returns
    -------
    IRRResult
        求解状态与全部根（%）
    """
    if lifetime <= 0:
        raise ValueError(f"寿命期必须为正数，当前: {lifetime}")

    if capital_cost <= 0:
        return IRRResult(
            status="no_investment",
            message="初始投资不大于 0，内部收益率无定义",
        )

    def f(r: float) -> float:
        return _npv_at_rate(
            r, capital_cost, annual_net_cash_flow, decommissioning_cost, lifetime
        )

    # 年净现金流不为正时，年金项与退役项均为非正，NPV 在 r>-1 上恒负
    if annual_net_cash_flow <= 0:
        return IRRResult(
            status="no_root",
            message="项目在寿命期内不产生正的净现金流，r>-1 域内不存在 IRR",
        )

    scale = max(
        1.0,
        abs(capital_cost),
        abs(annual_net_cash_flow) * lifetime,
        abs(decommissioning_cost),
    )
    tol = 1e-8 * scale

    # r = 0 处的精确根（临界项目）
    roots: list[float] = []
    if abs(f(0.0)) <= tol:
        roots.append(0.0)

    def bisect(s_lo: float, s_hi: float) -> float:
        """在 s 空间二分，返回 r 空间的根。"""
        f_lo = f(float(np.expm1(s_lo)))
        for _ in range(_IRR_BISECT_ITERS):
            s_mid = 0.5 * (s_lo + s_hi)
            f_mid = f(float(np.expm1(s_mid)))
            if (f_mid < 0.0) == (f_lo < 0.0):
                s_lo, f_lo = s_mid, f_mid
            else:
                s_hi = s_mid
        return float(np.expm1(0.5 * (s_lo + s_hi)))

    def record(candidate: float, residual: Optional[float] = None) -> None:
        """登记一个候选根（限 r>-1、残差可接受、与已有根去重）。"""
        if not np.isfinite(candidate) or candidate <= -1.0:
            return
        if residual is None:
            residual = abs(f(candidate))
        # 二分到机器精度时，浮点求残差受限于大数相消，放宽至相对尺度
        if residual > 1e-3 * scale:
            return
        for existing in roots:
            if abs(candidate - existing) <= 1e-7 * max(1.0, abs(existing)):
                return
        roots.append(candidate)

    def scan_brackets(s_grid: np.ndarray, f_grid: np.ndarray) -> None:
        """在采样网格上收集严格异号区间并二分；近零点局部细化。"""
        n = len(s_grid)
        for k in range(n - 1):
            f0, f1 = float(f_grid[k]), float(f_grid[k + 1])
            neg0, neg1 = f0 < 0.0, f1 < 0.0
            if neg0 != neg1:
                record(bisect(float(s_grid[k]), float(s_grid[k + 1])))
        # 网格点几乎为零（根贴着采样点）时，在相邻区间细化后再二分，
        # 避免把“粗网格近似根”和精确根重复登记
        for k in range(n):
            if abs(float(f_grid[k])) <= tol:
                lo = float(s_grid[max(0, k - 1)])
                hi = float(s_grid[min(n - 1, k + 1)])
                fine = np.linspace(lo, hi, 101)
                f_fine = np.array([f(float(np.expm1(s))) for s in fine])
                for j in range(len(fine) - 1):
                    if (f_fine[j] < 0.0) != (f_fine[j + 1] < 0.0):
                        record(bisect(float(fine[j]), float(fine[j + 1])))

    # 向高端扩展搜索区间：r -> inf 时 NPV -> -CapEx < 0
    r_hi = 1.0
    f_hi = f(r_hi)
    for _ in range(50):
        if f_hi <= 0:
            break
        r_hi *= 2.0
        f_hi = f(r_hi)
    else:
        return IRRResult(
            status="no_root",
            message="IRR 超出搜索上限，无法可靠求解",
        )

    # s = ln(1+r) 上均匀采样，自动在 r -> -1 处加密
    s_lo_boundary = _IRR_S_LOW
    s_grid = np.linspace(s_lo_boundary, np.log1p(r_hi), _IRR_GRID_POINTS)
    f_grid = np.array([f(float(np.expm1(s))) for s in s_grid])
    scan_brackets(s_grid, f_grid)

    # 若年退役费大于年净现金流，r -> -1+ 时 NPV -> -inf。最密网格点
    # 仍可能为正，需继续向 -1 方向扩展，直到 NPV 转负或接近浮点极限。
    if annual_net_cash_flow - decommissioning_cost < 0:
        for _ in range(6):
            r_check = float(np.expm1(s_lo_boundary))
            if f(r_check) <= 0:
                break
            s_new = s_lo_boundary - np.log(1e2)
            if s_new < -700.0:
                break
            block = np.linspace(s_new, s_lo_boundary, 401)
            f_block = np.array([f(float(np.expm1(s))) for s in block])
            scan_brackets(block, f_block)
            s_lo_boundary = s_new

    roots.sort()

    if not roots:
        return IRRResult(
            status="no_root",
            message="r>-1 域内 NPV 不变号为零：项目在任何可行折现率下均不可收回投资",
        )

    if len(roots) > 1:
        pcts = [r * 100.0 for r in roots]
        return IRRResult(
            status="multiple",
            all_irr_pct=pcts,
            message=(
                f"存在 {len(roots)} 个内部收益率 {['%.4f%%' % p for p in pcts]}，"
                "NPV 仅在最低与最高根之间为正，单一 IRR 不适用（通常由过大的"
                "期末退役支出造成）"
            ),
        )

    r_root = roots[0]
    return IRRResult(
        status="unique",
        irr_pct=r_root * 100.0,
        all_irr_pct=[r_root * 100.0],
        message="唯一内部收益率" + ("（临界，约为零）" if abs(r_root) < 1e-9 else
                                    ("（负收益）" if r_root < 0 else "")),
    )


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
        self.turbine_cost = turbine_cost
        self.farm_cost = farm_cost
        self.electricity_price = electricity_price

    # ------------------------------------------------------------------
    # 造价
    # ------------------------------------------------------------------
    def compute_capital_cost(
        self,
        n_turbines: int,
        rated_power_per_turbine_MW: float,
    ) -> tuple[float, dict[str, float]]:
        """计算初始投资。

        Returns
        -------
        tuple[float, dict[str, float]]
            - 总初始投资 (万元)
            - 成本分项明细（各项之和等于总投资）
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

    def compute_decommissioning_cost(self, total_capacity_MW: float) -> float:
        """计算寿命期末一次性退役拆除费用 (万元)。"""
        return float(total_capacity_MW * self.farm_cost.decommissioning_cost_per_MW)

    def compute_annual_om_cost(
        self,
        n_turbines: int,
        rated_power_per_turbine_MW: float,
    ) -> float:
        """计算年运维费用 (万元/year)。"""
        total_capacity = n_turbines * rated_power_per_turbine_MW
        return float(total_capacity * self.turbine_cost.o_and_m_cost_per_MW_per_year)

    def compute_annual_revenue(self, net_aep_GWh: float) -> float:
        """计算年发电收益 (万元/year)。"""
        return float(net_aep_GWh * 1e6 * self.electricity_price / 1e4)

    # ------------------------------------------------------------------
    # 统一的寿命期现金流
    # ------------------------------------------------------------------
    def build_lifecycle(
        self,
        total_capital_cost: float,
        annual_om_cost: float,
        annual_revenue: float,
        annual_energy_kwh: float,
        decommissioning_cost: float = 0.0,
        lifetime: Optional[float] = None,
    ) -> LifecycleEconomics:
        """构建并折现寿命期现金流（NPV 与 LCOE 的唯一计算口径）。

        退役费用在寿命期末（``t = lifetime``）一次性计入，运维与发电
        量按年等额发生。实际折现率为零（平价情景）时所有折现因子连续
        退化为 1，不抛除零异常。
        """
        if lifetime is None:
            lifetime = self.turbine_cost.design_lifetime
        if lifetime <= 0:
            raise ValueError(f"寿命期必须为正数，当前: {lifetime}")

        r_nominal = self.farm_cost.discount_rate
        inflation = self.farm_cost.inflation_rate
        r_real = real_discount_rate(r_nominal, inflation)

        af = annuity_factor(r_real, lifetime)
        df_end = discount_factor(r_real, lifetime)

        annual_net = annual_revenue - annual_om_cost
        decom_pv = decommissioning_cost * df_end
        om_pv = annual_om_cost * af
        revenue_pv = annual_revenue * af
        energy_pv = annual_energy_kwh * af
        cost_pv = total_capital_cost + om_pv + decom_pv
        npv = -total_capital_cost + annual_net * af - decom_pv

        return LifecycleEconomics(
            lifetime_years=float(lifetime),
            nominal_discount_rate=float(r_nominal),
            inflation_rate=float(inflation),
            real_discount_rate=float(r_real),
            capital_cost=float(total_capital_cost),
            decommissioning_cost=float(decommissioning_cost),
            annual_revenue=float(annual_revenue),
            annual_om_cost=float(annual_om_cost),
            annual_net_cash_flow=float(annual_net),
            annual_energy_kwh=float(annual_energy_kwh),
            annuity_factor=float(af),
            decommissioning_pv=float(decom_pv),
            om_pv=float(om_pv),
            revenue_pv=float(revenue_pv),
            energy_pv_kwh=float(energy_pv),
            lifecycle_cost_pv=float(cost_pv),
            npv=float(npv),
        )

    def build_cash_flow_schedule(
        self,
        lifecycle: LifecycleEconomics,
    ) -> dict[str, list[float]]:
        """生成逐年现金流明细，供输出复核。

        退役费用计入 ``t = n`` 当年；``net_pv`` 为各年净现金流按实际
        折现率折现到 ``t=0`` 的现值，逐年 ``net_pv`` 之和即 NPV。
        """
        n = int(round(lifecycle.lifetime_years))
        r = lifecycle.real_discount_rate
        years = list(range(n + 1))
        capex = [0.0] * (n + 1)
        om = [0.0] * (n + 1)
        revenue = [0.0] * (n + 1)
        decom = [0.0] * (n + 1)
        factors = [1.0] * (n + 1)

        capex[0] = lifecycle.capital_cost
        for t in range(1, n + 1):
            om[t] = lifecycle.annual_om_cost
            revenue[t] = lifecycle.annual_revenue
            factors[t] = discount_factor(r, t)
        decom[n] = lifecycle.decommissioning_cost

        net = [
            revenue[t] - om[t] - decom[t] - (capex[t] if t == 0 else 0.0)
            for t in years
        ]
        net_pv = [net[t] * factors[t] for t in years]
        return {
            "year": [float(t) for t in years],
            "capital_cost": capex,
            "om_cost": om,
            "revenue": revenue,
            "decommissioning_cost": decom,
            "net_cash_flow": net,
            "discount_factor": factors,
            "net_present_value": net_pv,
        }

    # ------------------------------------------------------------------
    # 经济指标
    # ------------------------------------------------------------------
    def compute_lcoe(
        self,
        total_capital_cost: float,
        total_om_cost_annual: float,
        net_aep_GWh: float,
        lifetime: Optional[float] = None,
        decommissioning_cost: float = 0.0,
    ) -> float:
        """计算度电成本(LCOE)，单位元/kWh。

        LCOE = 寿命期成本现值 / 寿命期发电量现值，与 NPV 共用同一套
        折现现金流（含期末退役费用）。

        无有效发电量（``net_aep_GWh <= 0``）时返回 ``numpy.inf``。
        """
        if net_aep_GWh <= 0:
            return float(np.inf)

        lc = self.build_lifecycle(
            total_capital_cost=total_capital_cost,
            annual_om_cost=total_om_cost_annual,
            annual_revenue=0.0,
            annual_energy_kwh=net_aep_GWh * 1e6,
            decommissioning_cost=decommissioning_cost,
            lifetime=lifetime,
        )
        return float(lc.lifecycle_cost_pv * 1e4 / lc.energy_pv_kwh)

    def compute_npv(
        self,
        total_capital_cost: float,
        annual_revenue: float,
        annual_om_cost: float,
        lifetime: Optional[float] = None,
        decommissioning_cost: float = 0.0,
    ) -> float:
        """计算净现值(NPV，万元)。

        退役费用在寿命期末折现计入；实际折现率为零时结果连续。
        """
        lc = self.build_lifecycle(
            total_capital_cost=total_capital_cost,
            annual_om_cost=annual_om_cost,
            annual_revenue=annual_revenue,
            annual_energy_kwh=0.0,
            decommissioning_cost=decommissioning_cost,
            lifetime=lifetime,
        )
        return lc.npv

    def compute_payback_period(
        self,
        total_capital_cost: float,
        annual_revenue: float,
        annual_om_cost: float,
        lifetime: Optional[float] = None,
    ) -> Optional[float]:
        """计算静态投资回收期（不折现）。

        年净现金流不为正，或简单回收期超过寿命期（即寿命期内无法收
        回投资）时返回 ``None``。
        """
        if lifetime is None:
            lifetime = self.turbine_cost.design_lifetime
        net_annual = annual_revenue - annual_om_cost
        if net_annual <= 0:
            return None
        payback = total_capital_cost / net_annual
        if payback > lifetime:
            return None
        return float(payback)

    def compute_irr(
        self,
        total_capital_cost: float,
        annual_revenue: float,
        annual_om_cost: float,
        lifetime: Optional[float] = None,
        decommissioning_cost: float = 0.0,
    ) -> IRRResult:
        """计算内部收益率（实际口径，允许为负）。

        详见 :func:`solve_irr` 的返回语义（唯一根 / 多根 / 无根 /
        无投资）。
        """
        if lifetime is None:
            lifetime = self.turbine_cost.design_lifetime
        result = solve_irr(
            capital_cost=total_capital_cost,
            annual_net_cash_flow=annual_revenue - annual_om_cost,
            lifetime=float(lifetime),
            decommissioning_cost=decommissioning_cost,
        )
        # 唯一根时附上名义口径，便于投决对照
        if result.status == "unique" and result.irr_pct is not None:
            r_real = result.irr_pct / 100.0
            r_nominal = (1.0 + r_real) * (1.0 + self.farm_cost.inflation_rate) - 1.0
            result.irr_nominal_pct = r_nominal * 100.0
        return result

    def analyze(
        self,
        n_turbines: int,
        rated_power_per_turbine_MW: float,
        net_aep_GWh: float,
    ) -> EconomicResult:
        """进行完整的经济性分析。

        所有指标（NPV、LCOE、IRR、成本分项、逐年现金流）均由同一组
        寿命期现金流导出，保证摘要、分项与结果文件数值一致。
        """
        total_capacity = n_turbines * rated_power_per_turbine_MW

        total_capital_cost, cost_breakdown = self.compute_capital_cost(
            n_turbines, rated_power_per_turbine_MW
        )
        annual_om_cost = self.compute_annual_om_cost(
            n_turbines, rated_power_per_turbine_MW
        )
        decommissioning_cost = self.compute_decommissioning_cost(total_capacity)
        annual_revenue = self.compute_annual_revenue(net_aep_GWh)

        lifecycle = self.build_lifecycle(
            total_capital_cost=total_capital_cost,
            annual_om_cost=annual_om_cost,
            annual_revenue=annual_revenue,
            annual_energy_kwh=net_aep_GWh * 1e6,
            decommissioning_cost=decommissioning_cost,
        )

        if net_aep_GWh <= 0:
            lcoe = float(np.inf)
            lcoe_status = "no_energy"
        else:
            lcoe = float(lifecycle.lifecycle_cost_pv * 1e4 / lifecycle.energy_pv_kwh)
            lcoe_status = "ok"

        irr_result = self.compute_irr(
            total_capital_cost=total_capital_cost,
            annual_revenue=annual_revenue,
            annual_om_cost=annual_om_cost,
            lifetime=lifecycle.lifetime_years,
            decommissioning_cost=decommissioning_cost,
        )
        payback = self.compute_payback_period(
            total_capital_cost,
            annual_revenue,
            annual_om_cost,
            lifetime=lifecycle.lifetime_years,
        )

        return EconomicResult(
            total_installed_capacity=float(total_capacity),
            net_aep=float(net_aep_GWh),
            annual_revenue=float(annual_revenue),
            lcoe=lcoe,
            lcoe_status=lcoe_status,
            total_capital_cost=float(total_capital_cost),
            total_om_cost_annual=float(annual_om_cost),
            decommissioning_cost=float(decommissioning_cost),
            decommissioning_cost_pv=lifecycle.decommissioning_pv,
            lifecycle_cost_pv=lifecycle.lifecycle_cost_pv,
            npv=lifecycle.npv,
            irr=irr_result.irr_pct,
            irr_result=irr_result,
            payback_period=payback,
            cost_breakdown=cost_breakdown,
        )


def get_default_turbine_cost(model: str = "V164-9.5MW") -> TurbineCostModel:
    """获取默认风机造价模型。"""
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
