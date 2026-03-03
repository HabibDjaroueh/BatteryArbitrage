import numpy as np
import pandas as pd
from dataclasses import dataclass


# ── Battery Specification ──────────────────────────────────────────────────

@dataclass
class BatterySpec:
    """
    Physical specification of the battery asset.

    Parameters
    ----------
    power_mw        : Rated power in MW (max charge or discharge rate)
    duration_hours  : Storage duration in hours
    charge_eff      : One-way charge efficiency (default 0.92)
    discharge_eff   : One-way discharge efficiency (default 0.92)
    min_soc_pct     : Minimum SoC as % of capacity (default 10%)
    max_soc_pct     : Maximum SoC as % of capacity (default 100%)
    initial_soc_pct : Starting SoC as % of capacity (default 50%)
    ramp_rate_pct   : Max ramp per hour as % of power rating (default 100%)
                      e.g. 0.5 = battery can only ramp 50% of rated MW per hour

    Notes
    -----
    Round-trip efficiency = charge_eff × discharge_eff
    Default: 0.92 × 0.92 = 0.846 ≈ 85% round-trip
    This is more physically accurate than applying efficiency
    only on discharge — losses occur in both directions.
    """
    power_mw:         float
    duration_hours:   float
    charge_eff:       float = 0.92
    discharge_eff:    float = 0.92
    min_soc_pct:      float = 10.0
    max_soc_pct:      float = 100.0
    initial_soc_pct:  float = 50.0
    ramp_rate_pct:    float = 100.0

    def __post_init__(self):
        assert 0 < self.power_mw, "power_mw must be positive"
        assert 0 < self.duration_hours, "duration_hours must be positive"
        assert 0 < self.charge_eff <= 1, "charge_eff must be in (0, 1]"
        assert 0 < self.discharge_eff <= 1, "discharge_eff must be in (0, 1]"
        assert 0 <= self.min_soc_pct < self.max_soc_pct <= 100, "SoC bounds invalid"
        assert 0 < self.ramp_rate_pct <= 100, "ramp_rate_pct must be in (0, 100]"

    @property
    def energy_capacity_mwh(self) -> float:
        """Total energy capacity in MWh."""
        return self.power_mw * self.duration_hours

    @property
    def min_soc_mwh(self) -> float:
        """Minimum SoC in MWh."""
        return self.energy_capacity_mwh * self.min_soc_pct / 100

    @property
    def max_soc_mwh(self) -> float:
        """Maximum SoC in MWh."""
        return self.energy_capacity_mwh * self.max_soc_pct / 100

    @property
    def initial_soc_mwh(self) -> float:
        """Initial SoC in MWh."""
        return self.energy_capacity_mwh * self.initial_soc_pct / 100

    @property
    def max_ramp_mw(self) -> float:
        """Maximum MW change per hour."""
        return self.power_mw * self.ramp_rate_pct / 100

    @property
    def round_trip_efficiency(self) -> float:
        """Round-trip efficiency = charge_eff × discharge_eff."""
        return self.charge_eff * self.discharge_eff


# ── Hourly Dispatch Action ─────────────────────────────────────────────────

@dataclass
class DispatchAction:
    """
    Represents the battery's intended action for a single hour.

    charge_mw    : MW to charge (positive value, 0 if not charging)
    discharge_mw : MW to discharge (positive value, 0 if not discharging)

    Constraints enforced by SoCEngine:
    - Cannot charge and discharge simultaneously
    - Cannot exceed power rating
    - Cannot exceed ramp rate from previous hour
    """
    charge_mw:    float = 0.0
    discharge_mw: float = 0.0

    def __post_init__(self):
        assert self.charge_mw >= 0, "charge_mw must be non-negative"
        assert self.discharge_mw >= 0, "discharge_mw must be non-negative"
        assert not (self.charge_mw > 0 and self.discharge_mw > 0), (
            "Cannot charge and discharge simultaneously"
        )


# ── SoC Engine ────────────────────────────────────────────────────────────

class SoCEngine:
    """
    Physical battery state of charge engine.

    Enforces all physical constraints on battery operation:
    - SoC bounds (min/max)
    - Power rating limits
    - Ramp rate limits
    - No simultaneous charge/discharge
    - Efficiency applied separately on charge and discharge

    Usage
    -----
    engine = SoCEngine(spec)
    result = engine.step(action)   # advance one hour
    engine.reset()                 # reset to initial state

    Parameters
    ----------
    spec : BatterySpec defining the physical characteristics
    """

    def __init__(self, spec: BatterySpec):
        self.spec = spec
        self.soc_mwh = spec.initial_soc_mwh
        self._prev_charge_mw = 0.0
        self._prev_discharge_mw = 0.0

    def reset(self):
        """Reset battery to initial state."""
        self.soc_mwh = self.spec.initial_soc_mwh
        self._prev_charge_mw = 0.0
        self._prev_discharge_mw = 0.0

    @property
    def soc_pct(self) -> float:
        """Current SoC as percentage of total capacity."""
        return self.soc_mwh / self.spec.energy_capacity_mwh * 100

    @property
    def available_charge_mwh(self) -> float:
        """How much more energy can be stored right now."""
        return self.spec.max_soc_mwh - self.soc_mwh

    @property
    def available_discharge_mwh(self) -> float:
        """How much energy can be dispatched right now."""
        return self.soc_mwh - self.spec.min_soc_mwh

    def _apply_ramp_limit(
        self,
        requested_mw: float,
        prev_mw: float,
    ) -> float:
        """
        Clips MW to the ramp rate limit from the previous hour.
        Ramp applies symmetrically — both ramping up and down.
        """
        max_ramp = self.spec.max_ramp_mw
        return float(np.clip(requested_mw, prev_mw - max_ramp, prev_mw + max_ramp))

    def step(self, action: DispatchAction) -> dict:
        """
        Advance the battery state by one hour given a dispatch action.

        Applies constraints in this order:
        1. No simultaneous charge/discharge (enforced by DispatchAction)
        2. Clip to power rating
        3. Apply ramp rate limit from previous hour
        4. Clip to SoC bounds (cannot overfill or overdrain)
        5. Apply efficiency to compute actual energy transferred
        6. Update SoC

        Parameters
        ----------
        action : DispatchAction with requested charge_mw and discharge_mw

        Returns
        -------
        dict with:
            charge_mw_requested    : what was requested
            discharge_mw_requested : what was requested
            charge_mw_actual       : after all constraints applied
            discharge_mw_actual    : after all constraints applied
            energy_charged_mwh     : energy added to battery (after charge_eff)
            energy_discharged_mwh  : energy removed from battery
            energy_delivered_mwh   : energy sent to grid (after discharge_eff)
            soc_mwh_before         : SoC at start of hour
            soc_mwh_after          : SoC at end of hour
            soc_pct_before         : SoC % at start of hour
            soc_pct_after          : SoC % at end of hour
            charge_clipped         : True if charge was clipped by any constraint
            discharge_clipped      : True if discharge was clipped by any constraint
        """
        soc_before = self.soc_mwh
        soc_pct_before = self.soc_pct

        charge_mw_req = action.charge_mw
        discharge_mw_req = action.discharge_mw

        # ── Charging path ──────────────────────────────────────────────────
        charge_mw_actual = 0.0
        energy_charged_mwh = 0.0

        if charge_mw_req > 0:
            # 1. Clip to power rating
            capped = min(charge_mw_req, self.spec.power_mw)

            # 2. Apply ramp limit
            ramped = self._apply_ramp_limit(capped, self._prev_charge_mw)
            ramped = max(0.0, ramped)

            # 3. Clip to available capacity
            #    Energy into battery = MW × charge_eff × 1 hour
            #    So max MW = available_charge_mwh / charge_eff
            max_charge_mw = self.available_charge_mwh / self.spec.charge_eff
            charge_mw_actual = min(ramped, max_charge_mw)
            charge_mw_actual = max(0.0, charge_mw_actual)

            # Energy added to battery (losses occur on the way in)
            energy_charged_mwh = charge_mw_actual * self.spec.charge_eff

        # ── Discharging path ───────────────────────────────────────────────
        discharge_mw_actual = 0.0
        energy_discharged_mwh = 0.0
        energy_delivered_mwh = 0.0

        if discharge_mw_req > 0:
            # 1. Clip to power rating
            capped = min(discharge_mw_req, self.spec.power_mw)

            # 2. Apply ramp limit
            ramped = self._apply_ramp_limit(capped, self._prev_discharge_mw)
            ramped = max(0.0, ramped)

            # 3. Clip to available energy
            #    Energy from battery = MW × 1 hour
            #    Delivered to grid   = MW × discharge_eff × 1 hour
            max_discharge_mw = self.available_discharge_mwh
            discharge_mw_actual = min(ramped, max_discharge_mw)
            discharge_mw_actual = max(0.0, discharge_mw_actual)

            # Energy removed from battery and delivered to grid
            energy_discharged_mwh = discharge_mw_actual
            energy_delivered_mwh = discharge_mw_actual * self.spec.discharge_eff

        # ── Update SoC ─────────────────────────────────────────────────────
        self.soc_mwh += energy_charged_mwh - energy_discharged_mwh
        self.soc_mwh = float(np.clip(
            self.soc_mwh,
            self.spec.min_soc_mwh,
            self.spec.max_soc_mwh,
        ))

        # ── Update previous MW for next ramp calculation ───────────────────
        self._prev_charge_mw = charge_mw_actual
        self._prev_discharge_mw = discharge_mw_actual

        return {
            "charge_mw_requested": charge_mw_req,
            "discharge_mw_requested": discharge_mw_req,
            "charge_mw_actual": round(charge_mw_actual, 3),
            "discharge_mw_actual": round(discharge_mw_actual, 3),
            "energy_charged_mwh": round(energy_charged_mwh, 3),
            "energy_discharged_mwh": round(energy_discharged_mwh, 3),
            "energy_delivered_mwh": round(energy_delivered_mwh, 3),
            "soc_mwh_before": round(soc_before, 3),
            "soc_mwh_after": round(self.soc_mwh, 3),
            "soc_pct_before": round(soc_pct_before, 2),
            "soc_pct_after": round(self.soc_pct, 2),
            "charge_clipped": charge_mw_actual < charge_mw_req,
            "discharge_clipped": discharge_mw_actual < discharge_mw_req,
        }


# ── Multi-Hour Simulation Runner ───────────────────────────────────────────

def simulate_dispatch_schedule(
    spec: BatterySpec,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """
    Runs the SoC engine over a full dispatch schedule (hourly resolution).

    Parameters
    ----------
    spec     : BatterySpec
    schedule : pd.DataFrame with datetime index and columns:
                 charge_mw    — requested charge MW per hour
                 discharge_mw — requested discharge MW per hour

    Returns
    -------
    pd.DataFrame with one row per hour containing all SoC engine
    step() outputs plus the original schedule columns.
    """
    engine = SoCEngine(spec)
    results = []

    for ts, row in schedule.iterrows():
        action = DispatchAction(
            charge_mw=max(0.0, float(row.get("charge_mw", 0))),
            discharge_mw=max(0.0, float(row.get("discharge_mw", 0))),
        )
        result = engine.step(action)
        result["timestamp"] = ts
        results.append(result)

    return pd.DataFrame(results).set_index("timestamp")


# ── Revenue Calculator ─────────────────────────────────────────────────────

def calculate_revenue(
    dispatch_df: pd.DataFrame,
    prices: pd.Series,
    charge_price_col: str = "charge_price",
    discharge_price_col: str = "discharge_price",
) -> pd.DataFrame:
    """
    Calculates hourly revenue from a completed dispatch simulation.

    Revenue per hour:
        charge hour:    revenue = -charge_price    × energy_charged_mwh
                                   (negative — cost to buy power)
        discharge hour: revenue = +discharge_price × energy_delivered_mwh
                                   (positive — revenue from selling power)
        net_revenue     = discharge_revenue + charge_cost (charge_cost is negative)

    Parameters
    ----------
    dispatch_df         : output of simulate_dispatch_schedule()
    prices              : pd.Series of hourly prices aligned to dispatch_df index
    charge_price_col    : column name for charge prices (unused if prices provided)
    discharge_price_col : column name for discharge prices (unused if prices provided)

    Returns
    -------
    dispatch_df with additional columns:
        price, charge_cost, discharge_revenue, net_revenue, cumulative_revenue
    """
    df = dispatch_df.copy()
    df["price"] = prices.reindex(df.index)

    df["charge_cost"] = -df["price"] * df["energy_charged_mwh"]
    df["discharge_revenue"] = df["price"] * df["energy_delivered_mwh"]
    df["net_revenue"] = df["charge_cost"] + df["discharge_revenue"]
    df["cumulative_revenue"] = df["net_revenue"].cumsum()

    return df


# ── SoC Diagnostic Summary ─────────────────────────────────────────────────

def soc_diagnostics(dispatch_df: pd.DataFrame, spec: BatterySpec) -> dict:
    """
    Computes diagnostic statistics from a completed simulation run.

    Returns
    -------
    dict with:
        min_soc_pct          : lowest SoC reached (%)
        max_soc_pct          : highest SoC reached (%)
        avg_soc_pct          : average SoC across all hours (%)
        charge_clipped_hours : hours where charge was constrained
        discharge_clipped_hours : hours where discharge was constrained
        total_energy_charged_mwh   : total energy into battery
        total_energy_delivered_mwh : total energy delivered to grid
        effective_rte        : actual round-trip efficiency achieved
        total_cycles         : estimated full cycles (delivered / capacity)
    """
    min_soc = dispatch_df["soc_pct_after"].min()
    max_soc = dispatch_df["soc_pct_after"].max()
    avg_soc = dispatch_df["soc_pct_after"].mean()

    charge_clipped = dispatch_df["charge_clipped"].sum()
    discharge_clipped = dispatch_df["discharge_clipped"].sum()

    total_charged = dispatch_df["energy_charged_mwh"].sum()
    total_delivered = dispatch_df["energy_delivered_mwh"].sum()

    effective_rte = (
        total_delivered / total_charged
        if total_charged > 0 else 0.0
    )

    total_cycles = (
        total_delivered / spec.energy_capacity_mwh
        if spec.energy_capacity_mwh > 0 else 0.0
    )

    return {
        "min_soc_pct": round(min_soc, 2),
        "max_soc_pct": round(max_soc, 2),
        "avg_soc_pct": round(avg_soc, 2),
        "charge_clipped_hours": int(charge_clipped),
        "discharge_clipped_hours": int(discharge_clipped),
        "total_energy_charged_mwh": round(total_charged, 2),
        "total_energy_delivered_mwh": round(total_delivered, 2),
        "effective_rte": round(effective_rte, 4),
        "total_cycles": round(total_cycles, 2),
    }
