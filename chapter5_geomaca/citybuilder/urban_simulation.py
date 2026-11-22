#!/usr/bin/env python3
"""
Urban Development Simulation - Three Pillars of Sustainability

A realistic urban simulation based on:
1. ECONOMY - Developer profit, GDP, tax revenue, market dynamics
2. ENVIRONMENT - Green space, woodland, carrying capacity  
3. SOCIETY - Segregation, QoL, housing affordability, satisfaction

Game mechanics:
- Population immigrates/emigrates based on city attractiveness
- Developers must profit or face bankruptcy
- Government balances conflicting objectives
- Housing supply must match demand by wealth class
- Environmental capacity limits growth

Win/Lose conditions create meaningful training signal.
"""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random


class WealthClass(IntEnum):
    POOR = 0
    MIDDLE = 1
    RICH = 2


class GameState(IntEnum):
    RUNNING = 0
    WIN = 1
    LOSE_POPULATION = 2
    LOSE_ECONOMY = 3
    LOSE_SEGREGATION = 4
    LOSE_BANKRUPTCY = 5


@dataclass
class Household:
    """
    A household unit with age-based lifecycle and class mobility.
    
    Lifecycle stages (gamma-like income distribution):
    - 20-30: Early career (low income, upward mobility)
    - 30-50: Peak earning years (highest income)
    - 50-65: Late career (stable income)
    - 65+: Retirement (fixed income, downward mobility)
    """
    id: int
    wealth_class: WealthClass
    location: Tuple[int, int]  # (y, x)
    satisfaction: float = 3.0  # 0-5 scale
    turns_dissatisfied: int = 0
    age: int = 35  # Default age
    education_level: int = 1  # 0=low, 1=medium, 2=high (affects mobility)
    years_in_class: int = 0  # Stability in current class
    
    # Base incomes by class (will be modified by age)
    BASE_INCOME = {
        WealthClass.POOR: 28000,
        WealthClass.MIDDLE: 70000,
        WealthClass.RICH: 160000
    }
    
    @property
    def age_income_multiplier(self) -> float:
        """
        Gamma-like income curve based on age:
        - Young (20-30): 0.5-0.8 (building career)
        - Prime (30-50): 1.0-1.3 (peak earnings)
        - Late (50-65): 0.9-1.1 (stable)
        - Retired (65+): 0.3-0.5 (fixed income/pension)
        """
        if self.age < 25:
            return 0.5 + (self.age - 20) * 0.06  # 0.5 -> 0.8
        elif self.age < 35:
            return 0.8 + (self.age - 25) * 0.05  # 0.8 -> 1.3
        elif self.age < 50:
            return 1.3 - (self.age - 35) * 0.01  # 1.3 -> 1.15
        elif self.age < 65:
            return 1.15 - (self.age - 50) * 0.02  # 1.15 -> 0.85
        else:
            # Retirement: sharp drop
            return max(0.3, 0.85 - (self.age - 65) * 0.03)  # 0.85 -> 0.3
    
    @property
    def income(self) -> float:
        """Annual income adjusted by age (gamma distribution effect)"""
        base = self.BASE_INCOME[self.wealth_class]
        return base * self.age_income_multiplier
    
    @property
    def housing_budget(self) -> float:
        """Max rent they can afford (30% of income)"""
        return self.income * 0.30 / 12  # Monthly
    
    @property
    def is_retired(self) -> bool:
        return self.age >= 65
    
    @property
    def mobility_potential(self) -> float:
        """
        Probability modifier for class mobility.
        Higher for: young, educated, stable in current class
        Lower for: old, less educated, recently changed
        """
        # Age factor: young have more mobility
        if self.age < 30:
            age_factor = 1.5
        elif self.age < 45:
            age_factor = 1.2
        elif self.age < 55:
            age_factor = 0.8
        else:
            age_factor = 0.3  # Hard to change class late in life
        
        # Education factor
        edu_factor = 0.7 + self.education_level * 0.3  # 0.7 to 1.3
        
        # Stability penalty (recently changed = less likely to change again)
        stability_factor = min(1.0, self.years_in_class / 5.0)  # Full after 5 years
        
        return age_factor * edu_factor * stability_factor


@dataclass
class Zone:
    """A development zone with all attributes"""
    id: int
    x: int
    y: int
    size: int = 20
    developed: bool = False
    
    # Housing counts by type
    poor_housing: int = 0
    medium_housing: int = 0
    luxury_housing: int = 0
    
    # Land use
    green_space: int = 0
    woodland: int = 0
    
    # Economics
    development_cost: float = 0.0
    monthly_revenue: float = 0.0
    property_value: float = 0.0
    monthly_mortgage: float = 0.0
    monthly_maintenance: float = 0.0
    monthly_tax: float = 0.0
    monthly_subsidy: float = 0.0
    
    # Environment
    pollution_level: float = 0.0
    environmental_score: float = 0.5
    
    @property
    def total_housing(self) -> int:
        return self.poor_housing + self.medium_housing + self.luxury_housing
    
    @property
    def density(self) -> float:
        """Housing density (0-1)"""
        max_cells = self.size * self.size
        return self.total_housing / max_cells if max_cells > 0 else 0


@dataclass 
class CityMetrics:
    """All city-wide metrics"""
    # Economy
    gdp: float = 0.0
    tax_revenue: float = 0.0
    developer_profit: float = 0.0
    developer_cumulative_profit: float = 0.0
    unemployment_rate: float = 0.05
    job_capacity: int = 0  # Total jobs the city can support
    employed: int = 0      # Currently employed residents
    
    # City Budget
    city_budget: float = 500000  # Starting budget $500k
    service_level: float = 0.5   # 0-1, affects QoL
    infrastructure_quality: float = 0.5  # 0-1, affects attractiveness
    
    # City Reputation - drives immigration
    reputation: float = 0.5  # 0-1, composite score
    regional_demand: float = 1000  # External population wanting to move
    
    # Environment  
    green_space_ratio: float = 0.0
    woodland_ratio: float = 0.0
    avg_pollution: float = 0.0
    sustainability_score: float = 0.5
    
    # Society
    population: int = 0
    segregation_index: float = 0.0
    avg_qol: float = 2.5
    avg_satisfaction: float = 3.0
    housing_vacancy_rate: float = 0.0
    affordability_index: float = 0.5
    
    # Housing
    total_housing: int = 0
    poor_housing: int = 0
    medium_housing: int = 0
    luxury_housing: int = 0
    
    # Population by class
    poor_pop: int = 0
    middle_pop: int = 0
    rich_pop: int = 0
    
    # Flow
    immigrants: int = 0
    emigrants: int = 0


@dataclass
class GovernmentConstraints:
    """Government policy constraints for development"""
    affordable_housing_pct: float = 0.25  # Min % poor housing
    green_space_pct: float = 0.15         # Min % green space
    woodland_preservation: float = 0.10    # Min % woodland
    max_density: float = 0.7              # Max housing density
    tax_rate: float = 0.02                # Property tax rate (realistic: 1-3%)
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.affordable_housing_pct,
            self.green_space_pct,
            self.woodland_preservation,
            self.max_density,
            (self.tax_rate + 0.2) / 0.4  # Normalize to 0-1
        ], dtype=np.float32)
    
    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'GovernmentConstraints':
        return cls(
            affordable_housing_pct=np.clip(vec[0], 0.15, 0.45),
            green_space_pct=np.clip(vec[1], 0.05, 0.35),
            woodland_preservation=np.clip(vec[2], 0.05, 0.25),
            max_density=np.clip(vec[3], 0.4, 0.85),
            tax_rate=np.clip(vec[4] * 0.4 - 0.2, -0.2, 0.2)
        )


# Land use types for spatial grid
LAND_USE_EMPTY = 0
LAND_USE_POOR = 1
LAND_USE_MEDIUM = 2
LAND_USE_LUXURY = 3
LAND_USE_GREEN = 4
LAND_USE_WOODLAND = 5

@dataclass
class DeveloperDecision:
    """Developer's SPATIAL land use decision - 20x20 grid per zone"""
    # Each cell is a land use type: 0=empty, 1=poor, 2=medium, 3=luxury, 4=green, 5=woodland
    land_use_grid: np.ndarray = None  # Shape: (20, 20)
    
    def __post_init__(self):
        if self.land_use_grid is None:
            # Default: balanced mix
            grid = np.zeros((20, 20), dtype=np.int32)
            for y in range(20):
                for x in range(20):
                    r = np.random.random()
                    if r < 0.3:
                        grid[y, x] = LAND_USE_POOR
                    elif r < 0.6:
                        grid[y, x] = LAND_USE_MEDIUM
                    elif r < 0.85:
                        grid[y, x] = LAND_USE_LUXURY
                    else:
                        grid[y, x] = LAND_USE_GREEN
            self.land_use_grid = grid
    
    def get_counts(self) -> dict:
        """Count each land use type"""
        return {
            'empty': int(np.sum(self.land_use_grid == LAND_USE_EMPTY)),
            'poor': int(np.sum(self.land_use_grid == LAND_USE_POOR)),
            'medium': int(np.sum(self.land_use_grid == LAND_USE_MEDIUM)),
            'luxury': int(np.sum(self.land_use_grid == LAND_USE_LUXURY)),
            'green': int(np.sum(self.land_use_grid == LAND_USE_GREEN)),
            'woodland': int(np.sum(self.land_use_grid == LAND_USE_WOODLAND))
        }
    
    @classmethod
    def from_grid(cls, grid: np.ndarray) -> 'DeveloperDecision':
        """Create from 20x20 grid"""
        return cls(land_use_grid=grid.astype(np.int32))


class UrbanSimulation:
    """
    Complete urban development simulation with three pillars of sustainability.
    
    This creates a realistic environment where:
    - Decisions have consequences
    - Population responds to conditions
    - Economy must remain viable
    - Environment affects quality of life
    - Win/lose conditions exist
    """
    
    # Economic parameters (set so that development can be profitable)
    # Construction costs low enough to make development viable
    # Woodland costs 2x green
    CONSTRUCTION_COST = {'poor': 20000, 'medium': 50000, 'luxury': 80000,
                         'green': 5000, 'woodland': 10000}
    # HIGH RENTS for clear profit signals
    MONTHLY_RENT = {'poor': 10000, 'medium': 25000, 'luxury': 60000}
    # Maintenance costs scaled to match the rents
    MAINTENANCE_COST = {'poor': 500, 'medium': 1000, 'luxury': 2000}
    
    # VACANCY PENALTY: Strong signal to match supply with demand precisely
    # -$10k/month per empty unit forces developer to build exactly what's needed
    VACANCY_PENALTY_PER_UNIT = 10000  # $10k/month per vacant unit
    
    MORTGAGE_YEARS = 30  # Construction financed over 30 years
    INTEREST_RATE = 0.03  # 3% annual interest (lower financing cost)
    
    # JOB CREATION per housing unit - economic engine!
    JOBS_PER_HOUSING = {'poor': 0.8, 'medium': 1.5, 'luxury': 3.0}
    
    # CITY BUDGET parameters
    SERVICE_COST_PER_CAPITA = 500  # Annual cost per resident
    INCOME_TAX_RATE = 0.03  # 3% income tax
    
    # REGIONAL MARKET - external population pool
    REGIONAL_POPULATION = 50000  # External pool that can immigrate
    BASE_REGIONAL_GROWTH = 0.02  # 2% annual growth in regional demand
    
    # Game thresholds (checked after turn 5)
    MIN_POPULATION = 200  # Low enough to give the city some runway
    MIN_GDP = 10_000_000  # $10M
    MAX_SEGREGATION = 0.75
    BANKRUPTCY_THRESHOLD = -2_000_000  # Losses of up to $2M are tolerated
    LOSE_CHECK_START_TURN = 6  # Lose conditions are checked from turn 6

    # Win thresholds
    WIN_MIN_POPULATION = 2000
    WIN_MAX_SEGREGATION = 0.15
    WIN_MIN_QOL = 3.0
    
    # QoL thresholds - people won't move in if QoL is below this
    # Thresholds low enough that people actually move in
    QOL_THRESHOLD = {'poor': 1.5, 'middle': 2.0, 'rich': 2.8}
    
    def __init__(self, map_size: int = 100, zone_size: int = 20):
        self.map_size = map_size
        self.zone_size = zone_size
        self.num_zones = 9
        
        # Initialize grids
        self.land_grid = np.zeros((map_size, map_size), dtype=np.int8)
        self.housing_grid = np.zeros((map_size, map_size), dtype=np.int8)
        self.pollution_grid = np.zeros((map_size, map_size), dtype=np.float32)
        
        # Initialize zones in 3x3 grid
        self.zones: List[Zone] = []
        spacing = (map_size - zone_size) // 2
        start_offset = spacing // 2
        
        zone_id = 0
        for row in range(3):
            for col in range(3):
                x = start_offset + col * (zone_size + 5)
                y = start_offset + row * (zone_size + 5)
                self.zones.append(Zone(id=zone_id, x=x, y=y, size=zone_size))
                zone_id += 1
        
        # Population
        self.households: List[Household] = []
        self.next_household_id = 0
        
        # Game state
        self.current_turn = 0
        self.max_turns = 9
        self.game_state = GameState.RUNNING
        self.game_over_reason = ""
        
        # Metrics history
        self.metrics = CityMetrics()
        self.metrics_history: List[CityMetrics] = []
        
        # Economic tracking
        self.cumulative_developer_profit = 0.0
        self.cumulative_tax_revenue = 0.0
        
        # Generate terrain outside zones
        self._generate_terrain()
        
        # No seed population - people only come after housing is built
        # self._seed_population(2000)
        
        # Migration tracking
        self.migration_log: List[Dict] = []
    
    def _generate_terrain(self):
        """Generate random terrain outside development zones"""
        for y in range(self.map_size):
            for x in range(self.map_size):
                if not self._is_in_any_zone(x, y):
                    r = random.random()
                    if r < 0.05:
                        self.land_grid[y, x] = 7  # Water
                    elif r < 0.15:
                        self.land_grid[y, x] = 6  # Woodland
                    elif r < 0.25:
                        self.land_grid[y, x] = 5  # Grass
                    # else stays 0 (undeveloped)
    
    def _is_in_any_zone(self, x: int, y: int) -> bool:
        for zone in self.zones:
            if zone.x <= x < zone.x + zone.size and zone.y <= y < zone.y + zone.size:
                return True
        return False
    
    def _get_zone_at(self, x: int, y: int) -> Optional[Zone]:
        for zone in self.zones:
            if zone.x <= x < zone.x + zone.size and zone.y <= y < zone.y + zone.size:
                return zone
        return None
    
    def _get_neighboring_zones(self, zone: Zone) -> List[Zone]:
        """Get zones adjacent to this zone (share border or corner)"""
        if zone is None:
            return []
        
        neighbors = []
        zone_cx = zone.x + zone.size // 2
        zone_cy = zone.y + zone.size // 2
        
        for other in self.zones:
            if other.id == zone.id:
                continue
            
            other_cx = other.x + other.size // 2
            other_cy = other.y + other.size // 2
            
            # Adjacent if within ~1.5 zone widths
            dist = ((zone_cx - other_cx)**2 + (zone_cy - other_cy)**2)**0.5
            if dist < zone.size * 1.8:
                neighbors.append(other)
        
        return neighbors
    
    def _seed_population(self, count: int):
        """Add initial population spread across early zones"""
        # Distribute: 40% poor, 40% middle, 20% rich
        for _ in range(count):
            r = random.random()
            if r < 0.4:
                wc = WealthClass.POOR
            elif r < 0.8:
                wc = WealthClass.MIDDLE
            else:
                wc = WealthClass.RICH
            
            # Random location in map
            x = random.randint(0, self.map_size - 1)
            y = random.randint(0, self.map_size - 1)
            
            hh = Household(
                id=self.next_household_id,
                wealth_class=wc,
                location=(y, x),
                satisfaction=3.0
            )
            self.households.append(hh)
            self.next_household_id += 1
    
    def develop_zone(
        self, 
        zone_id: int, 
        constraints: GovernmentConstraints,
        decision: DeveloperDecision
    ) -> Dict:
        """
        Develop a zone with given constraints and developer decision.
        Returns detailed results including profit/loss.
        """
        if zone_id >= len(self.zones):
            return {'error': 'Invalid zone'}
        
        zone = self.zones[zone_id]
        if zone.developed:
            return {'error': 'Zone already developed'}
        
        zone_area = zone.size * zone.size
        
        # === SPATIAL GRID PROCESSING ===
        # Developer provides 20x20 grid of land use decisions
        grid = decision.land_use_grid
        counts = decision.get_counts()
        
        # Get raw counts from developer's spatial decision
        raw_poor = counts['poor']
        raw_medium = counts['medium']
        raw_luxury = counts['luxury']
        raw_green = counts['green']
        raw_woodland = counts['woodland']
        raw_housing = raw_poor + raw_medium + raw_luxury
        
        # Apply government constraints
        min_green = int(zone_area * constraints.green_space_pct)
        min_woodland = int(zone_area * constraints.woodland_preservation)
        max_housing = int(zone_area * constraints.max_density)
        
        # Enforce green space and woodland minima, then clamp the SUM so that
        # green_area + woodland_area never exceeds zone_area (otherwise the
        # minima could overflow the zone and give green_space_ratio > 1.0).
        green_area = max(raw_green, min_green)
        woodland_area = max(raw_woodland, min_woodland)
        if green_area + woodland_area > zone_area:
            # Apportion the available zone area between green and woodland
            # in their requested proportions, preserving constraint satisfaction
            # where possible.
            total = green_area + woodland_area
            green_area = int(zone_area * green_area / total)
            woodland_area = zone_area - green_area
        # Calculate actual housing (limited by density and after green/woodland)
        available_for_housing = zone_area - green_area - woodland_area
        total_housing = min(raw_housing, max_housing, available_for_housing)
        
        # Scale housing proportionally if needed
        if raw_housing > 0 and total_housing < raw_housing:
            scale = total_housing / raw_housing
            poor_units = int(raw_poor * scale)
            medium_units = int(raw_medium * scale)
            # Ensure luxury_units is never negative (rounding can cause issues)
            luxury_units = max(0, total_housing - poor_units - medium_units)
        else:
            poor_units = max(0, raw_poor)
            medium_units = max(0, raw_medium)
            luxury_units = max(0, raw_luxury)
        
        # Enforce affordable housing constraint
        min_affordable = int(total_housing * constraints.affordable_housing_pct)
        if poor_units < min_affordable:
            deficit = min_affordable - poor_units
            # Take from luxury first
            if luxury_units >= deficit:
                luxury_units -= deficit
                poor_units += deficit
            else:
                poor_units += luxury_units
                deficit -= luxury_units
                luxury_units = 0
                medium_units = max(0, medium_units - deficit)
                poor_units = min_affordable
        
        # Final safety clamp - ensure no negative values
        poor_units = max(0, poor_units)
        medium_units = max(0, medium_units)
        luxury_units = max(0, luxury_units)
        
        # === CONSTRAINT VIOLATION PENALTIES ===
        # Developer receives penalties for trying to violate constraints
        # (even though constraints are enforced, developer loses money for non-compliance)
        violation_penalty = 0.0
        violations = {}
        
        # Green space violation: developer tried to build less than required
        if raw_green < min_green:
            green_violation = (min_green - raw_green) / zone_area
            violation_penalty += 50000 * green_violation  # $50k per 1% violation
            violations['green_space'] = green_violation
        
        # Woodland violation
        if raw_woodland < min_woodland:
            woodland_violation = (min_woodland - raw_woodland) / zone_area
            violation_penalty += 50000 * woodland_violation
            violations['woodland'] = woodland_violation
        
        # Density violation: developer tried to build more than allowed
        if raw_housing > max_housing:
            density_violation = (raw_housing - max_housing) / zone_area
            violation_penalty += 30000 * density_violation  # $30k per 1% over-density
            violations['density'] = density_violation
        
        # Affordable housing violation
        raw_poor_pct = raw_poor / max(1, raw_housing) if raw_housing > 0 else 0
        if raw_poor_pct < constraints.affordable_housing_pct:
            affordable_violation = constraints.affordable_housing_pct - raw_poor_pct
            violation_penalty += 80000 * affordable_violation  # $80k per 1% under affordable
            violations['affordable_housing'] = affordable_violation
        
        # Calculate effective percentages for logging
        total = max(1, poor_units + medium_units + luxury_units)
        effective_poor_pct = poor_units / total
        effective_medium_pct = medium_units / total
        effective_luxury_pct = luxury_units / total
        
        # Store in zone (ensure non-negative)
        zone.poor_housing = max(0, poor_units)
        zone.medium_housing = max(0, medium_units)
        zone.luxury_housing = max(0, luxury_units)
        zone.green_space = green_area
        zone.woodland = woodland_area
        
        # Calculate construction costs (fixed costs)
        construction_cost = (
            poor_units * self.CONSTRUCTION_COST['poor'] +
            medium_units * self.CONSTRUCTION_COST['medium'] +
            luxury_units * self.CONSTRUCTION_COST['luxury']
        )
        
        # Monthly mortgage payment (construction financed over MORTGAGE_YEARS)
        monthly_rate = self.INTEREST_RATE / 12
        num_payments = self.MORTGAGE_YEARS * 12
        if construction_cost > 0 and monthly_rate > 0:
            monthly_mortgage = construction_cost * (
                monthly_rate * (1 + monthly_rate)**num_payments
            ) / ((1 + monthly_rate)**num_payments - 1)
        else:
            monthly_mortgage = 0
        
        # Fixed costs (must pay even if vacant!)
        monthly_maintenance = (
            poor_units * self.MAINTENANCE_COST['poor'] +
            medium_units * self.MAINTENANCE_COST['medium'] +
            luxury_units * self.MAINTENANCE_COST['luxury']
        )
        
        # Property tax (annual)
        property_value = construction_cost * 1.2
        annual_tax = property_value * max(0, constraints.tax_rate)
        monthly_tax = annual_tax / 12
        
        # Government subsidy if negative tax rate
        monthly_subsidy = 0
        if constraints.tax_rate < 0:
            monthly_subsidy = property_value * abs(constraints.tax_rate) / 12
        
        # Store zone data
        zone.development_cost = construction_cost
        zone.property_value = property_value
        zone.monthly_mortgage = monthly_mortgage
        zone.monthly_maintenance = monthly_maintenance
        zone.monthly_tax = monthly_tax
        zone.monthly_subsidy = monthly_subsidy
        zone.developed = True
        
        # Place housing in grid BEFORE immigration
        # Pass the developer's spatial grid for exact pixel-level placement
        self._place_zone_in_grid(zone, constraints, spatial_grid=decision.land_use_grid)
        
        # Now attract population - profit depends on ACTUAL occupancy!
        immigrants, turned_away = self._process_immigration(zone)
        
        # Penalty for turning away potential residents (lost revenue opportunity)
        # weighted_turned_away = sum of annual rent of all turned-away residents
        turned_away_penalty = getattr(self, 'last_weighted_turned_away', 0)  # Already in dollars
        
        # Calculate ACTUAL occupancy and revenue AFTER immigration
        # Count actual household objects in housing
        actual_poor_occupied = self._count_occupied_housing(zone, 1)
        actual_medium_occupied = self._count_occupied_housing(zone, 2)
        actual_luxury_occupied = self._count_occupied_housing(zone, 3)
        
        actual_revenue = (
            actual_poor_occupied * self.MONTHLY_RENT['poor'] +
            actual_medium_occupied * self.MONTHLY_RENT['medium'] +
            actual_luxury_occupied * self.MONTHLY_RENT['luxury']
        )
        
        # Net monthly profit = ACTUAL Revenue - Fixed Costs
        monthly_profit = actual_revenue - monthly_mortgage - monthly_maintenance - monthly_tax + monthly_subsidy
        
        # First year profit (can be NEGATIVE if units are vacant!)
        first_year_profit = monthly_profit * 12
        
        # Apply constraint violation penalty (developer pays for non-compliance)
        first_year_profit -= violation_penalty
        
        # Apply turned away penalty (lost revenue from unhoused demand)
        first_year_profit -= turned_away_penalty
        
        zone.monthly_revenue = actual_revenue
        
        # Track occupancy for display
        total_units = poor_units + medium_units + luxury_units
        occupied_units = actual_poor_occupied + actual_medium_occupied + actual_luxury_occupied
        vacant_units = total_units - occupied_units
        occupancy_rate = occupied_units / total_units if total_units > 0 else 0
        
        # VACANCY PENALTY: Strong signal to match supply with demand precisely
        # Each empty unit costs $10k/month (forces developer to build exactly what's needed)
        monthly_vacancy_penalty = vacant_units * self.VACANCY_PENALTY_PER_UNIT
        first_year_profit -= monthly_vacancy_penalty * 12  # Apply to first year profit
        
        # === REGRET CALCULATION ===
        # What COULD have been earned with optimal housing mix?
        # Optimal = maximize revenue given demand, assuming 100% occupancy
        max_housing = int(zone.size * zone.size * constraints.max_density)
        
        # Calculate demand by class (how many are waiting to move in)
        poor_demand = len([h for h in self.households if h.wealth_class == WealthClass.POOR]) + 500
        middle_demand = len([h for h in self.households if h.wealth_class == WealthClass.MIDDLE]) + 300
        rich_demand = len([h for h in self.households if h.wealth_class == WealthClass.RICH]) + 100
        
        # Optimal allocation: prioritize luxury (highest ROI), then medium, then poor
        # But can't exceed demand for each class
        optimal_luxury = min(rich_demand, max_housing)
        remaining = max_housing - optimal_luxury
        optimal_medium = min(middle_demand, remaining)
        remaining -= optimal_medium
        optimal_poor = min(poor_demand, remaining)
        
        # Optimal revenue (assuming 100% occupancy)
        optimal_monthly_revenue = (
            optimal_poor * self.MONTHLY_RENT['poor'] +
            optimal_medium * self.MONTHLY_RENT['medium'] +
            optimal_luxury * self.MONTHLY_RENT['luxury']
        )
        
        # Optimal costs
        optimal_construction = (
            optimal_poor * self.CONSTRUCTION_COST['poor'] +
            optimal_medium * self.CONSTRUCTION_COST['medium'] +
            optimal_luxury * self.CONSTRUCTION_COST['luxury']
        )
        if optimal_construction > 0:
            optimal_mortgage = optimal_construction * (
                monthly_rate * (1 + monthly_rate)**num_payments
            ) / ((1 + monthly_rate)**num_payments - 1)
        else:
            optimal_mortgage = 0
        optimal_maintenance = (
            optimal_poor * self.MAINTENANCE_COST['poor'] +
            optimal_medium * self.MAINTENANCE_COST['medium'] +
            optimal_luxury * self.MAINTENANCE_COST['luxury']
        )
        optimal_monthly_profit = optimal_monthly_revenue - optimal_mortgage - optimal_maintenance
        optimal_annual_profit = optimal_monthly_profit * 12
        
        # REGRET = what we missed
        regret = optimal_annual_profit - first_year_profit
        regret_pct = (regret / optimal_annual_profit * 100) if optimal_annual_profit > 0 else 0
        
        # Update cumulative profit (can go negative!)
        self.cumulative_developer_profit += first_year_profit
        self.cumulative_tax_revenue += annual_tax
        
        # Process migration and emigration
        emigrants, internal_moves, migration_details = self._process_population_dynamics()
        
        # Log migration
        self.migration_log.append({
            'turn': self.current_turn,
            'zone_id': zone_id,
            'immigrants': immigrants,
            'emigrants': emigrants + turned_away,  # Include turned away
            'turned_away': turned_away,
            'internal_moves': internal_moves,
            'details': migration_details
        })
        
        # Update all metrics
        self._update_metrics()
        
        # Check game state (lose-only; keep learning pressure)
        self._check_game_state()
        
        self.current_turn += 1
        
        return {
            'zone_id': zone_id,
            'housing_built': {
                'poor': poor_units,
                'medium': medium_units,
                'luxury': luxury_units
            },
            'housing_occupied': {
                'poor': actual_poor_occupied,
                'medium': actual_medium_occupied,
                'luxury': actual_luxury_occupied
            },
            'occupancy_rate': occupancy_rate,
            'construction_cost': construction_cost,
            'monthly_revenue': actual_revenue,
            'monthly_costs': monthly_mortgage + monthly_maintenance + monthly_tax,
            'monthly_profit': monthly_profit,
            'first_year_profit': first_year_profit,
            'cumulative_profit': self.cumulative_developer_profit,
            'developer_profitable': first_year_profit > 0,
            'regret': {
                'optimal_mix': {'poor': optimal_poor, 'medium': optimal_medium, 'luxury': optimal_luxury},
                'optimal_profit': optimal_annual_profit,
                'actual_profit': first_year_profit,
                'missed_profit': regret,
                'regret_pct': regret_pct
            },
            'immigrants': immigrants,
            'emigrants': emigrants + turned_away,  # Include turned away as emigrants
            'turned_away': turned_away,  # People who couldn't find housing
            'turned_away_penalty': turned_away_penalty,
            'internal_moves': internal_moves,
            'migration_details': migration_details,
            'government_policy': {
                'affordable_housing_requirement': constraints.affordable_housing_pct,
                'affordable_housing_actual': effective_poor_pct,
                'green_space_requirement': constraints.green_space_pct,
                'woodland_preservation': constraints.woodland_preservation,
                'max_density_allowed': constraints.max_density,
                'tax_rate': constraints.tax_rate,
                'is_subsidy': constraints.tax_rate < 0,
                'subsidy_amount': monthly_subsidy * 12 if constraints.tax_rate < 0 else 0
            },
            'constraint_violations': violations,
            'violation_penalty': violation_penalty,
            'population_after': len(self.households),
            'city_metrics': {
                'reputation': getattr(self.metrics, 'reputation', 0.5),
                'service_level': getattr(self.metrics, 'service_level', 0.5),
                'unemployment_rate': getattr(self.metrics, 'unemployment_rate', 0),
                'job_capacity': getattr(self.metrics, 'job_capacity', 0)
            }
        }
    
    def _place_zone_in_grid(self, zone: Zone, constraints: GovernmentConstraints, spatial_grid: np.ndarray = None):
        """
        Place zone content in the grid.
        If spatial_grid is provided, use developer's exact pixel decisions.
        Otherwise fall back to random placement based on counts.
        """
        if spatial_grid is not None:
            # USE DEVELOPER'S EXACT SPATIAL GRID
            for dy in range(zone.size):
                for dx in range(zone.size):
                    y, x = zone.y + dy, zone.x + dx
                    if y < self.map_size and x < self.map_size:
                        cell_type = spatial_grid[dy, dx]
                        
                        # Map developer decision to land/housing grids
                        # 0=empty, 1=poor, 2=medium, 3=luxury, 4=green, 5=woodland
                        if cell_type == LAND_USE_POOR:
                            self.land_grid[y, x] = 1  # Developed
                            self.housing_grid[y, x] = 1  # Poor housing
                        elif cell_type == LAND_USE_MEDIUM:
                            self.land_grid[y, x] = 1  # Developed
                            self.housing_grid[y, x] = 2  # Medium housing
                        elif cell_type == LAND_USE_LUXURY:
                            self.land_grid[y, x] = 1  # Developed
                            self.housing_grid[y, x] = 3  # Luxury housing
                        elif cell_type == LAND_USE_GREEN:
                            self.land_grid[y, x] = 5  # Green space
                        elif cell_type == LAND_USE_WOODLAND:
                            self.land_grid[y, x] = 6  # Woodland
                        # else: leave as empty (0)
            return
        
        # FALLBACK: Random placement based on zone counts (no spatial grid given)
        cells = []
        for dy in range(zone.size):
            for dx in range(zone.size):
                y, x = zone.y + dy, zone.x + dx
                if y < self.map_size and x < self.map_size:
                    cells.append((y, x))
        
        random.shuffle(cells)
        idx = 0
        
        # Place green space
        for _ in range(zone.green_space):
            if idx < len(cells):
                y, x = cells[idx]
                self.land_grid[y, x] = 5
                idx += 1
        
        # Place woodland
        for _ in range(zone.woodland):
            if idx < len(cells):
                y, x = cells[idx]
                self.land_grid[y, x] = 6
                idx += 1
        
        # Place housing
        for _ in range(zone.poor_housing):
            if idx < len(cells):
                y, x = cells[idx]
                self.land_grid[y, x] = 1
                self.housing_grid[y, x] = 1
                idx += 1
        
        for _ in range(zone.medium_housing):
            if idx < len(cells):
                y, x = cells[idx]
                self.land_grid[y, x] = 1
                self.housing_grid[y, x] = 2
                idx += 1
        
        for _ in range(zone.luxury_housing):
            if idx < len(cells):
                y, x = cells[idx]
                self.land_grid[y, x] = 1
                self.housing_grid[y, x] = 3
                idx += 1
    
    def _process_immigration(self, new_zone: Zone) -> Tuple[int, int]:
        """
        SOPHISTICATED IMMIGRATION with QoL-dependent demand.
        
        Each wealth class has different requirements:
        - POOR: Less picky, will come if basic QoL threshold met
        - MIDDLE: Need decent QoL AND job security (unemployment < 15%)
        - RICH: Very picky - need HIGH QoL + reputation > 0.40 + unemployment < 10%
        
        This creates meaningful differentiation - random policies can't just build 
        anything and expect it to fill!
        
        Returns: (immigrants_placed, turned_away)
        """
        immigrants = 0
        turned_away = 0
        
        # Get zone center for QoL calculation
        center_y = new_zone.y + new_zone.size // 2
        center_x = new_zone.x + new_zone.size // 2
        
        # City-wide factors
        reputation = getattr(self.metrics, 'reputation', 0.5)
        unemployment = getattr(self.metrics, 'unemployment_rate', 0.05)
        base_fill_rate = 0.6 + (reputation * 0.25) + ((1 - unemployment) * 0.15)
        
        # ========== POOR HOUSEHOLDS ==========
        # Less picky, will come if housing is affordable and basic QoL met
        poor_qol = self._calculate_qol(center_y, center_x, WealthClass.POOR)
        
        if poor_qol >= self.QOL_THRESHOLD.get('poor', 1.5):
            # Good conditions - strong demand
            qol_modifier = min(1.2, max(0.5, poor_qol / 3.0))
            fill_rate = base_fill_rate * qol_modifier
            target_poor = int(new_zone.poor_housing * fill_rate)
        else:
            # Poor conditions - still some desperate demand
            target_poor = int(new_zone.poor_housing * 0.3)
        
        for _ in range(target_poor):
            if self._place_immigrant_in_zone(new_zone, WealthClass.POOR, 1):
                immigrants += 1
            else:
                turned_away += 1
        
        # ========== MIDDLE CLASS ==========
        # Need decent QoL AND job security
        middle_qol = self._calculate_qol(center_y, center_x, WealthClass.MIDDLE)
        
        if middle_qol >= self.QOL_THRESHOLD.get('middle', 2.0) and unemployment < 0.15:
            # Good conditions for middle class
            qol_modifier = min(1.1, max(0.4, (middle_qol - 1.0) / 2.5))
            job_modifier = 1.0 - unemployment
            fill_rate = base_fill_rate * qol_modifier * job_modifier
            target_medium = int(new_zone.medium_housing * fill_rate)
        elif middle_qol >= self.QOL_THRESHOLD.get('middle', 2.0) * 0.8:
            # Marginal conditions - some demand
            target_medium = int(new_zone.medium_housing * 0.2)
        else:
            # Bad conditions - almost no middle class demand
            target_medium = int(new_zone.medium_housing * 0.05)
        
        for _ in range(target_medium):
            if self._place_immigrant_in_zone(new_zone, WealthClass.MIDDLE, 2):
                immigrants += 1
            else:
                turned_away += 1
        
        # ========== RICH HOUSEHOLDS ==========
        # Very picky - need HIGH QoL + reputation + full employment
        rich_qol = self._calculate_qol(center_y, center_x, WealthClass.RICH)
        
        meets_rich_standards = (
            rich_qol >= self.QOL_THRESHOLD.get('rich', 3.0) and 
            reputation > 0.40 and 
            unemployment < 0.10
        )
        
        if meets_rich_standards:
            # Luxury conditions - strong rich demand
            qol_modifier = min(1.0, max(0.3, (rich_qol - 2.5) / 2.0))
            reputation_modifier = reputation * 1.5
            fill_rate = base_fill_rate * qol_modifier * reputation_modifier * 0.8
            target_luxury = int(new_zone.luxury_housing * fill_rate)
        elif rich_qol >= self.QOL_THRESHOLD.get('rich', 3.0) * 0.9:
            # Almost there - minimal rich demand
            target_luxury = int(new_zone.luxury_housing * 0.1)
        else:
            # Not meeting rich standards - NO rich demand
            target_luxury = 0
        
        for _ in range(target_luxury):
            if self._place_immigrant_in_zone(new_zone, WealthClass.RICH, 3):
                immigrants += 1
            else:
                turned_away += 1
        
        # Track totals
        self.total_immigrants = getattr(self, 'total_immigrants', 0) + immigrants
        self.total_emigrants = getattr(self, 'total_emigrants', 0) + turned_away
        
        # Weighted penalty for turning away (lost rent by class)
        # Rich = $60k/yr, Medium = $25k/yr, Poor = $10k/yr
        weighted_penalty = turned_away * 15000  # Average weighted penalty
        self.last_weighted_turned_away = weighted_penalty
        
        # Track cumulative turned away
        self.total_turned_away = getattr(self, 'total_turned_away', 0) + turned_away
        
        return immigrants, turned_away
    
    def _place_immigrant_in_zone(self, zone: Zone, wealth_class: WealthClass, housing_type: int) -> bool:
        """Try to place an immigrant in a zone. Returns True if successful."""
        # Find empty housing of the right type
        for dy in range(zone.size):
            for dx in range(zone.size):
                y, x = zone.y + dy, zone.x + dx
                if y < self.map_size and x < self.map_size:
                    if self.housing_grid[y, x] == housing_type:
                        # Check if occupied
                        occupied = any(h.location == (y, x) for h in self.households)
                        if not occupied:
                            # Place household with auto-incrementing id
                            new_id = len(self.households) + 1
                            hh = Household(
                                id=new_id,
                                wealth_class=wealth_class,
                                location=(y, x),
                                satisfaction=3.0  # Fixed satisfaction - no tracking
                            )
                            self.households.append(hh)
                            return True
        return False
    
    def _process_immigration_OLD(self, new_zone: Zone) -> int:
        """Alternative immigration model with QoL requirements (not called by the simulation)."""
        immigrants = 0
        center_y = new_zone.y + new_zone.size // 2
        center_x = new_zone.x + new_zone.size // 2
        reputation = getattr(self.metrics, 'reputation', 0.5)
        unemployment = getattr(self.metrics, 'unemployment_rate', 0.05)
        base_fill_rate = 0.7 + (reputation * 0.2) + ((1 - unemployment) * 0.1)
        POP_MULT = 1
        
        # POOR: Less picky, will come if housing is affordable
        poor_qol = self._calculate_qol(center_y, center_x, WealthClass.POOR)
        
        if poor_qol >= self.QOL_THRESHOLD['poor']:
            qol_modifier = min(1.2, max(0.5, poor_qol / 3.0))
            fill_rate = base_fill_rate * qol_modifier
            target = int(new_zone.poor_housing * fill_rate) * POP_MULT
            
            for _ in range(int(target * random.uniform(0.85, 1.15))):
                if self._add_household(WealthClass.POOR, new_zone):
                    immigrants += 1
        else:
            target = int(new_zone.poor_housing * 0.2) * POP_MULT
            for _ in range(target):
                if self._add_household(WealthClass.POOR, new_zone):
                    immigrants += 1
        
        # MIDDLE: Need decent QoL and job security
        middle_qol = self._calculate_qol(center_y, center_x, WealthClass.MIDDLE)
        
        if middle_qol >= self.QOL_THRESHOLD['middle'] and unemployment < 0.15:
            qol_modifier = min(1.1, max(0.4, (middle_qol - 1.0) / 2.5))
            job_modifier = 1.0 - unemployment
            fill_rate = base_fill_rate * qol_modifier * job_modifier
            target = int(new_zone.medium_housing * fill_rate) * POP_MULT
            
            for _ in range(int(target * random.uniform(0.8, 1.2))):
                if self._add_household(WealthClass.MIDDLE, new_zone):
                    immigrants += 1
        elif middle_qol >= self.QOL_THRESHOLD['middle'] * 0.8:
            target = int(new_zone.medium_housing * 0.15) * POP_MULT
            for _ in range(target):
                if self._add_household(WealthClass.MIDDLE, new_zone):
                    immigrants += 1
        
        # RICH: Very picky - need HIGH QoL + reputation + full employment
        rich_qol = self._calculate_qol(center_y, center_x, WealthClass.RICH)
        
        meets_rich_standards = (
            rich_qol >= self.QOL_THRESHOLD['rich'] and 
            reputation > 0.40 and 
            unemployment < 0.10
        )
        
        if meets_rich_standards:
            qol_modifier = min(1.0, max(0.3, (rich_qol - 2.5) / 2.0))
            reputation_modifier = reputation * 1.5
            fill_rate = base_fill_rate * qol_modifier * reputation_modifier * 0.8
            target = int(new_zone.luxury_housing * fill_rate) * POP_MULT
            
            for _ in range(int(target * random.uniform(0.7, 1.1))):
                if self._add_household(WealthClass.RICH, new_zone):
                    immigrants += 1
        
        return immigrants
    
    def _calculate_city_attractiveness(self) -> float:
        """Calculate overall city attractiveness (0-1)"""
        if not self.households:
            return 0.8
        
        avg_qol = np.mean([h.satisfaction for h in self.households])
        qol_factor = avg_qol / 5.0
        
        seg = self._calculate_segregation()
        seg_factor = 1.0 - seg
        
        green_ratio = self._calculate_green_ratio()
        green_factor = min(1.0, green_ratio * 5)
        
        vacancy = self._calculate_vacancy_rate()
        vacancy_factor = 1.0 - abs(vacancy - 0.1) * 2
        
        attractiveness = (
            0.4 * qol_factor +
            0.3 * seg_factor +
            0.2 * green_factor +
            0.1 * vacancy_factor
        )
        
        return np.clip(attractiveness, 0.1, 1.0)
    
    def _count_occupied_housing(self, zone: Zone, housing_type: int) -> int:
        """Count how many units of a housing type are occupied in a zone"""
        occupied = 0
        for dy in range(zone.size):
            for dx in range(zone.size):
                y, x = zone.y + dy, zone.x + dx
                if y < self.map_size and x < self.map_size:
                    if self.housing_grid[y, x] == housing_type:
                        occupants = sum(1 for h in self.households if h.location == (y, x))
                        if occupants > 0:
                            occupied += 1
        return occupied
    
    def _add_household(self, wealth_class: WealthClass, zone: Zone) -> bool:
        """Add a household to a zone. Returns True if successful, False if no space."""
        housing_type = wealth_class + 1
        
        for dy in range(zone.size):
            for dx in range(zone.size):
                y, x = zone.y + dy, zone.x + dx
                if y < self.map_size and x < self.map_size:
                    if self.housing_grid[y, x] == housing_type:
                        occupants = sum(1 for h in self.households if h.location == (y, x))
                        if occupants < 2:
                            hh = Household(
                                id=self.next_household_id,
                                wealth_class=wealth_class,
                                location=(y, x),
                                satisfaction=3.0 + random.uniform(-0.5, 0.5)
                            )
                            self.households.append(hh)
                            self.next_household_id += 1
                            return True
        return False
    
    def _process_population_dynamics(self) -> Tuple[int, int, Dict]:
        """
        Full population dynamics with emigration based on satisfaction.
        Households leave if their QoL is below threshold.
        """
        emigrants = 0
        internal_moves = 0
        emigrant_details = {'poor': 0, 'middle': 0, 'rich': 0}
        move_details = {'poor': 0, 'middle': 0, 'rich': 0}
        
        emigration_threshold = {
            WealthClass.POOR: 1.8,
            WealthClass.MIDDLE: 2.2,
            WealthClass.RICH: 2.8
        }
        
        for hh in list(self.households):
            y, x = hh.location
            qol = self._calculate_qol(y, x, hh.wealth_class)
            
            # Update satisfaction (smoothed)
            hh.satisfaction = 0.5 * hh.satisfaction + 0.5 * qol
            
            # Check if household leaves due to low satisfaction
            if hh.satisfaction < emigration_threshold.get(WealthClass(hh.wealth_class), 2.0):
                # Emigrate with probability proportional to dissatisfaction
                emigrate_prob = 0.1 * (emigration_threshold.get(WealthClass(hh.wealth_class), 2.0) - hh.satisfaction)
                if np.random.random() < emigrate_prob:
                    self.households.remove(hh)
                    emigrants += 1
                    class_name = ['poor', 'middle', 'rich'][hh.wealth_class]
                    emigrant_details[class_name] += 1
        
        return emigrants, internal_moves, {'emigrants_by_class': emigrant_details, 
                                            'moves_by_class': move_details}
    
    def _process_population_dynamics_OLD(self) -> Tuple[int, int, Dict]:
        """Alternative model with full emigration and internal migration (not called by the simulation)."""
        emigrants = 0
        internal_moves = 0
        emigrant_details = {'poor': 0, 'middle': 0, 'rich': 0}
        move_details = {'poor': 0, 'middle': 0, 'rich': 0}
        
        # EMIGRATION MULTIPLIER - 1x for fast training (match immigration)
        EMIG_MULT = 1
        
        emigration_threshold = {
            WealthClass.POOR: 1.8,
            WealthClass.MIDDLE: 2.2,
            WealthClass.RICH: 2.8
        }
        
        for hh in list(self.households):
            y, x = hh.location
            qol = self._calculate_qol(y, x, hh.wealth_class)
            
            hh.satisfaction = 0.5 * hh.satisfaction + 0.5 * qol
            
            housing_type = self.housing_grid[y, x] if y < self.map_size and x < self.map_size else 0
            expected_type = hh.wealth_class + 1
            
            if housing_type > 0 and housing_type != expected_type:
                if housing_type < expected_type:
                    hh.satisfaction -= 0.4
                elif housing_type > expected_type:
                    hh.satisfaction -= 0.6 + 0.3 * (housing_type - expected_type)
            
            zone = self._get_zone_at(x, y)
            if zone is None or not zone.developed:
                hh.satisfaction -= 0.5
            
            threshold = emigration_threshold[hh.wealth_class]
            
            if hh.satisfaction < threshold:
                hh.turns_dissatisfied += 1
                
                turns_before_leave = 1 if hh.wealth_class == WealthClass.RICH else 2
                
                if hh.turns_dissatisfied >= turns_before_leave or hh.satisfaction < 1.5:
                    self.households.remove(hh)
                    emigrants += 1
                    class_name = ['poor', 'middle', 'rich'][hh.wealth_class]
                    emigrant_details[class_name] += 1
                    
                    # Remove additional households of same class (EMIG_MULT effect)
                    extra_to_remove = min(EMIG_MULT - 1, len([h for h in self.households if h.wealth_class == hh.wealth_class]))
                    removed = 0
                    for other in list(self.households):
                        if other.wealth_class == hh.wealth_class and removed < extra_to_remove:
                            self.households.remove(other)
                            emigrants += 1
                            emigrant_details[class_name] += 1
                            removed += 1
                    continue
            else:
                hh.turns_dissatisfied = 0
            
            migration_chance = 0.1 if hh.satisfaction >= 3.0 else 0.4
            
            if random.random() < migration_chance:
                new_loc = self._find_better_location(hh)
                if new_loc and new_loc != hh.location:
                    hh.location = new_loc
                    internal_moves += 1
                    class_name = ['poor', 'middle', 'rich'][hh.wealth_class]
                    move_details[class_name] += 1
        
        migration_details = {
            'emigrants_by_class': emigrant_details,
            'moves_by_class': move_details
        }
        
        return emigrants, internal_moves, migration_details
    
    def _find_better_location(self, hh: Household) -> Optional[Tuple[int, int]]:
        """Find a better location for a household"""
        current_qol = self._calculate_qol(hh.location[0], hh.location[1], hh.wealth_class)
        best_loc = None
        best_qol = current_qol
        
        housing_type = hh.wealth_class + 1
        
        for zone in self.zones:
            if not zone.developed:
                continue
            
            for _ in range(10):
                dy = random.randint(0, zone.size - 1)
                dx = random.randint(0, zone.size - 1)
                y, x = zone.y + dy, zone.x + dx
                
                if y >= self.map_size or x >= self.map_size:
                    continue
                
                if self.housing_grid[y, x] == housing_type:
                    occupants = sum(1 for h in self.households if h.location == (y, x))
                    if occupants < 2:
                        qol = self._calculate_qol(y, x, hh.wealth_class)
                        if qol > best_qol:
                            best_qol = qol
                            best_loc = (y, x)
        
        return best_loc
    
    def _calculate_qol(self, y: int, x: int, wealth_class: WealthClass) -> float:
        """Calculate Quality of Life at EXACT pixel location with DISTANCE DECAY."""
        qol = 2.5
        
        # Distance-based green/woodland bonus (woodland stronger)
        green_bonus = 0.0
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.map_size and 0 <= nx < self.map_size:
                    if self.land_grid[ny, nx] in [5, 6]:
                        distance = max(1, (dy**2 + dx**2)**0.5)
                        decay = 1.0 / distance
                        if self.land_grid[ny, nx] == 6:
                            green_bonus += 0.30 * decay  # woodland double weight
                        else:
                            green_bonus += 0.15 * decay
        qol += min(2.0, green_bonus)
        
        # Distance-based water bonus
        water_bonus = 0.0
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.map_size and 0 <= nx < self.map_size:
                    if self.land_grid[ny, nx] == 7:
                        distance = max(1, (dy**2 + dx**2)**0.5)
                        water_bonus += 0.1 / distance
        qol += min(0.8, water_bonus)
        
        # Distance-based housing density penalty
        density_penalty = 0.0
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.map_size and 0 <= nx < self.map_size:
                    if self.housing_grid[ny, nx] > 0:
                        distance = max(1, (dy**2 + dx**2)**0.5)
                        density_penalty += 0.03 / distance
        qol -= min(1.0, density_penalty)
        
        # Neighboring zone spillover
        current_zone = self._get_zone_at(x, y)
        if current_zone:
            for neighbor in self._get_neighboring_zones(current_zone):
                if neighbor.developed:
                    if neighbor.density > 0.5:
                        spillover_penalty = (neighbor.density - 0.5) * 0.6
                        qol -= spillover_penalty
                    
                    if wealth_class == WealthClass.RICH:
                        poor_ratio = neighbor.poor_housing / max(1, neighbor.total_housing)
                        if poor_ratio > 0.5:
                            qol -= (poor_ratio - 0.5) * 0.8
                    
                    if wealth_class == WealthClass.POOR:
                        luxury_ratio = neighbor.luxury_housing / max(1, neighbor.total_housing)
                        if luxury_ratio > 0.4:
                            qol -= (luxury_ratio - 0.4) * 0.6
                    
                    neighbor_green = (neighbor.green_space + neighbor.woodland) / max(1, neighbor.size**2)
                    qol += neighbor_green * 0.3
        
        # Housing match bonus
        if y < self.map_size and x < self.map_size:
            housing_type = self.housing_grid[y, x]
            expected = wealth_class + 1
            if housing_type == expected:
                qol += 0.5
            elif housing_type > expected:
                qol -= 0.4 * (housing_type - expected)
            elif housing_type > 0:
                qol -= 0.3 * (expected - housing_type)
        
        # Current zone density penalty
        if current_zone and current_zone.density > 0.5:
            qol -= (current_zone.density - 0.5) * 1.2
        
        # New city bonus
        developed_count = sum(1 for z in self.zones if z.developed)
        if developed_count < 3:
            qol += 0.4
        
        return np.clip(qol, 0, 5)
    
    def _calculate_segregation(self) -> float:
        """
        Calculate multi-class segregation index.
        Uses average of pairwise dissimilarity indices between all 3 wealth classes.
        Also penalizes missing classes to prevent reward hacking.
        """
        if len(self.households) < 10:
            return 0.0
        
        poor_total = sum(1 for h in self.households if h.wealth_class == WealthClass.POOR)
        middle_total = sum(1 for h in self.households if h.wealth_class == WealthClass.MIDDLE)
        rich_total = sum(1 for h in self.households if h.wealth_class == WealthClass.RICH)
        
        total_pop = poor_total + middle_total + rich_total
        if total_pop == 0:
            return 0.0
        
        # PENALTY for missing classes (prevents reward hacking by avoiding luxury)
        # If any class is 0, add a STRONG penalty to segregation
        num_missing = (1 if poor_total == 0 else 0) + (1 if middle_total == 0 else 0) + (1 if rich_total == 0 else 0)
        missing_penalty = 0.5 * num_missing  # 0.5 per missing class - MUST build all 3 types!
        
        # Count per zone
        zone_poor = {z.id: 0 for z in self.zones}
        zone_middle = {z.id: 0 for z in self.zones}
        zone_rich = {z.id: 0 for z in self.zones}
        
        for hh in self.households:
            zone = self._get_zone_at(hh.location[1], hh.location[0])
            if zone:
                if hh.wealth_class == WealthClass.POOR:
                    zone_poor[zone.id] += 1
                elif hh.wealth_class == WealthClass.MIDDLE:
                    zone_middle[zone.id] += 1
                elif hh.wealth_class == WealthClass.RICH:
                    zone_rich[zone.id] += 1
        
        # Calculate pairwise dissimilarity indices
        D_poor_middle = 0.0
        D_poor_rich = 0.0
        D_middle_rich = 0.0
        
        if poor_total > 0 and middle_total > 0:
            D_poor_middle = 0.5 * sum(
                abs(zone_poor[z.id]/poor_total - zone_middle[z.id]/middle_total)
                for z in self.zones
            )
        
        if poor_total > 0 and rich_total > 0:
            D_poor_rich = 0.5 * sum(
            abs(zone_poor[z.id]/poor_total - zone_rich[z.id]/rich_total)
            for z in self.zones
        )
        
        if middle_total > 0 and rich_total > 0:
            D_middle_rich = 0.5 * sum(
                abs(zone_middle[z.id]/middle_total - zone_rich[z.id]/rich_total)
                for z in self.zones
            )
        
        # Average of valid pairwise indices
        num_valid = (1 if poor_total > 0 and middle_total > 0 else 0) + \
                    (1 if poor_total > 0 and rich_total > 0 else 0) + \
                    (1 if middle_total > 0 and rich_total > 0 else 0)
        
        if num_valid > 0:
            D = (D_poor_middle + D_poor_rich + D_middle_rich) / num_valid
        else:
            D = 0.0
        
        # Add missing class penalty
        D = min(1.0, D + missing_penalty)
        
        return float(D)
    
    def get_zone_household_distributions(self) -> list:
        """
        Get household wealth class distribution per zone.
        Returns list of dicts: [{'poor': X, 'middle': Y, 'rich': Z}, ...]
        This matches how segregation is calculated.
        """
        distributions = []
        for zone in self.zones:
            if zone.developed:
                zone_poor = 0
                zone_middle = 0
                zone_rich = 0
                for hh in self.households:
                    hz = self._get_zone_at(hh.location[1], hh.location[0])
                    if hz and hz.id == zone.id:
                        if hh.wealth_class == WealthClass.POOR:
                            zone_poor += 1
                        elif hh.wealth_class == WealthClass.MIDDLE:
                            zone_middle += 1
                        elif hh.wealth_class == WealthClass.RICH:
                            zone_rich += 1
                distributions.append({
                    'poor': zone_poor,
                    'middle': zone_middle,
                    'rich': zone_rich
                })
            else:
                distributions.append({'poor': 0, 'middle': 0, 'rich': 0})
        return distributions
    
    def _calculate_green_ratio(self) -> float:
        """Ratio of GREEN-only space to total developed area.

        Woodland is not counted here: it is reported separately by
        ``_calculate_woodland_ratio``, so that the two environment-pillar
        metrics measure distinct quantities.
        """
        total_developed = 0
        total_green = 0
        for zone in self.zones:
            if zone.developed:
                total_developed += zone.size * zone.size
                total_green += zone.green_space
        if total_developed == 0:
            return 0.0
        return total_green / total_developed

    def _calculate_woodland_ratio(self) -> float:
        """Ratio of WOODLAND cells to total developed area.

        Assigned to ``MetricsState.woodland_ratio`` in ``_update_metrics``.
        """
        total_developed = 0
        total_woodland = 0
        for zone in self.zones:
            if zone.developed:
                total_developed += zone.size * zone.size
                total_woodland += zone.woodland
        if total_developed == 0:
            return 0.0
        return total_woodland / total_developed
    
    def _calculate_vacancy_rate(self) -> float:
        """Calculate housing vacancy rate"""
        total_housing = sum(z.total_housing for z in self.zones)
        if total_housing == 0:
            return 1.0
        
        occupied = len(self.households)
        return max(0, (total_housing - occupied) / max(1, total_housing))
    
    def _update_metrics(self):
        """Update all city metrics including economic engine"""
        m = self.metrics
        
        # Population - use zone-based counts (fast mode) or household list
        if hasattr(self, 'total_poor_pop'):
            # Fast training mode - use zone population counts
            m.poor_pop = getattr(self, 'total_poor_pop', 0)
            m.middle_pop = getattr(self, 'total_medium_pop', 0)
            m.rich_pop = getattr(self, 'total_luxury_pop', 0)
            m.population = m.poor_pop + m.middle_pop + m.rich_pop
        else:
            # Household mode - use household list
            m.population = len(self.households)
            m.poor_pop = sum(1 for h in self.households if h.wealth_class == WealthClass.POOR)
            m.middle_pop = sum(1 for h in self.households if h.wealth_class == WealthClass.MIDDLE)
            m.rich_pop = sum(1 for h in self.households if h.wealth_class == WealthClass.RICH)
        
        # Housing
        m.poor_housing = sum(z.poor_housing for z in self.zones)
        m.medium_housing = sum(z.medium_housing for z in self.zones)
        m.luxury_housing = sum(z.luxury_housing for z in self.zones)
        m.total_housing = m.poor_housing + m.medium_housing + m.luxury_housing
        m.housing_vacancy_rate = self._calculate_vacancy_rate()
        
        # Economic engine
        m.job_capacity = int(
            m.poor_housing * self.JOBS_PER_HOUSING['poor'] +
            m.medium_housing * self.JOBS_PER_HOUSING['medium'] +
            m.luxury_housing * self.JOBS_PER_HOUSING['luxury']
        )
        
        working_age_pop = int(m.population * 0.65)
        m.employed = min(working_age_pop, m.job_capacity)
        m.unemployment_rate = 1 - (m.employed / working_age_pop) if working_age_pop > 0 else 0.5
        
        # GDP calculation
        if hasattr(self, 'total_poor_pop'):
            # Fast mode - estimate income from population mix
            avg_income_poor = 30000
            avg_income_middle = 60000
            avg_income_rich = 120000
            total_income = (
                m.poor_pop * avg_income_poor +
                m.middle_pop * avg_income_middle +
                m.rich_pop * avg_income_rich
            )
        else:
            total_income = sum(h.income for h in self.households)
        
        productivity_multiplier = 1.5 if m.unemployment_rate < 0.1 else 1.2
        m.gdp = total_income * productivity_multiplier * (1 - m.unemployment_rate * 0.5)
        
        property_tax_revenue = self.cumulative_tax_revenue / max(1, self.current_turn)
        income_tax_revenue = total_income * self.INCOME_TAX_RATE
        annual_revenue = property_tax_revenue + income_tax_revenue
        
        service_costs = m.population * self.SERVICE_COST_PER_CAPITA
        annual_expenses = service_costs
        
        budget_surplus = annual_revenue - annual_expenses
        m.city_budget = max(0, m.city_budget + budget_surplus / 12)
        
        if m.population > 0:
            budget_per_capita = m.city_budget / m.population
            m.service_level = min(1.0, budget_per_capita / 5000)
        else:
            m.service_level = 0.5
        
        if m.city_budget > 100000:
            m.infrastructure_quality = min(1.0, m.infrastructure_quality + 0.02)
        else:
            m.infrastructure_quality = max(0.2, m.infrastructure_quality - 0.01)
        
        # City Reputation
        qol_factor = m.avg_qol / 5.0 if m.avg_qol > 0 else 0.5
        job_factor = 1 - m.unemployment_rate
        env_factor = m.sustainability_score if hasattr(m, 'sustainability_score') else 0.5
        service_factor = m.service_level
        seg_factor = 1 - m.segregation_index
        
        m.reputation = (
            0.25 * qol_factor +
            0.25 * job_factor +
            0.20 * service_factor +
            0.15 * env_factor +
            0.15 * seg_factor
        )
        
        base_demand = self.REGIONAL_POPULATION * (1 + self.BASE_REGIONAL_GROWTH) ** self.current_turn
        m.regional_demand = base_demand * m.reputation
        
        # Environment
        m.green_space_ratio = self._calculate_green_ratio()
        m.woodland_ratio = self._calculate_woodland_ratio()
        m.avg_pollution = np.mean(self.pollution_grid)
        # sustainability_score blends green and woodland (each on its own
        # [0,1] scale, capped) with the pollution complement.
        m.sustainability_score = (
            min(1.0, m.green_space_ratio) +
            min(1.0, m.woodland_ratio) +
            (1.0 - m.avg_pollution)
        ) / 3.0
        
        # Society
        m.segregation_index = self._calculate_segregation()
        m.avg_qol = self._calculate_city_qol()
        m.avg_satisfaction = m.avg_qol
        
        if m.poor_pop > 0:
            m.affordability_index = min(1.0, m.poor_housing / m.poor_pop)
        else:
            m.affordability_index = 1.0
        
        m.developer_profit = self.cumulative_developer_profit
        m.developer_cumulative_profit = self.cumulative_developer_profit
        m.tax_revenue = self.cumulative_tax_revenue
        
        self.metrics_history.append(CityMetrics(
            gdp=m.gdp,
            population=m.population,
            segregation_index=m.segregation_index,
            avg_qol=m.avg_qol,
            developer_profit=m.developer_profit,
            job_capacity=m.job_capacity,
            unemployment_rate=m.unemployment_rate,
            reputation=m.reputation,
            regional_demand=m.regional_demand
        ))
    
    def _calculate_city_qol(self) -> float:
        """Calculate average QoL including service level and infrastructure"""
        if not self.households:
            return 2.5
        
        base_qol = np.mean([h.satisfaction for h in self.households])
        
        service_bonus = self.metrics.service_level * 0.5
        infra_bonus = self.metrics.infrastructure_quality * 0.3
        unemployment_penalty = self.metrics.unemployment_rate * 1.0
        
        return max(0, base_qol + service_bonus + infra_bonus - unemployment_penalty)
    
    def _check_game_state(self):
        """Check for lose conditions only (win disabled to keep training pressure)"""
        m = self.metrics
        
        if self.current_turn >= self.LOSE_CHECK_START_TURN:
            if m.population < self.MIN_POPULATION:
                self.game_state = GameState.LOSE_POPULATION
                self.game_over_reason = f"Population exodus! Only {m.population} remain (need {self.MIN_POPULATION})"
                return
            
            if m.gdp < self.MIN_GDP:
                self.game_state = GameState.LOSE_ECONOMY
                self.game_over_reason = f"Economic collapse! GDP ${m.gdp:,.0f} (need ${self.MIN_GDP:,.0f})"
                return
            
            if m.segregation_index > self.MAX_SEGREGATION:
                self.game_state = GameState.LOSE_SEGREGATION
                self.game_over_reason = f"Social breakdown! Segregation {m.segregation_index:.2f} (max {self.MAX_SEGREGATION})"
                return
            
            if self.cumulative_developer_profit < self.BANKRUPTCY_THRESHOLD:
                self.game_state = GameState.LOSE_BANKRUPTCY
                self.game_over_reason = f"Developer bankruptcy! Losses ${-self.cumulative_developer_profit:,.0f}"
                return
        
        if self.current_turn >= self.max_turns:
            # Do not set WIN; continue running; if no lose condition hit, remain RUNNING
            self.game_over_reason = "Learning in progress (win disabled); no lose condition triggered."
    
    def _get_zone_qol_stats(self, zone: Zone) -> Dict:
        """Calculate QoL statistics for a specific zone"""
        if not zone.developed:
            return {'avg_qol': 0.0, 'min_qol': 0.0, 'max_qol': 0.0}
        
        # Find households in this zone
        zone_households = [
            h for h in self.households 
            if zone.x <= h.location[1] < zone.x + zone.size 
            and zone.y <= h.location[0] < zone.y + zone.size
        ]
        
        if not zone_households:
            # No households but zone is developed - calculate from center
            center_y = zone.y + zone.size // 2
            center_x = zone.x + zone.size // 2
            qol = self._calculate_qol(center_y, center_x, WealthClass.MIDDLE)
            return {'avg_qol': qol, 'min_qol': qol, 'max_qol': qol}
        
        qols = [h.satisfaction for h in zone_households]
        return {
            'avg_qol': float(np.mean(qols)),
            'min_qol': float(np.min(qols)),
            'max_qol': float(np.max(qols))
        }
    
    def get_state_vector(self) -> np.ndarray:
        """Get base state vector for RL agent (20 dimensions)"""
        m = self.metrics
        
        return np.array([
            self.current_turn / self.max_turns,
            m.population / 5000,
            m.poor_pop / (m.population + 1),
            m.middle_pop / (m.population + 1),
            m.rich_pop / (m.population + 1),
            m.segregation_index,
            m.avg_qol / 5.0,
            m.gdp / 500_000_000,
            self.cumulative_developer_profit / 1_000_000,
            m.green_space_ratio,
            m.housing_vacancy_rate,
            m.affordability_index,
            float(m.poor_housing) / max(1, int(m.total_housing) + 1),
            float(m.medium_housing) / max(1, int(m.total_housing) + 1),
            float(m.luxury_housing) / max(1, int(m.total_housing) + 1),
            len([z for z in self.zones if z.developed]) / self.num_zones,
            m.sustainability_score,
            m.avg_pollution,
            float(self.game_state == GameState.RUNNING),
            self._calculate_city_attractiveness()
        ], dtype=np.float32)
    
    def get_developer_state(self, zone_id: int) -> np.ndarray:
        """
        Enhanced state vector for developer with spatial context.
        
        Structure (113 dimensions total):
        - Base state (20): Global city metrics
        - Zone-specific (3): Current zone ID, row, col
        - Global zones (45): 9 zones × 5 features each
        - Neighbor awareness (45): 8 neighbors + self × 5 features each
        
        Features per zone: [developed?, poor%, medium%, luxury%, green%]
        """
        # 1. Base state (20 dims)
        base_state = self.get_state_vector()
        
        # 2. Zone-specific info (3 dims)
        zone = self.zones[zone_id]
        zone_row = zone_id // 3  # 0, 1, 2
        zone_col = zone_id % 3   # 0, 1, 2
        zone_info = np.array([
            zone_id / (self.num_zones - 1),  # Normalized zone ID
            zone_row / 2.0,  # Row position (0, 0.5, 1)
            zone_col / 2.0,  # Col position (0, 0.5, 1)
        ], dtype=np.float32)
        
        # 3. Global developed zones vector (9 zones × 5 features = 45 dims)
        global_zones = []
        for z in self.zones:
            total_built = z.poor_housing + z.medium_housing + z.luxury_housing + z.green_space + z.woodland
            if total_built > 0:
                global_zones.extend([
                    float(z.developed),
                    z.poor_housing / total_built,
                    z.medium_housing / total_built,
                    z.luxury_housing / total_built,
                    (z.green_space + z.woodland) / total_built,
                ])
            else:
                global_zones.extend([float(z.developed), 0.0, 0.0, 0.0, 0.0])
        global_zones = np.array(global_zones, dtype=np.float32)
        
        # 4. Neighbor awareness (8 neighbors + self = 9 × 5 = 45 dims)
        # Neighbor order: NW, N, NE, W, Self, E, SW, S, SE
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),  # Top row: NW, N, NE
            (0, -1),  (0, 0),  (0, 1),   # Middle row: W, Self, E
            (1, -1),  (1, 0),  (1, 1),   # Bottom row: SW, S, SE
        ]
        
        neighbor_features = []
        for dr, dc in neighbor_offsets:
            nr, nc = zone_row + dr, zone_col + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                nz = self.zones[nr * 3 + nc]
                total_built = nz.poor_housing + nz.medium_housing + nz.luxury_housing + nz.green_space + nz.woodland
                if total_built > 0:
                    neighbor_features.extend([
                        float(nz.developed),
                        nz.poor_housing / total_built,
                        nz.medium_housing / total_built,
                        nz.luxury_housing / total_built,
                        (nz.green_space + nz.woodland) / total_built,
                    ])
                else:
                    neighbor_features.extend([float(nz.developed), 0.0, 0.0, 0.0, 0.0])
            else:
                # Out of bounds - return zeros (edge zones have no neighbors)
                neighbor_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        neighbor_features = np.array(neighbor_features, dtype=np.float32)
        
        # Concatenate all components
        # Total: 20 + 3 + 45 + 45 = 113 dimensions
        enhanced_state = np.concatenate([base_state, zone_info, global_zones, neighbor_features])
        
        return enhanced_state
    
    def get_full_state(self) -> Dict:
        """Get full state for visualization"""
        return {
            'land_grid': self.land_grid.tolist(),
            'housing_grid': self.housing_grid.tolist(),
            'zones': [
                {
                    'id': z.id,
                    'x': z.x,
                    'y': z.y,
                    'size': z.size,
                    'developed': z.developed,
                    'poor_housing': z.poor_housing,
                    'medium_housing': z.medium_housing,
                    'luxury_housing': z.luxury_housing,
                    'green_space': z.green_space,
                    'woodland': z.woodland,
                    **self._get_zone_qol_stats(z)
                }
                for z in self.zones
            ],
            'households': [
                {
                    'id': h.id,
                    'x': h.location[1],
                    'y': h.location[0],
                    'wealth_class': int(h.wealth_class),
                    'satisfaction': h.satisfaction
                }
                for h in self.households
            ],
            'metrics': {
                'gdp': self.metrics.gdp,
                'population': self.metrics.population,
                'segregation': self.metrics.segregation_index,
                'avg_qol': self.metrics.avg_qol,
                'developer_profit': self.cumulative_developer_profit,
                'affordable_pct': self.metrics.poor_housing / (self.metrics.total_housing + 1),
                'green_ratio': self.metrics.green_space_ratio,
                'vacancy_rate': self.metrics.housing_vacancy_rate,
                'poor_count': self.metrics.poor_pop,
                'middle_count': self.metrics.middle_pop,
                'rich_count': self.metrics.rich_pop,
                'poor_housing': self.metrics.poor_housing,
                'medium_housing': self.metrics.medium_housing,
                'luxury_housing': self.metrics.luxury_housing,
                'total_housing': self.metrics.total_housing,
                'attractiveness': self._calculate_city_attractiveness()
            },
            'game_state': int(self.game_state),
            'game_over_reason': self.game_over_reason,
            'current_turn': self.current_turn,
            'max_turns': self.max_turns,
            'migration_log': self.migration_log[-5:] if hasattr(self, 'migration_log') else []
        }


if __name__ == '__main__':
    """Test the simulation"""
    print("=" * 60)
    print("Testing Urban Simulation - Three Pillars")
    print("=" * 60)
    
    sim = UrbanSimulation()
    print(f"Initial population: {len(sim.households)}")
    print(f"Zones: {len(sim.zones)}")
    print("Test passed!")

