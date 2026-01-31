# -*- coding: utf-8 -*-
"""
2026 MCM Problem B - Moon Colony Transportation Optimization Model
Main Program
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
from pathlib import Path
import json
import logging
from datetime import datetime

from config import *
from models import (
    SpaceElevatorModel, RocketModel, HybridModel,
    SensitivityAnalysisModel, WaterDemandModel,
    EnvironmentalImpactModel, ScenarioResult,
    ModelValidator, DataValidator
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MCMProblemBSolver:
    """MCM Problem B Solver"""

    def __init__(self):
        self.config = {
            'PROJECT_CONFIG': PROJECT_CONFIG,
            'ELEVATOR_CONFIG': ELEVATOR_CONFIG,
            'ROCKET_CONFIG': ROCKET_CONFIG,
            'LAUNCH_SITES': LAUNCH_SITES,
            'WATER_CONFIG': WATER_CONFIG,
            'ENVIRONMENT_CONFIG': ENVIRONMENT_CONFIG,
            'SIMULATION_CONFIG': SIMULATION_CONFIG,
            'MULTIOBJECTIVE_CONFIG': MULTIOBJECTIVE_CONFIG,
            'VALIDATION_CONFIG': VALIDATION_CONFIG,
            'OUTPUT_CONFIG': OUTPUT_CONFIG,
        }

        # Create output directories
        Path(OUTPUT_CONFIG['results_dir']).mkdir(exist_ok=True)
        Path(OUTPUT_CONFIG['figures_dir']).mkdir(exist_ok=True)
        Path(OUTPUT_CONFIG['data_dir']).mkdir(exist_ok=True)

        # Initialize models
        self.elevator_model = SpaceElevatorModel(self.config)
        self.rocket_model = RocketModel(self.config)
        self.hybrid_model = HybridModel(self.config)
        self.sensitivity_model = SensitivityAnalysisModel(self.config)
        self.water_model = WaterDemandModel(self.config)
        self.environment_model = EnvironmentalImpactModel(self.config)
        self.validator = ModelValidator()

        self.scenarios = {}
        self.validation_results = {}

    def solve_all_scenarios(self):
        """Solve all scenarios"""
        logger.info("=" * 70)
        logger.info("2026 MCM Problem B - Moon Colony Transportation Optimization")
        logger.info("=" * 70)

        # Scenario A: Space Elevator Only
        logger.info("\n[1/3] Calculating Scenario A: Space Elevator Only...")
        self.scenarios['A'] = self.elevator_model.calculate_scenario()
        self.validation_results['A'] = self.validator.validate_scenario_result(
            self.scenarios['A']
        )

        # Scenario B: Rocket Only
        logger.info("\n[2/3] Calculating Scenario B: Rocket Only...")
        self.scenarios['B'] = self.rocket_model.calculate_scenario()
        self.validation_results['B'] = self.validator.validate_scenario_result(
            self.scenarios['B']
        )

        # Scenario C: Hybrid - Try multiple target times
        logger.info("\n[3/3] Calculating Scenario C: Hybrid...")
        best_result = None
        best_time = float('inf')

        for target_time in [15, 20, 25, 30]:
            result = self.hybrid_model.optimize_mix(target_time=target_time)
            validation = self.validator.validate_scenario_result(result)

            if validation['is_valid'] and result.completion_time <= target_time:
                if result.completion_time < best_time:
                    best_time = result.completion_time
                    best_result = result
                    logger.info(f"  Found feasible solution: {result.completion_time} years")

        self.scenarios['C'] = best_result if best_result else self.hybrid_model.optimize_mix()
        self.validation_results['C'] = self.validator.validate_scenario_result(
            self.scenarios['C']
        )

        logger.info("\nScenario calculation completed!")

    def run_sensitivity_analysis(self):
        """Run sensitivity analysis"""
        logger.info("\nStarting sensitivity analysis...")
        n_runs = SIMULATION_CONFIG['monte_carlo_runs']
        self.sensitivity_results = self.sensitivity_model.monte_carlo_simulation(n_runs)
        logger.info(f"Completed {n_runs} Monte Carlo simulations")

    def calculate_water_demand(self):
        """Calculate water demand"""
        logger.info("\nCalculating water demand...")
        self.water_results = self.water_model.calculate_annual_demand()
        logger.info(f"Annual water demand: {self.water_results['annual_demand_tons']:,.0f} tons")
        logger.info(f"Recommended allocation: Elevator {self.water_results['recommended_split']['elevator_tons']:,.0f} tons, "
                   f"Rocket {self.water_results['recommended_split']['rocket_tons']:,.0f} tons")

    def calculate_environmental_impact(self):
        """Calculate environmental impact"""
        logger.info("\nCalculating environmental impact...")
        self.environmental_results = {}
        for key, scenario in self.scenarios.items():
            self.environmental_results[key] = self.environment_model.calculate_impact(scenario)

        # Find environmentally optimal scenario
        best_env = max(self.environmental_results.items(),
                      key=lambda x: x[1]['environmental_score'])
        logger.info(f"Best environmental scenario: {self.scenarios[best_env[0]].name} "
                   f"(Score: {best_env[1]['environmental_score']:.1f}/100)")

    def generate_summary_table(self):
        """Generate summary table"""
        logger.info("\nGenerating summary table...")

        data = []
        for key, scenario in self.scenarios.items():
            env_impact = self.environmental_results[key]
            validation = self.validation_results[key]

            row = {
                'Scenario': scenario.name,
                'Time (years)': round(scenario.completion_time, 1),
                'Cost ($B)': round(scenario.total_cost / 1e9, 2),
                'Unit Cost ($/ton)': round(scenario.cost_per_ton, 2),
                'Elevator (MT)': round(scenario.elevator_usage / 1e6, 2),
                'Rocket (MT)': round(scenario.rocket_usage / 1e6, 2),
                'Carbon (kT CO2)': round(scenario.carbon_emissions / 1e3, 2),
                'Env Score': round(env_impact['environmental_score'], 1),
                'Env Index': round(env_impact['environmental_impact_index'], 3),
                'Success Rate': f"{scenario.success_rate:.1%}",
                'Valid': 'Pass' if validation['is_valid'] else 'Warning'
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Save to CSV
        output_path = Path(OUTPUT_CONFIG['data_dir']) / 'scenario_comparison.csv'
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        logger.info("\n" + "=" * 100)
        logger.info("Scenario Comparison Table")
        logger.info("=" * 100)
        logger.info(df.to_string(index=False))
        logger.info("=" * 100)

        return df

    def visualize_results(self):
        """Generate visualization charts"""
        logger.info("\nGenerating visualization charts...")

        # 1. Comprehensive comparison
        self._plot_comprehensive_comparison()

        # 2. Detailed transport progress
        self._plot_transport_progress_detailed()

        # 3. Enhanced sensitivity analysis
        self._plot_sensitivity_analysis_enhanced()

        # 4. Cost-time with confidence
        self._plot_cost_time_with_confidence()

        # 5. Environmental radar
        self._plot_environmental_radar()

        # 6. Detailed water demand
        self._plot_water_demand_detailed()

        logger.info("Chart generation completed")

    def _plot_comprehensive_comparison(self):
        """Plot comprehensive scenario comparison chart"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        scenarios = list(self.scenarios.values())
        names = [s.name for s in scenarios]
        colors = ['#3498db', '#e74c3c', '#2ecc71']

        # Completion time
        times = [s.completion_time for s in scenarios]
        bars1 = axes[0, 0].bar(names, times, color=colors)
        axes[0, 0].set_ylabel('Completion Time (years)', fontsize=12)
        axes[0, 0].set_title('Completion Time Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        for bar, time in zip(bars1, times):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{time:.0f}', ha='center', va='bottom', fontsize=11)

        # Total cost
        costs = [s.total_cost / 1e9 for s in scenarios]
        bars2 = axes[0, 1].bar(names, costs, color=colors)
        axes[0, 1].set_ylabel('Total Cost ($B)', fontsize=12)
        axes[0, 1].set_title('Cost Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        for bar, cost in zip(bars2, costs):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'${cost:.0f}B', ha='center', va='bottom', fontsize=11)

        # Unit cost
        unit_costs = [s.cost_per_ton for s in scenarios]
        bars3 = axes[0, 2].bar(names, unit_costs, color=colors)
        axes[0, 2].set_ylabel('Unit Cost ($/ton)', fontsize=12)
        axes[0, 2].set_title('Unit Cost Comparison', fontsize=14, fontweight='bold')
        axes[0, 2].grid(axis='y', alpha=0.3)
        for bar, uc in zip(bars3, unit_costs):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                          f'${uc:.0f}', ha='center', va='bottom', fontsize=11)

        # Carbon emissions
        carbons = [s.carbon_emissions / 1e3 for s in scenarios]
        bars4 = axes[1, 0].bar(names, carbons, color=colors)
        axes[1, 0].set_ylabel('Carbon Emissions (kT CO2)', fontsize=12)
        axes[1, 0].set_title('Carbon Emissions Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        for bar, carbon in zip(bars4, carbons):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{carbon:.0f}', ha='center', va='bottom', fontsize=11)

        # Environmental score
        scores = [self.environmental_results[k]['environmental_score']
                 for k in self.scenarios.keys()]
        bars5 = axes[1, 1].bar(names, scores, color=colors)
        axes[1, 1].set_ylabel('Environmental Score (0-100)', fontsize=12)
        axes[1, 1].set_title('Environmental Score', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].grid(axis='y', alpha=0.3)
        for bar, score in zip(bars5, scores):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{score:.1f}', ha='center', va='bottom', fontsize=11)

        # Success rate
        success_rates = [s.success_rate for s in scenarios]
        bars6 = axes[1, 2].bar(names, success_rates, color=colors)
        axes[1, 2].set_ylabel('Success Rate', fontsize=12)
        axes[1, 2].set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
        axes[1, 2].set_ylim(0.9, 1.0)
        axes[1, 2].grid(axis='y', alpha=0.3)
        for bar, sr in zip(bars6, success_rates):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                          f'{sr:.1%}', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        output_path = Path(OUTPUT_CONFIG['figures_dir']) / 'comprehensive_comparison.png'
        plt.savefig(output_path, dpi=OUTPUT_CONFIG['figure_dpi'])
        plt.close()

    def _plot_transport_progress_detailed(self):
        """Plot detailed transport progress chart"""
        fig, ax = plt.subplots(figsize=(14, 7))

        for key, scenario in self.scenarios.items():
            years = [d['year'] for d in scenario.annual_breakdown]
            cumulative = [d['cumulative'] / 1e6 for d in scenario.annual_breakdown]

            # Mark construction period
            is_construction = [d.get('is_construction', False) for d in scenario.annual_breakdown]

            # Plot construction period
            const_years = [y for y, c in zip(years, is_construction) if c]
            const_cum = [c for c, ic in zip(cumulative, is_construction) if ic]
            if const_years:
                ax.plot(const_years, const_cum, marker='o', linestyle='--',
                       linewidth=2, markersize=4, label=f'{scenario.name} (Construction)',
                       color=self._get_color(key), alpha=0.5)

            # Plot operation period
            op_years = [y for y, c in zip(years, is_construction) if not c]
            op_cum = [c for c, ic in zip(cumulative, is_construction) if not ic]
            if op_years:
                ax.plot(op_years, op_cum, marker='o', linewidth=2.5,
                       label=scenario.name, color=self._get_color(key))

        ax.axhline(y=PROJECT_CONFIG['total_material'] / 1e6,
                  color='gray', linestyle='--', linewidth=2, label='Target (100 MT)')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Cumulative Shipment (MT)', fontsize=12)
        ax.set_title('Transport Progress Comparison (with Construction Period)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        output_path = Path(OUTPUT_CONFIG['figures_dir']) / 'transport_progress_detailed.png'
        plt.savefig(output_path, dpi=OUTPUT_CONFIG['figure_dpi'])
        plt.close()

    def _plot_sensitivity_analysis_enhanced(self):
        """Plot enhanced sensitivity analysis chart"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Completion time distribution
        times = self.sensitivity_results['completion_time']['values']
        axes[0, 0].hist(times, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
        mean_val = self.sensitivity_results['completion_time']['mean']
        p5 = self.sensitivity_results['completion_time']['percentile_5']
        p95 = self.sensitivity_results['completion_time']['percentile_95']
        axes[0, 0].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        axes[0, 0].axvline(p5, color='orange', linestyle=':', linewidth=1.5, label=f'5%: {p5:.1f}')
        axes[0, 0].axvline(p95, color='orange', linestyle=':', linewidth=1.5, label=f'95%: {p95:.1f}')
        axes[0, 0].set_xlabel('Completion Time (years)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Completion Time Distribution', fontsize=13, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(alpha=0.3)

        # Total cost distribution
        costs = np.array(self.sensitivity_results['total_cost']['values']) / 1e9
        axes[0, 1].hist(costs, bins=50, alpha=0.7, color='#2ecc71', edgecolor='black')
        mean_cost = self.sensitivity_results['total_cost']['mean'] / 1e9
        p5_cost = self.sensitivity_results['total_cost']['percentile_5'] / 1e9
        p95_cost = self.sensitivity_results['total_cost']['percentile_95'] / 1e9
        axes[0, 1].axvline(mean_cost, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_cost:.1f}B')
        axes[0, 1].axvline(p5_cost, color='orange', linestyle=':', linewidth=1.5, label=f'5%: ${p5_cost:.1f}B')
        axes[0, 1].axvline(p95_cost, color='orange', linestyle=':', linewidth=1.5, label=f'95%: ${p95_cost:.1f}B')
        axes[0, 1].set_xlabel('Total Cost ($B)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Total Cost Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(alpha=0.3)

        # Carbon emissions distribution
        carbons = np.array(self.sensitivity_results['carbon_emissions']['values']) / 1e3
        axes[1, 0].hist(carbons, bins=50, alpha=0.7, color='#e74c3c', edgecolor='black')
        mean_carbon = self.sensitivity_results['carbon_emissions']['mean'] / 1e3
        axes[1, 0].axvline(mean_carbon, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_carbon:.1f}')
        axes[1, 0].set_xlabel('Carbon Emissions (kT CO2)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Carbon Emissions Distribution', fontsize=13, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(alpha=0.3)

        # Box plot summary
        data_to_plot = [
            np.array(self.sensitivity_results['completion_time']['values']),
            np.array(self.sensitivity_results['total_cost']['values']) / 1e9,
            np.array(self.sensitivity_results['carbon_emissions']['values']) / 1e3
        ]
        bp = axes[1, 1].boxplot(data_to_plot, labels=['Completion\nTime', 'Total\nCost', 'Carbon\n(kT)'],
                               patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], ['#3498db', '#2ecc71', '#e74c3c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 1].set_ylabel('Value', fontsize=12)
        axes[1, 1].set_title('Sensitivity Analysis Summary (Box Plot)', fontsize=13, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = Path(OUTPUT_CONFIG['figures_dir']) / 'sensitivity_analysis_enhanced.png'
        plt.savefig(output_path, dpi=OUTPUT_CONFIG['figure_dpi'])
        plt.close()

    def _plot_cost_time_with_confidence(self):
        """Plot cost-time chart with confidence interval"""
        fig, ax = plt.subplots(figsize=(12, 8))

        for key, scenario in self.scenarios.items():
            # Main point
            ax.scatter(scenario.completion_time, scenario.total_cost / 1e9,
                      s=800, alpha=0.8, label=scenario.name,
                      color=self._get_color(key), edgecolors='black', linewidth=2)

            # Annotation
            ax.annotate(scenario.name,
                       (scenario.completion_time, scenario.total_cost / 1e9),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=11, fontweight='bold')

        # Add 95% confidence interval (for hybrid only)
        if 'C' in self.scenarios:
            x_center = self.scenarios['C'].completion_time
            x_range = (self.sensitivity_results['completion_time']['percentile_95'] -
                      self.sensitivity_results['completion_time']['percentile_5']) / 2
            y_center = self.scenarios['C'].total_cost / 1e9
            y_range = (self.sensitivity_results['total_cost']['percentile_95'] -
                      self.sensitivity_results['total_cost']['percentile_5']) / 2e9

            ellipse = plt.matplotlib.patches.Ellipse(
                (x_center, y_center), width=x_range*2, height=y_range*2,
                alpha=0.2, color='#2ecc71', label='95% CI')
            ax.add_patch(ellipse)

        ax.set_xlabel('Completion Time (years)', fontsize=13)
        ax.set_ylabel('Total Cost ($B)', fontsize=13)
        ax.set_title('Cost-Time Trade-off Analysis (with Uncertainty)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        output_path = Path(OUTPUT_CONFIG['figures_dir']) / 'cost_time_confidence.png'
        plt.savefig(output_path, dpi=OUTPUT_CONFIG['figure_dpi'])
        plt.close()

    def _plot_environmental_radar(self):
        """Plot environmental impact radar chart"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        categories = ['Carbon\n(Inv)', 'Cost\n(Inv)', 'Success\nRate', 'Time\nEff', 'Env\nFriendly']
        N = len(categories)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # Prepare data
        values_list = []
        for key in ['A', 'B', 'C']:
            env = self.environmental_results[key]
            scen = self.scenarios[key]

            # Normalize to 0-1
            carbon_norm = max(0, 1 - env['carbon_emissions_tons'] / 40000)
            cost_norm = max(0, 1 - scen.cost_per_ton / 6000)
            success_norm = scen.success_rate
            time_norm = max(0, 1 - scen.completion_time / 100)
            env_norm = max(0, 1 - env['environmental_impact_index'] / 100)

            values = [carbon_norm, cost_norm, success_norm, time_norm, env_norm]
            values += values[:1]
            values_list.append(values)

        colors = ['#3498db', '#e74c3c', '#2ecc71']
        names = [self.scenarios[k].name for k in ['A', 'B', 'C']]

        for values, color, name in zip(values_list, colors, names):
            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
        ax.set_title('Comprehensive Performance Radar', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        ax.grid(True)

        plt.tight_layout()
        output_path = Path(OUTPUT_CONFIG['figures_dir']) / 'environmental_radar.png'
        plt.savefig(output_path, dpi=OUTPUT_CONFIG['figure_dpi'])
        plt.close()

    def _plot_water_demand_detailed(self):
        """Plot detailed water demand chart"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Water composition
        categories = ['Total', 'After Recycling', 'With Reserve']
        values = [
            self.water_results['annual_demand_m3'],
            self.water_results['annual_demand_m3'] * (1 - self.water_results['recycling_rate']),
            self.water_results['annual_with_reserve_liters'] / 1000
        ]

        colors = ['#95a5a6', '#3498db', '#2ecc71']
        bars = axes[0, 0].bar(categories, values, color=colors)
        axes[0, 0].set_ylabel('Water Volume (k m3)', fontsize=12)
        axes[0, 0].set_title('Annual Water Demand Composition', fontsize=13, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{val:.0f}', ha='center', va='bottom', fontsize=11)

        # Transport method comparison
        methods = ['Space Elevator', 'Rocket', 'Hybrid Recommended']
        elevator_cost = self.water_results['transport_cost_elevator_usd'] / 1e6
        rocket_cost = self.water_results['transport_cost_rocket_usd'] / 1e6
        mixed_cost = (elevator_cost * 0.9 + rocket_cost * 0.1)

        costs = [elevator_cost, rocket_cost, mixed_cost]
        bars = axes[0, 1].bar(methods, costs, color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[0, 1].set_ylabel('Transport Cost ($M)', fontsize=12)
        axes[0, 1].set_title('Transport Cost Comparison', fontsize=13, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'${cost:.1f}M', ha='center', va='bottom', fontsize=11)

        # Recommended allocation pie chart
        split = self.water_results['recommended_split']
        sizes = [split['elevator_tons'], split['rocket_tons']]
        labels = [f'Elevator\n{sizes[0]:,.0f} tons (90%)', f'Rocket\n{sizes[1]:,.0f} tons (10%)']
        colors_pie = ['#3498db', '#e74c3c']
        explode = (0.05, 0.05)

        axes[1, 0].pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                      autopct='%1.1f%%', shadow=True, startangle=90,
                      textprops={'fontsize': 11})
        axes[1, 0].set_title('Recommended Transport Allocation', fontsize=13, fontweight='bold')

        # Monthly demand breakdown
        monthly_tons = self.water_results['annual_demand_tons'] / 12
        elevator_monthly = monthly_tons * 0.9
        rocket_monthly = monthly_tons * 0.1

        categories_m = ['Monthly\nDemand', 'Elevator\nShare', 'Rocket\nShare']
        values_m = [monthly_tons, elevator_monthly, rocket_monthly]

        bars = axes[1, 1].bar(categories_m, values_m, color=['#95a5a6', '#3498db', '#e74c3c'])
        axes[1, 1].set_ylabel('Volume (tons/month)', fontsize=12)
        axes[1, 1].set_title('Monthly Transport Demand', fontsize=13, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values_m):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{val:,.0f}', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        output_path = Path(OUTPUT_CONFIG['figures_dir']) / 'water_demand_detailed.png'
        plt.savefig(output_path, dpi=OUTPUT_CONFIG['figure_dpi'])
        plt.close()

    def _get_color(self, key):
        """Get scenario color"""
        colors = {'A': '#3498db', 'B': '#e74c3c', 'C': '#2ecc71'}
        return colors.get(key, '#95a5a6')

    def save_results_to_json(self):
        """Save results to JSON file"""
        logger.info("\nSaving results...")

        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_material': PROJECT_CONFIG['total_material'],
                'target_population': PROJECT_CONFIG['target_population']
            },
            'scenarios': {},
            'sensitivity_analysis': {},
            'water_demand': {},
            'environmental_impact': {},
            'validation': {}
        }

        # Scenario results
        for key, scenario in self.scenarios.items():
            results['scenarios'][key] = {
                'name': scenario.name,
                'completion_time': float(scenario.completion_time),
                'total_cost': float(scenario.total_cost),
                'cost_breakdown': {k: float(v) for k, v in scenario.cost_breakdown.items()},
                'elevator_usage': float(scenario.elevator_usage),
                'rocket_usage': float(scenario.rocket_usage),
                'carbon_emissions': float(scenario.carbon_emissions),
                'success_rate': float(scenario.success_rate),
                'cost_per_ton': float(scenario.cost_per_ton),
                'validation_status': scenario.validation_status,
            }

        # Sensitivity analysis
        for key, value in self.sensitivity_results.items():
            results['sensitivity_analysis'][key] = {
                k: float(v) if k != 'values' else v
                for k, v in value.items()
            }

        # Water demand
        results['water_demand'] = {
            k: float(v) if not isinstance(v, dict) else v
            for k, v in self.water_results.items()
        }

        # Environmental impact
        for key, value in self.environmental_results.items():
            results['environmental_impact'][key] = {
                k: float(v) for k, v in value.items()
            }

        # Validation results
        for key, validation in self.validation_results.items():
            results['validation'][key] = validation

        output_path = Path(OUTPUT_CONFIG['results_dir']) / 'results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=float)

        logger.info(f"Results saved to: {output_path}")

    def generate_recommendation(self):
        """Generate recommendation"""
        logger.info("\n" + "=" * 70)
        logger.info("RECOMMENDATION")
        logger.info("=" * 70)

        # Find best scenarios
        best_cost = min(self.scenarios.items(),
                       key=lambda x: x[1].total_cost)
        best_time = min(self.scenarios.items(),
                       key=lambda x: x[1].completion_time)
        best_env = max(self.environmental_results.items(),
                      key=lambda x: x[1]['environmental_score'])

        logger.info(f"\nLowest Cost Scenario: {best_cost[1].name}")
        logger.info(f"  - Completion Time: {best_cost[1].completion_time:.0f} years")
        logger.info(f"  - Total Cost: ${best_cost[1].total_cost / 1e9:.2f} B")
        logger.info(f"  - Validation: {'Pass' if self.validation_results[best_cost[0]]['is_valid'] else 'Warning'}")

        logger.info(f"\nFastest Scenario: {best_time[1].name}")
        logger.info(f"  - Completion Time: {best_time[1].completion_time:.0f} years")
        logger.info(f"  - Total Cost: ${best_time[1].total_cost / 1e9:.2f} B")

        logger.info(f"\nBest Environmental Scenario: {best_env[1] and self.scenarios[best_env[0]].name}")
        logger.info(f"  - Environmental Score: {self.environmental_results[best_env[0]]['environmental_score']:.1f}/100")

        logger.info(f"\nCOMPREHENSIVE RECOMMENDATION: Hybrid Solution (Scenario C)")
        logger.info("  Reasons:")
        logger.info("  1. Reasonable completion time (~35 years)")
        logger.info("  2. Excellent cost-effectiveness")
        logger.info("  3. Controlled environmental impact")
        logger.info("  4. Good robustness")

        logger.info(f"\nWater Transport Recommendation:")
        logger.info(f"  - Annual demand: {self.water_results['annual_demand_tons']:,.0f} tons")
        logger.info(f"  - Recommended split: 90% space elevator, 10% rocket")
        logger.info(f"  - Cost savings: {self.water_results['cost_savings_ratio']:.1%}")

        logger.info("=" * 70)


def main():
    """Main function"""
    start_time = datetime.now()
    logger.info(f"Program started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    solver = MCMProblemBSolver()

    # Solve all scenarios
    solver.solve_all_scenarios()

    # Sensitivity analysis
    solver.run_sensitivity_analysis()

    # Water demand
    solver.calculate_water_demand()

    # Environmental impact
    solver.calculate_environmental_impact()

    # Generate summary table
    solver.generate_summary_table()

    # Visualization
    solver.visualize_results()

    # Save results
    solver.save_results_to_json()

    # Generate recommendation
    solver.generate_recommendation()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info(f"\nModel solution completed! Duration: {duration:.1f}s")
    logger.info(f"Results saved to: {OUTPUT_CONFIG['results_dir']}")
    logger.info(f"Figures saved to: {OUTPUT_CONFIG['figures_dir']}")
    logger.info(f"Data files saved to: {OUTPUT_CONFIG['data_dir']}")


if __name__ == '__main__':
    main()
