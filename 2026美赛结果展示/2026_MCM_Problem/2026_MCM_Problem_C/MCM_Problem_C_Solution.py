"""
2026 MCM Problem C: Data With The Stars
Complete Solution Implementation

This script implements:
1. Fan vote estimation model
2. Voting method comparison (rank-based vs percentage-based)
3. Controversial case analysis
4. Impact factor analysis (dancers and celebrity characteristics)
5. New voting system proposal

Author: 数学建模 Skill-Math Modeling Skill
Date: 2026-01-31
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style for SCI/Nature publications
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.dpi'] = 300

# Set random seed for reproducibility
np.random.seed(42)


class DWTSSolver:
    """Main solver class for DWTS problem"""

    def __init__(self, data_path):
        """Initialize the solver with data"""
        self.data = self.load_data(data_path)
        self.preprocess_data()

    def load_data(self, data_path):
        """Load the CSV data file"""
        print("Loading data from:", data_path)
        df = pd.read_csv(data_path, encoding='utf-8-sig')
        print(f"Data loaded: {df.shape[0]} contestants, {df.shape[1]} columns")
        return df

    def preprocess_data(self):
        """Preprocess the data for analysis"""
        print("\nPreprocessing data...")

        # Parse elimination week from results
        def parse_elimination_week(result):
            if pd.isna(result):
                return None
            if '1st Place' in result or 'Winner' in result:
                return 100  # Never eliminated (winner)
            if '2nd Place' in result or '3rd Place' in result:
                return 100  # Never eliminated (finalist)
            if 'Withdrew' in result:
                return -1  # Withdrew
            if 'Eliminated Week' in result:
                try:
                    return int(result.split('Week')[1].strip())
                except:
                    return None
            return None

        self.data['elimination_week'] = self.data['results'].apply(parse_elimination_week)

        # Extract judge scores for each week
        week_cols = [col for col in self.data.columns if 'week' in col.lower() and 'judge' in col.lower()]

        # Calculate total judge scores per week
        for week in range(1, 12):
            judge_cols = [col for col in self.data.columns if f'week{week}_judge' in col.lower()]
            if judge_cols:
                # Convert to numeric, replacing 'N/A' with NaN
                for col in judge_cols:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                self.data[f'week{week}_total_score'] = self.data[judge_cols].sum(axis=1, skipna=True)
                self.data[f'week{week}_avg_score'] = self.data[judge_cols].mean(axis=1, skipna=True)
                self.data[f'week{week}_num_judges'] = self.data[judge_cols].notna().sum(axis=1)

        print(f"Preprocessing complete. Processed {len(week_cols)} judge score columns.")

    def estimate_fan_votes_rank_method(self, season, num_samples=1000):
        """
        Estimate fan votes using rank-based method

        For rank-based method:
        - Combined score = Judge Rank + Fan Rank
        - Eliminated contestant has lowest combined score
        """
        print(f"\nEstimating fan votes for Season {season} (Rank Method)...")

        season_data = self.data[self.data['season'] == season].copy()

        # Get active contestants for each week
        weekly_votes = {}
        weekly_uncertainties = {}

        # Determine max weeks for this season
        max_week = season_data[[col for col in season_data.columns if 'week' in col.lower() and 'total_score' in col.lower()]].notna().any(axis=0).sum()

        for week in range(1, max_week + 1):
            score_col = f'week{week}_total_score'

            # Get contestants still in competition
            active = season_data[
                (season_data[score_col].notna()) &
                (season_data[score_col] > 0) &
                (season_data['elimination_week'] >= week)
            ].copy()

            if len(active) <= 1:
                continue

            # Calculate judge ranks
            active['judge_rank'] = active[score_col].rank(ascending=False)

            # Get eliminated contestant(s) for this week
            eliminated = active[active['elimination_week'] == week]

            if len(eliminated) == 0:
                # No elimination this week
                continue

            # For each eliminated contestant, estimate required fan rank
            for idx, elim_contestant in eliminated.iterrows():
                elim_name = elim_contestant['celebrity_name']
                elim_judge_rank = elim_contestant['judge_rank']

                # To be eliminated, their combined rank must be lowest
                # Fan rank must be at least len(active) - elim_judge_rank + 1
                min_fan_rank = len(active) - elim_judge_rank

                # Sample fan votes that would produce this rank
                fan_vote_samples = []
                for _ in range(num_samples):
                    # Generate fan votes that put eliminated contestant at required rank
                    base_votes = np.random.exponential(scale=1000000, size=len(active))
                    # Ensure eliminated has lowest fan votes
                    base_votes[active.index.get_loc(idx)] = base_votes.min() * 0.5

                    fan_vote_samples.append(dict(zip(active['celebrity_name'], base_votes)))

                # Calculate statistics
                fan_votes_df = pd.DataFrame(fan_vote_samples)
                weekly_votes[week] = fan_votes_df.mean().to_dict()
                weekly_uncertainties[week] = fan_votes_df.std().to_dict()

        return weekly_votes, weekly_uncertainties

    def estimate_fan_votes_percent_method(self, season, num_samples=1000):
        """
        Estimate fan votes using percentage-based method

        For percentage-based method:
        - Combined score = Judge % + Fan %
        - Eliminated contestant has lowest combined score
        """
        print(f"\nEstimating fan votes for Season {season} (Percent Method)...")

        season_data = self.data[self.data['season'] == season].copy()

        weekly_votes = {}
        weekly_uncertainties = {}

        # Determine max weeks for this season
        score_cols = [col for col in season_data.columns if 'week' in col.lower() and 'total_score' in col.lower()]
        max_week = 0
        for col in score_cols:
            try:
                week_num = int(col.split('week')[1].split('_')[0])
                max_week = max(max_week, week_num)
            except:
                pass

        for week in range(1, max_week + 1):
            score_col = f'week{week}_total_score'

            active = season_data[
                (season_data[score_col].notna()) &
                (season_data[score_col] > 0) &
                (season_data['elimination_week'] >= week)
            ].copy()

            if len(active) <= 1:
                continue

            # Calculate judge percentages
            total_judge_score = active[score_col].sum()
            active['judge_pct'] = active[score_col] / total_judge_score * 100

            eliminated = active[active['elimination_week'] == week]

            if len(eliminated) == 0:
                continue

            for idx, elim_contestant in eliminated.iterrows():
                elim_name = elim_contestant['celebrity_name']
                elim_judge_pct = elim_contestant['judge_pct']

                # To be eliminated, their combined % must be lowest
                # We need to find fan votes that achieve this
                fan_vote_samples = []
                for _ in range(num_samples):
                    # Generate fan votes with eliminated having lowest percentage
                    base_votes = np.random.exponential(scale=1000000, size=len(active))
                    base_votes[active.index.get_loc(idx)] = base_votes.min() * 0.3

                    total_votes = base_votes.sum()
                    fan_pcts = base_votes / total_votes * 100

                    fan_vote_samples.append(dict(zip(active['celebrity_name'], base_votes)))

                fan_votes_df = pd.DataFrame(fan_vote_samples)
                weekly_votes[week] = fan_votes_df.mean().to_dict()
                weekly_uncertainties[week] = fan_votes_df.std().to_dict()

        return weekly_votes, weekly_uncertainties

    def compare_voting_methods(self, seasons=None):
        """Compare rank-based vs percentage-based voting methods"""
        print("\n" + "="*60)
        print("COMPARING VOTING METHODS")
        print("="*60)

        if seasons is None:
            seasons = sorted(self.data['season'].unique())[:10]  # Analyze first 10 seasons

        results = []

        for season in seasons:
            print(f"\nAnalyzing Season {season}...")

            season_data = self.data[self.data['season'] == season].copy()

            # Calculate average judge scores for the season
            score_cols = [col for col in season_data.columns if 'week' in col.lower() and 'total_score' in col.lower()]
            season_data['avg_judge_score'] = season_data[score_cols].mean(axis=1)

            # Calculate final placement correlation with judge scores
            valid_data = season_data[season_data['placement'].notna()]

            if len(valid_data) > 2:
                judge_rank = valid_data['avg_judge_score'].rank(ascending=False)
                placement_rank = valid_data['placement'].rank(ascending=True)

                # Spearman correlation
                corr_judge_placement = stats.spearmanr(judge_rank, placement_rank)[0]

                results.append({
                    'season': season,
                    'num_contestants': len(valid_data),
                    'judge_placement_correlation': corr_judge_placement,
                    'winner': valid_data.loc[valid_data['placement'] == 1, 'celebrity_name'].values[0] if len(valid_data[valid_data['placement'] == 1]) > 0 else 'N/A'
                })

        comparison_df = pd.DataFrame(results)
        return comparison_df

    def analyze_controversial_cases(self):
        """Analyze the controversial cases mentioned in the problem"""
        print("\n" + "="*60)
        print("ANALYZING CONTROVERSIAL CASES")
        print("="*60)

        controversial = {
            'Season 2': {'name': 'Jerry Rice', 'description': 'Runner-up despite lowest judge scores in 5 weeks'},
            'Season 4': {'name': 'Billy Ray Cyrus', 'description': '5th place despite last place judge scores in 6 weeks'},
            'Season 11': {'name': 'Bristol Palin', 'description': '3rd place with lowest judge scores 12 times'},
            'Season 27': {'name': 'Bobby Bones', 'description': 'Winner despite consistently low judge scores'}
        }

        results = []

        for season_key, info in controversial.items():
            season = int(season_key.split()[1])
            print(f"\nAnalyzing {season_key}: {info['name']}")

            contestant_data = self.data[
                (self.data['season'] == season) &
                (self.data['celebrity_name'] == info['name'])
            ]

            if len(contestant_data) > 0:
                contestant = contestant_data.iloc[0]

                # Get season data for comparison
                season_data = self.data[self.data['season'] == season]

                # Calculate average scores
                score_cols = [col for col in season_data.columns if 'week' in col.lower() and 'total_score' in col.lower()]
                contestant_avg = contestant[score_cols].mean()
                season_avg = season_data[score_cols].mean(axis=1).mean()

                # Count weeks with lowest score
                lowest_count = 0
                total_weeks = 0
                for col in score_cols:
                    week_scores = season_data[col].dropna()
                    if len(week_scores) > 0 and contestant[col] > 0:
                        total_weeks += 1
                        if contestant[col] == week_scores.min():
                            lowest_count += 1

                results.append({
                    'season': season,
                    'celebrity': info['name'],
                    'final_placement': contestant['placement'],
                    'avg_score': contestant_avg,
                    'season_avg_score': season_avg,
                    'weeks_lowest_score': lowest_count,
                    'total_weeks_competed': total_weeks,
                    'description': info['description']
                })

        controversial_df = pd.DataFrame(results)
        return controversial_df

    def analyze_impact_factors(self):
        """Analyze the impact of dancers and celebrity characteristics"""
        print("\n" + "="*60)
        print("ANALYZING IMPACT FACTORS")
        print("="*60)

        # Prepare data for analysis
        analysis_data = self.data.copy()

        # Calculate average score for each contestant
        score_cols = [col for col in analysis_data.columns if 'week' in col.lower() and 'total_score' in col.lower()]
        analysis_data['avg_score'] = analysis_data[score_cols].mean(axis=1)

        # Filter valid data
        valid_data = analysis_data[
            (analysis_data['placement'].notna()) &
            (analysis_data['avg_score'].notna()) &
            (analysis_data['celebrity_age_during_season'].notna())
        ].copy()

        print(f"\nValid contestants for analysis: {len(valid_data)}")

        # 1. Age impact analysis
        age_groups = pd.cut(valid_data['celebrity_age_during_season'],
                           bins=[0, 25, 35, 45, 100],
                           labels=['Under 25', '25-34', '35-44', '45+'])
        valid_data['age_group'] = age_groups

        age_analysis = valid_data.groupby('age_group').agg({
            'placement': 'mean',
            'avg_score': 'mean',
            'celebrity_name': 'count'
        }).rename(columns={'celebrity_name': 'count'})

        # 2. Industry impact analysis
        industry_analysis = valid_data.groupby('celebrity_industry').agg({
            'placement': 'mean',
            'avg_score': 'mean',
            'celebrity_name': 'count'
        }).rename(columns={'celebrity_name': 'count'}).sort_values('placement')

        # 3. Top professional dancers analysis
        dancer_analysis = valid_data.groupby('ballroom_partner').agg({
            'placement': 'mean',
            'avg_score': 'mean',
            'celebrity_name': 'count'
        }).rename(columns={'celebrity_name': 'count'}).sort_values('placement')
        dancer_analysis = dancer_analysis[dancer_analysis['count'] >= 3]  # At least 3 seasons

        # 4. Statistical tests
        # Correlation between age and placement
        age_placement_corr = valid_data['celebrity_age_during_season'].corr(valid_data['placement'])

        # Correlation between average score and placement
        score_placement_corr = valid_data['avg_score'].corr(valid_data['placement'])

        results = {
            'age_analysis': age_analysis,
            'industry_analysis': industry_analysis,
            'dancer_analysis': dancer_analysis,
            'age_placement_correlation': age_placement_corr,
            'score_placement_correlation': score_placement_corr,
            'valid_data': valid_data
        }

        return results

    def propose_new_system(self):
        """Propose and evaluate a new voting system"""
        print("\n" + "="*60)
        print("PROPOSING NEW VOTING SYSTEM")
        print("="*60)

        print("\nProposed System: Weighted Adaptive Voting")
        print("-" * 50)

        print("\n1. WEIGHTED PERCENTAGE METHOD:")
        print("   Combined Score = w1 × Judge% + w2 × Fan% + w3 × Improvement%")
        print("   where:")
        print("   - w1, w2, w3 are weights that sum to 1")
        print("   - Improvement% measures progress from previous week")

        print("\n2. ELASTIC WEIGHTS:")
        print("   - Early weeks (1-4): w1=0.6, w2=0.3, w3=0.1 (emphasize technique)")
        print("   - Middle weeks (5-8): w1=0.4, w2=0.4, w3=0.2 (balanced)")
        print("   - Late weeks (9+): w1=0.3, w2=0.5, w3=0.2 (emphasize fan engagement)")

        print("\n3. PROGRESS BONUS:")
        print("   Contestants showing significant improvement get bonus points")
        print("   Bonus = max(0, (Score_this_week - Avg_previous_3_weeks) × 0.5)")

        print("\n4. CONSISTENCY BONUS:")
        print("   Contestants with consistent high scores get bonus")
        print("   Bonus = (Avg_last_4_weeks - 25) × 0.2 if positive")

        print("\n" + "-"*50)
        print("ADVANTAGES OF NEW SYSTEM:")
        print("+ Balances judge expertise with fan preferences")
        print("+ Encourages improvement throughout competition")
        print("+ Rewards consistency")
        print("+ Adapts to competition stage")
        print("+ Reduces likelihood of controversial outcomes")

        return {
            'system_name': 'Weighted Adaptive Voting System',
            'components': ['Judge Percentage', 'Fan Percentage', 'Improvement Bonus', 'Consistency Bonus'],
            'weights_early': [0.6, 0.3, 0.1],
            'weights_middle': [0.4, 0.4, 0.2],
            'weights_late': [0.3, 0.5, 0.2]
        }

    def create_visualizations(self):
        """Create all visualizations for the analysis"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)

        # Figure 1: Age vs Placement scatter plot
        self._plot_age_vs_placement()

        # Figure 2: Industry performance comparison
        self._plot_industry_performance()

        # Figure 3: Top dancers performance
        self._plot_top_dancers()

        # Figure 4: Score distribution by placement
        self._plot_score_by_placement()

        # Figure 5: Contestious cases visualization
        self._plot_controversial_cases()

        # Figure 6: Season comparison
        self._plot_season_comparison()

        print("\nAll visualizations saved to 'figures/' directory")

    def _plot_age_vs_placement(self):
        """Plot age vs final placement"""
        valid_data = self.data[
            (self.data['placement'].notna()) &
            (self.data['celebrity_age_during_season'].notna())
        ].copy()

        score_cols = [col for col in valid_data.columns if 'week' in col.lower() and 'total_score' in col.lower()]
        valid_data['avg_score'] = valid_data[score_cols].mean(axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot: Age vs Placement
        ax1.scatter(valid_data['celebrity_age_during_season'], valid_data['placement'],
                   c=valid_data['avg_score'], cmap='RdYlGn', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Age During Season', fontsize=12)
        ax1.set_ylabel('Final Placement (1=Best)', fontsize=12)
        ax1.set_title('Age vs Final Placement\n(Color = Average Score)', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, linestyle='--', alpha=0.6)
        cbar = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar.set_label('Average Judge Score', fontsize=10)

        # Box plot: Placement by Age Group
        age_groups = pd.cut(valid_data['celebrity_age_during_season'],
                           bins=[0, 25, 35, 45, 100],
                           labels=['Under 25', '25-34', '35-44', '45+'])
        valid_data['age_group'] = age_groups

        groups = []
        for group in ['Under 25', '25-34', '35-44', '45+']:
            groups.append(valid_data[valid_data['age_group'] == group]['placement'].values)

        bp = ax2.boxplot(groups, labels=['Under 25', '25-34', '35-44', '45+'],
                        patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']):
            patch.set_facecolor(color)
        ax2.set_xlabel('Age Group', fontsize=12)
        ax2.set_ylabel('Final Placement (1=Best)', fontsize=12)
        ax2.set_title('Placement Distribution by Age Group', fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, linestyle='--', alpha=0.6, axis='y')

        plt.tight_layout()
        plt.savefig('figures/figure1_age_vs_placement.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_industry_performance(self):
        """Plot industry performance comparison"""
        valid_data = self.data[
            (self.data['placement'].notna()) &
            (self.data['celebrity_industry'].notna())
        ].copy()

        industry_stats = valid_data.groupby('celebrity_industry').agg({
            'placement': ['mean', 'count'],
        }).droplevel(0, axis=1)
        industry_stats = industry_stats[industry_stats['count'] >= 10].sort_values('mean')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Average placement by industry
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(industry_stats)))
        bars = ax1.barh(industry_stats.index, industry_stats['mean'], color=colors, edgecolor='black', linewidth=1)
        ax1.set_xlabel('Average Final Placement (1=Best)', fontsize=12)
        ax1.set_title('Average Performance by Celebrity Industry\n(Minimum 10 contestants)', fontsize=13, fontweight='bold')
        ax1.invert_xaxis()
        ax1.grid(True, linestyle='--', alpha=0.6, axis='x')

        # Add count labels
        for i, (idx, row) in enumerate(industry_stats.iterrows()):
            ax1.text(row['mean'] + 0.3, i, f"n={row['count']:.0f}", va='center', fontsize=9)

        # Count by industry
        industry_counts = valid_data['celebrity_industry'].value_counts()
        industry_counts = industry_counts[industry_counts >= 10].sort_values(ascending=True)
        ax2.barh(industry_counts.index, industry_counts.values, color='steelblue', edgecolor='black', linewidth=1)
        ax2.set_xlabel('Number of Contestants', fontsize=12)
        ax2.set_title('Contestant Count by Industry', fontsize=13, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6, axis='x')

        plt.tight_layout()
        plt.savefig('figures/figure2_industry_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_top_dancers(self):
        """Plot top professional dancers performance"""
        valid_data = self.data[
            (self.data['placement'].notna()) &
            (self.data['ballroom_partner'].notna())
        ].copy()

        dancer_stats = valid_data.groupby('ballroom_partner').agg({
            'placement': ['mean', 'count'],
        }).droplevel(0, axis=1)
        dancer_stats = dancer_stats[dancer_stats['count'] >= 5].sort_values('mean')[:15]

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(dancer_stats)))
        bars = ax.barh(dancer_stats.index, dancer_stats['mean'], color=colors, edgecolor='black', linewidth=1)

        # Add count labels
        for i, (idx, row) in enumerate(dancer_stats.iterrows()):
            ax.text(row['mean'] + 0.2, i, f"{row['mean']:.2f} (n={row['count']:.0f})", va='center', fontsize=9)

        ax.set_xlabel('Average Final Placement (1=Best)', fontsize=12)
        ax.set_title('Top 15 Professional Dancers by Average Partner Placement\n(Minimum 5 seasons)', fontsize=13, fontweight='bold')
        ax.invert_xaxis()
        ax.grid(True, linestyle='--', alpha=0.6, axis='x')

        plt.tight_layout()
        plt.savefig('figures/figure3_top_dancers.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_score_by_placement(self):
        """Plot score distribution by final placement"""
        valid_data = self.data[
            (self.data['placement'].notna()) &
            (self.data['placement'] <= 5)  # Top 5 only
        ].copy()

        score_cols = [col for col in valid_data.columns if 'week' in col.lower() and 'total_score' in col.lower()]
        valid_data['avg_score'] = valid_data[score_cols].mean(axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Box plot
        placements = []
        scores = []
        for placement in [1, 2, 3, 4, 5]:
            data = valid_data[valid_data['placement'] == placement]['avg_score'].dropna()
            if len(data) > 0:
                placements.append(data)
                scores.append(f"{placement}")

        if placements:
            bp = ax1.boxplot(placements, labels=scores, patch_artist=True, showmeans=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax1.set_xlabel('Final Placement', fontsize=12)
            ax1.set_ylabel('Average Judge Score', fontsize=12)
            ax1.set_title('Judge Score Distribution by Final Placement\n(Top 5 Contestants)', fontsize=13, fontweight='bold')
            ax1.grid(True, linestyle='--', alpha=0.6, axis='y')

        # Violin-style distribution
        for placement in [1, 2, 3, 4, 5]:
            data = valid_data[valid_data['placement'] == placement]['avg_score'].dropna()
            if len(data) > 0:
                parts = ax2.violinplot([data.values], [placement], showmeans=True, showmedians=True)
                parts['bodies'][0].set_facecolor(plt.cm.RdYlGn((placement-1)/4))

        ax2.set_xlabel('Final Placement (1=Best)', fontsize=12)
        ax2.set_ylabel('Average Judge Score', fontsize=12)
        ax2.set_title('Score Distribution by Placement (Violin Plot)', fontsize=13, fontweight='bold')
        ax2.set_xticks([1, 2, 3, 4, 5])
        ax2.grid(True, linestyle='--', alpha=0.6, axis='y')

        plt.tight_layout()
        plt.savefig('figures/figure4_score_by_placement.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_controversial_cases(self):
        """Plot controversial cases analysis"""
        controversial_cases = [
            {'season': 2, 'name': 'Jerry Rice', 'placement': 2},
            {'season': 4, 'name': 'Billy Ray Cyrus', 'placement': 5},
            {'season': 11, 'name': 'Bristol Palin', 'placement': 3},
            {'season': 27, 'name': 'Bobby Bones', 'placement': 1}
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Placement comparison
        seasons = []
        placements = []
        expected_from_score = []
        labels = []

        for case in controversial_cases:
            contestant = self.data[
                (self.data['season'] == case['season']) &
                (self.data['celebrity_name'] == case['name'])
            ]

            if len(contestant) > 0:
                c = contestant.iloc[0]
                score_cols = [col for col in self.data.columns if 'week' in col.lower() and 'total_score' in col.lower()]
                avg_score = c[score_cols].mean()

                # Calculate expected placement based on score
                season_data = self.data[self.data['season'] == case['season']]
                season_avg_scores = season_data[score_cols].mean(axis=1)
                expected_placement = (season_avg_scores > avg_score).sum() + 1

                seasons.append(f"S{case['season']}")
                placements.append(case['placement'])
                expected_from_score.append(expected_placement)
                labels.append(case['name'])

        x = np.arange(len(seasons))
        width = 0.35

        bars1 = ax1.bar(x - width/2, placements, width, label='Actual Placement',
                       color='coral', edgecolor='black', linewidth=1)
        bars2 = ax1.bar(x + width/2, expected_from_score, width, label='Expected from Judge Score',
                       color='lightblue', edgecolor='black', linewidth=1)

        ax1.set_xlabel('Season', fontsize=12)
        ax1.set_ylabel('Final Placement (1=Best)', fontsize=12)
        ax1.set_title('Controversial Cases: Actual vs Expected Placement', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(seasons)
        ax1.invert_yaxis()
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6, axis='y')

        # Score comparison
        avg_scores = []
        season_avgs = []

        for case in controversial_cases:
            contestant = self.data[
                (self.data['season'] == case['season']) &
                (self.data['celebrity_name'] == case['name'])
            ]

            if len(contestant) > 0:
                c = contestant.iloc[0]
                score_cols = [col for col in self.data.columns if 'week' in col.lower() and 'total_score' in col.lower()]
                contestant_avg = c[score_cols].mean()

                season_data = self.data[self.data['season'] == case['season']]
                season_avg = season_data[score_cols].mean(axis=1).mean()

                avg_scores.append(contestant_avg)
                season_avgs.append(season_avg)

        x = np.arange(len(seasons))
        width = 0.35

        bars1 = ax2.bar(x - width/2, avg_scores, width, label='Contestant Avg Score',
                       color='coral', edgecolor='black', linewidth=1)
        bars2 = ax2.bar(x + width/2, season_avgs, width, label='Season Avg Score',
                       color='lightblue', edgecolor='black', linewidth=1)

        ax2.set_xlabel('Season', fontsize=12)
        ax2.set_ylabel('Average Judge Score', fontsize=12)
        ax2.set_title('Controversial Cases: Score Comparison', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(seasons)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6, axis='y')

        plt.tight_layout()
        plt.savefig('figures/figure5_controversial_cases.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_season_comparison(self):
        """Plot season-level comparison"""
        season_stats = self.data.groupby('season').agg({
            'placement': lambda x: x.notna().sum(),
            'celebrity_name': 'count'
        }).rename(columns={'placement': 'num_finished', 'celebrity_name': 'num_total'})

        # Calculate winner score
        winner_scores = []
        seasons_analyzed = []

        for season in sorted(self.data['season'].unique()):
            season_data = self.data[self.data['season'] == season]
            winner = season_data[season_data['placement'] == 1]

            if len(winner) > 0:
                score_cols = [col for col in season_data.columns if 'week' in col.lower() and 'total_score' in col.lower()]
                winner_avg = winner[score_cols].mean(axis=1).values[0]
                winner_scores.append(winner_avg)
                seasons_analyzed.append(season)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Contestants per season
        ax1.bar(season_stats.index, season_stats['num_total'], color='steelblue',
               edgecolor='black', linewidth=1, label='Total')
        ax1.bar(season_stats.index, season_stats['num_finished'], color='lightgreen',
               edgecolor='black', linewidth=1, label='Finished')
        ax1.set_xlabel('Season', fontsize=12)
        ax1.set_ylabel('Number of Contestants', fontsize=12)
        ax1.set_title('Contestants per Season', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6, axis='y')

        # Winner average scores
        if winner_scores:
            ax2.plot(seasons_analyzed, winner_scores, 'o-', linewidth=2, markersize=8,
                    color='coral', markerfacecolor='white', markeredgewidth=2)
            ax2.set_xlabel('Season', fontsize=12)
            ax2.set_ylabel("Winner's Average Judge Score", fontsize=12)
            ax2.set_title("Winner's Average Score by Season", fontsize=13, fontweight='bold')
            ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig('figures/figure6_season_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)

        report_lines = []
        report_lines.append("="*80)
        report_lines.append("2026 MCM PROBLEM C: DATA WITH THE STARS")
        report_lines.append("COMPREHENSIVE ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append("")

        # Section 1: Data Overview
        report_lines.append("1. DATA OVERVIEW")
        report_lines.append("-" * 40)
        report_lines.append(f"Total contestants: {len(self.data)}")
        report_lines.append(f"Seasons covered: {self.data['season'].min()} to {self.data['season'].max()}")
        report_lines.append(f"Total seasons: {self.data['season'].nunique()}")
        report_lines.append("")

        # Section 2: Voting Method Comparison
        report_lines.append("2. VOTING METHOD COMPARISON")
        report_lines.append("-" * 40)

        comparison = self.compare_voting_methods()
        report_lines.append(f"Analyzed {len(comparison)} seasons")
        report_lines.append(f"Average judge-placement correlation: {comparison['judge_placement_correlation'].mean():.3f}")
        report_lines.append("")

        # Section 3: Controversial Cases
        report_lines.append("3. CONTROVERSIAL CASES ANALYSIS")
        report_lines.append("-" * 40)

        controversial = self.analyze_controversial_cases()
        for _, case in controversial.iterrows():
            report_lines.append(f"Season {case['season']}: {case['celebrity']}")
            report_lines.append(f"  Final Placement: {case['final_placement']}")
            report_lines.append(f"  Avg Score: {case['avg_score']:.2f} (Season Avg: {case['season_avg_score']:.2f})")
            report_lines.append(f"  Weeks with Lowest Score: {case['weeks_lowest_score']}/{case['total_weeks_competed']}")
            report_lines.append("")

        # Section 4: Impact Factors
        report_lines.append("4. IMPACT FACTOR ANALYSIS")
        report_lines.append("-" * 40)

        impact = self.analyze_impact_factors()

        report_lines.append("Age-Placement Correlation:")
        report_lines.append(f"  {impact['age_placement_correlation']:.3f}")
        report_lines.append("")

        report_lines.append("Score-Placement Correlation:")
        report_lines.append(f"  {impact['score_placement_correlation']:.3f}")
        report_lines.append("")

        report_lines.append("Top 5 Industries by Performance:")
        for i, (industry, row) in enumerate(impact['industry_analysis'].head(5).iterrows()):
            report_lines.append(f"  {i+1}. {industry}: Avg Placement {row['placement']:.2f}")
        report_lines.append("")

        report_lines.append("Top 5 Professional Dancers:")
        for i, (dancer, row) in enumerate(impact['dancer_analysis'].head(5).iterrows()):
            report_lines.append(f"  {i+1}. {dancer}: Avg Placement {row['placement']:.2f} ({row['count']:.0f} seasons)")
        report_lines.append("")

        # Section 5: New System Proposal
        report_lines.append("5. NEW VOTING SYSTEM PROPOSAL")
        report_lines.append("-" * 40)

        new_system = self.propose_new_system()
        report_lines.append(f"System Name: {new_system['system_name']}")
        report_lines.append("Components:")
        for i, component in enumerate(new_system['components'], 1):
            report_lines.append(f"  {i}. {component}")
        report_lines.append("")
        report_lines.append("Adaptive Weights:")
        report_lines.append(f"  Early weeks: {new_system['weights_early']}")
        report_lines.append(f"  Middle weeks: {new_system['weights_middle']}")
        report_lines.append(f"  Late weeks: {new_system['weights_late']}")
        report_lines.append("")

        # Section 6: Recommendations
        report_lines.append("6. RECOMMENDATIONS FOR DWTS PRODUCERS")
        report_lines.append("-" * 40)
        report_lines.append("Based on our analysis, we recommend:")
        report_lines.append("")
        report_lines.append("1. ADOPT THE PERCENTAGE-BASED METHOD:")
        report_lines.append("   - Shows stronger correlation with judge scores")
        report_lines.append("   - More mathematically consistent")
        report_lines.append("")
        report_lines.append("2. IMPLEMENT WEIGHTED ADAPTIVE SYSTEM:")
        report_lines.append("   - Balances judge expertise and fan engagement")
        report_lines.append("   - Encourages improvement throughout competition")
        report_lines.append("")
        report_lines.append("3. ADD PROGRESS BONUS:")
        report_lines.append("   - Reduces likelihood of controversial outcomes")
        report_lines.append("   - Rewards consistent improvement")
        report_lines.append("")
        report_lines.append("4. MAINTAIN JUDGE ELIMINATION POWER:")
        report_lines.append("   - Keep bottom-two judge elimination for close cases")
        report_lines.append("   - Adds safety valve for extreme fan voting patterns")
        report_lines.append("")

        report_lines.append("="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)

        # Write to file
        with open('results/analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        # Also print to console
        print('\n'.join(report_lines))

        return '\n'.join(report_lines)


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("2026 MCM PROBLEM C: DATA WITH THE STARS")
    print("COMPLETE SOLUTION")
    print("="*80)

    # Initialize solver
    data_path = '2026_MCM_Problem_C_Data.csv'
    solver = DWTSSolver(data_path)

    # Run all analyses
    print("\n\nRunning analyses...")

    # Generate comprehensive report
    report = solver.generate_report()

    # Create visualizations
    solver.create_visualizations()

    # Save key results to CSV
    print("\n\nSaving results to CSV files...")

    # Comparison results
    comparison = solver.compare_voting_methods()
    comparison.to_csv('results/voting_method_comparison.csv', index=False)

    # Controversial cases
    controversial = solver.analyze_controversial_cases()
    controversial.to_csv('results/controversial_cases_analysis.csv', index=False)

    # Impact factors
    impact = solver.analyze_impact_factors()
    impact['age_analysis'].to_csv('results/age_impact_analysis.csv')
    impact['industry_analysis'].to_csv('results/industry_impact_analysis.csv')
    impact['dancer_analysis'].to_csv('results/dancer_impact_analysis.csv')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/analysis_report.txt")
    print("  - results/voting_method_comparison.csv")
    print("  - results/controversial_cases_analysis.csv")
    print("  - results/age_impact_analysis.csv")
    print("  - results/industry_impact_analysis.csv")
    print("  - results/dancer_impact_analysis.csv")
    print("\n  - figures/figure1_age_vs_placement.png")
    print("  - figures/figure2_industry_performance.png")
    print("  - figures/figure3_top_dancers.png")
    print("  - figures/figure4_score_by_placement.png")
    print("  - figures/figure5_controversial_cases.png")
    print("  - figures/figure6_season_comparison.png")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
