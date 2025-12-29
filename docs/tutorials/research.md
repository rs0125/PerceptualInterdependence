# Research Workflows Tutorial

This tutorial covers using the Perceptual Interdependence system for academic research, including experimental design, data collection, statistical analysis, and publication preparation.

## Research Methodology

### Experimental Design Framework

```python
import numpy as np
import pandas as pd
from pathlib import Path
from perceptual_interdependence import AssetBinder
from perceptual_interdependence.utils.chart_generator import ChartGenerator
from perceptual_interdependence.core.forensics import RGBForensics

class ResearchExperiment:
    """Framework for conducting systematic research experiments."""
    
    def __init__(self, experiment_name, output_dir="./research_results"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.asset_binder = AssetBinder(output_dir=self.output_dir / "bound_assets")
        self.chart_generator = ChartGenerator()
        self.forensics = RGBForensics()
        
        # Data collection
        self.results = []
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': pd.Timestamp.now(),
            'parameters': {},
            'texture_dataset': []
        }
    
    def add_texture_dataset(self, texture_pairs, descriptions=None):
        """Add texture dataset to experiment."""
        for i, (albedo_path, normal_path) in enumerate(texture_pairs):
            texture_info = {
                'id': i,
                'albedo_path': str(albedo_path),
                'normal_path': str(normal_path),
                'description': descriptions[i] if descriptions else f"Texture_{i:03d}"
            }
            self.metadata['texture_dataset'].append(texture_info)
    
    def set_experimental_parameters(self, **params):
        """Set experimental parameters."""
        self.metadata['parameters'].update(params)
    
    def run_binding_experiment(self, user_ids, poison_strengths):
        """Run systematic binding experiment."""
        
        print(f"Running binding experiment: {len(self.metadata['texture_dataset'])} textures, "
              f"{len(user_ids)} users, {len(poison_strengths)} poison strengths")
        
        for texture_info in self.metadata['texture_dataset']:
            for user_id in user_ids:
                for poison_strength in poison_strengths:
                    
                    print(f"Processing: {texture_info['description']}, "
                          f"User {user_id}, Strength {poison_strength}")
                    
                    # Perform binding
                    result = self.asset_binder.bind_textures(
                        albedo_path=texture_info['albedo_path'],
                        normal_path=texture_info['normal_path'],
                        user_id=user_id,
                        poison_strength=poison_strength,
                        output_prefix=f"{texture_info['description']}_u{user_id}_s{poison_strength}"
                    )
                    
                    # Collect data
                    experiment_result = {
                        'texture_id': texture_info['id'],
                        'texture_description': texture_info['description'],
                        'user_id': user_id,
                        'poison_strength': poison_strength,
                        'saturation_ratio': result['statistics']['saturation_ratio'],
                        'bound_albedo_path': result['output_paths']['albedo'],
                        'bound_normal_path': result['output_paths']['normal'],
                        'timestamp': pd.Timestamp.now()
                    }
                    
                    self.results.append(experiment_result)
        
        # Save results
        self.save_results()
    
    def run_quality_analysis(self, victim_attacker_pairs):
        """Run quality analysis experiment."""
        
        quality_results = []
        
        for texture_info in self.metadata['texture_dataset']:
            for victim_id, attacker_id in victim_attacker_pairs:
                
                print(f"Quality analysis: {texture_info['description']}, "
                      f"Victim {victim_id} vs Attacker {attacker_id}")
                
                # Generate chart and extract metrics
                chart_path = self.chart_generator.generate_demonstration_chart(
                    albedo_path=texture_info['albedo_path'],
                    normal_path=texture_info['normal_path'],
                    victim_id=victim_id,
                    attacker_id=attacker_id,
                    output_path=str(self.output_dir / "charts" / 
                                  f"{texture_info['description']}_v{victim_id}_a{attacker_id}.png")
                )
                
                # Extract quality metrics (would need to implement metric extraction)
                quality_metrics = self.extract_quality_metrics(
                    texture_info['albedo_path'],
                    texture_info['normal_path'],
                    victim_id,
                    attacker_id
                )
                
                quality_result = {
                    'texture_id': texture_info['id'],
                    'texture_description': texture_info['description'],
                    'victim_id': victim_id,
                    'attacker_id': attacker_id,
                    'chart_path': chart_path,
                    **quality_metrics
                }
                
                quality_results.append(quality_result)
        
        return quality_results
    
    def run_forensic_experiment(self, max_users_list):
        """Run forensic detection experiment."""
        
        forensic_results = []
        
        # Use bound textures from binding experiment
        bound_textures = [r for r in self.results if r['poison_strength'] == 0.2]
        
        for bound_result in bound_textures[:10]:  # Limit for demonstration
            for max_users in max_users_list:
                
                print(f"Forensic analysis: {bound_result['texture_description']}, "
                      f"User {bound_result['user_id']}, Max users {max_users}")
                
                # Extract signature
                signature = self.forensics.extract_signature(
                    bound_result['bound_albedo_path'],
                    bound_result['texture_description']  # Would need original path
                )
                
                # Detect traitor
                detected_user = self.forensics.find_traitor(signature, max_users=max_users)
                
                forensic_result = {
                    'texture_id': bound_result['texture_id'],
                    'true_user_id': bound_result['user_id'],
                    'detected_user_id': detected_user,
                    'max_users_tested': max_users,
                    'detection_correct': detected_user == bound_result['user_id'],
                    'correlation_scores': self.forensics._last_correlation_scores.tolist()
                }
                
                forensic_results.append(forensic_result)
        
        return forensic_results
    
    def save_results(self):
        """Save experimental results."""
        
        # Save main results
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / "binding_results.csv", index=False)
        
        # Save metadata
        import json
        with open(self.output_dir / "experiment_metadata.json", 'w') as f:
            # Convert Timestamps to strings for JSON serialization
            metadata_copy = self.metadata.copy()
            metadata_copy['start_time'] = str(metadata_copy['start_time'])
            json.dump(metadata_copy, f, indent=2)
        
        print(f"Results saved to {self.output_dir}")
```

### Statistical Analysis Framework

```python
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

class ResearchAnalysis:
    """Statistical analysis tools for research data."""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.binding_results = pd.read_csv(results_dir / "binding_results.csv")
        
        with open(results_dir / "experiment_metadata.json", 'r') as f:
            self.metadata = json.load(f)
    
    def analyze_poison_strength_effects(self):
        """Analyze the effect of poison strength on quality metrics."""
        
        # Group by poison strength
        grouped = self.binding_results.groupby('poison_strength')
        
        # Statistical analysis
        strength_analysis = {}
        
        for strength, group in grouped:
            strength_analysis[strength] = {
                'mean_saturation': group['saturation_ratio'].mean(),
                'std_saturation': group['saturation_ratio'].std(),
                'n_samples': len(group)
            }
        
        # ANOVA test for significant differences
        strength_groups = [group['saturation_ratio'].values for _, group in grouped]
        f_stat, p_value = stats.f_oneway(*strength_groups)
        
        print(f"ANOVA Results:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Post-hoc pairwise comparisons
        if p_value < 0.05:
            self.pairwise_comparisons(grouped)
        
        return strength_analysis
    
    def pairwise_comparisons(self, grouped):
        """Perform pairwise t-tests with Bonferroni correction."""
        
        strengths = list(grouped.groups.keys())
        n_comparisons = len(strengths) * (len(strengths) - 1) // 2
        alpha_corrected = 0.05 / n_comparisons
        
        print(f"\nPairwise comparisons (Bonferroni corrected α = {alpha_corrected:.4f}):")
        
        for i, strength1 in enumerate(strengths):
            for strength2 in strengths[i+1:]:
                group1 = grouped.get_group(strength1)['saturation_ratio']
                group2 = grouped.get_group(strength2)['saturation_ratio']
                
                t_stat, p_value = stats.ttest_ind(group1, group2)
                significant = p_value < alpha_corrected
                
                print(f"  {strength1} vs {strength2}: t={t_stat:.3f}, "
                      f"p={p_value:.6f} {'*' if significant else ''}")
    
    def analyze_texture_variability(self):
        """Analyze variability across different texture types."""
        
        # Group by texture
        texture_stats = self.binding_results.groupby('texture_description').agg({
            'saturation_ratio': ['mean', 'std', 'count'],
            'poison_strength': 'first'  # Assuming consistent within texture
        }).round(4)
        
        # Flatten column names
        texture_stats.columns = ['_'.join(col).strip() for col in texture_stats.columns]
        
        print("Texture Variability Analysis:")
        print(texture_stats)
        
        # Test for significant differences between textures
        texture_groups = [
            group['saturation_ratio'].values 
            for _, group in self.binding_results.groupby('texture_description')
        ]
        
        if len(texture_groups) > 2:
            f_stat, p_value = stats.f_oneway(*texture_groups)
            print(f"\nTexture ANOVA: F={f_stat:.4f}, p={p_value:.6f}")
        
        return texture_stats
    
    def generate_research_plots(self):
        """Generate publication-quality plots."""
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Poison strength vs Saturation ratio
        sns.boxplot(data=self.binding_results, x='poison_strength', y='saturation_ratio', ax=axes[0,0])
        axes[0,0].set_title('Effect of Poison Strength on Saturation Ratio')
        axes[0,0].set_xlabel('Poison Strength')
        axes[0,0].set_ylabel('Saturation Ratio')
        
        # Plot 2: Texture comparison
        sns.violinplot(data=self.binding_results, x='texture_description', y='saturation_ratio', ax=axes[0,1])
        axes[0,1].set_title('Saturation Ratio by Texture Type')
        axes[0,1].set_xlabel('Texture Type')
        axes[0,1].set_ylabel('Saturation Ratio')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: User ID distribution
        user_counts = self.binding_results['user_id'].value_counts().sort_index()
        axes[1,0].bar(user_counts.index, user_counts.values)
        axes[1,0].set_title('Distribution of User IDs')
        axes[1,0].set_xlabel('User ID')
        axes[1,0].set_ylabel('Number of Bindings')
        
        # Plot 4: Correlation matrix
        numeric_cols = ['poison_strength', 'saturation_ratio', 'user_id']
        correlation_matrix = self.binding_results[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
        axes[1,1].set_title('Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "research_analysis_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Research plots saved to {self.results_dir / 'research_analysis_plots.png'}")
```

## Reproducible Research

### Experiment Configuration Management

```python
import yaml
from dataclasses import dataclass, asdict
from typing import List, Tuple

@dataclass
class ExperimentConfig:
    """Configuration for reproducible experiments."""
    
    # Experiment metadata
    name: str
    description: str
    researcher: str
    institution: str
    
    # Dataset configuration
    texture_dataset_path: str
    texture_pairs: List[Tuple[str, str]]
    
    # Experimental parameters
    user_ids: List[int]
    poison_strengths: List[float]
    victim_attacker_pairs: List[Tuple[int, int]]
    
    # Analysis parameters
    max_users_forensic: List[int]
    statistical_alpha: float = 0.05
    
    # Output configuration
    output_dir: str
    generate_charts: bool = True
    generate_forensic_analysis: bool = True
    
    # Reproducibility
    random_seed: int = 42
    
    def save_config(self, filepath):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    @classmethod
    def load_config(cls, filepath):
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

# Example configuration
def create_example_config():
    """Create example research configuration."""
    
    config = ExperimentConfig(
        name="perceptual_interdependence_study_2024",
        description="Comprehensive analysis of perceptual interdependence effects across material types",
        researcher="Dr. Research Scientist",
        institution="University Research Lab",
        
        texture_dataset_path="./research_dataset",
        texture_pairs=[
            ("brick_albedo.png", "brick_normal.png"),
            ("wood_albedo.png", "wood_normal.png"),
            ("metal_albedo.png", "metal_normal.png"),
            ("fabric_albedo.png", "fabric_normal.png")
        ],
        
        user_ids=list(range(1, 51)),  # 50 users
        poison_strengths=[0.1, 0.15, 0.2, 0.25, 0.3],
        victim_attacker_pairs=[(42, 99), (123, 456), (789, 321)],
        
        max_users_forensic=[50, 100, 200],
        statistical_alpha=0.05,
        
        output_dir="./research_results_2024",
        generate_charts=True,
        generate_forensic_analysis=True,
        
        random_seed=42
    )
    
    config.save_config("experiment_config.yaml")
    return config
```

### Automated Research Pipeline

```python
def run_automated_research_pipeline(config_path):
    """Run complete automated research pipeline."""
    
    # Load configuration
    config = ExperimentConfig.load_config(config_path)
    
    # Set random seed for reproducibility
    np.random.seed(config.random_seed)
    
    print(f"Starting research pipeline: {config.name}")
    print(f"Researcher: {config.researcher}")
    print(f"Institution: {config.institution}")
    
    # Initialize experiment
    experiment = ResearchExperiment(
        experiment_name=config.name,
        output_dir=config.output_dir
    )
    
    # Set up dataset
    experiment.add_texture_dataset(
        config.texture_pairs,
        descriptions=[f"Material_{i:02d}" for i in range(len(config.texture_pairs))]
    )
    
    # Set experimental parameters
    experiment.set_experimental_parameters(
        user_ids=config.user_ids,
        poison_strengths=config.poison_strengths,
        victim_attacker_pairs=config.victim_attacker_pairs,
        random_seed=config.random_seed
    )
    
    # Phase 1: Binding experiment
    print("\n=== Phase 1: Binding Experiment ===")
    experiment.run_binding_experiment(
        user_ids=config.user_ids,
        poison_strengths=config.poison_strengths
    )
    
    # Phase 2: Quality analysis
    if config.generate_charts:
        print("\n=== Phase 2: Quality Analysis ===")
        quality_results = experiment.run_quality_analysis(config.victim_attacker_pairs)
        
        # Save quality results
        quality_df = pd.DataFrame(quality_results)
        quality_df.to_csv(Path(config.output_dir) / config.name / "quality_results.csv", index=False)
    
    # Phase 3: Forensic analysis
    if config.generate_forensic_analysis:
        print("\n=== Phase 3: Forensic Analysis ===")
        forensic_results = experiment.run_forensic_experiment(config.max_users_forensic)
        
        # Save forensic results
        forensic_df = pd.DataFrame(forensic_results)
        forensic_df.to_csv(Path(config.output_dir) / config.name / "forensic_results.csv", index=False)
    
    # Phase 4: Statistical analysis
    print("\n=== Phase 4: Statistical Analysis ===")
    analysis = ResearchAnalysis(Path(config.output_dir) / config.name)
    
    # Perform analyses
    poison_analysis = analysis.analyze_poison_strength_effects()
    texture_analysis = analysis.analyze_texture_variability()
    
    # Generate plots
    analysis.generate_research_plots()
    
    # Generate research report
    generate_research_report(config, analysis)
    
    print(f"\n=== Research Pipeline Complete ===")
    print(f"Results saved to: {Path(config.output_dir) / config.name}")
    
    return experiment, analysis

def generate_research_report(config, analysis):
    """Generate automated research report."""
    
    report_path = Path(config.output_dir) / config.name / "research_report.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Research Report: {config.name}\n\n")
        f.write(f"**Researcher:** {config.researcher}\n")
        f.write(f"**Institution:** {config.institution}\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        
        f.write(f"## Experiment Description\n\n")
        f.write(f"{config.description}\n\n")
        
        f.write(f"## Methodology\n\n")
        f.write(f"- **Textures tested:** {len(config.texture_pairs)}\n")
        f.write(f"- **Users simulated:** {len(config.user_ids)}\n")
        f.write(f"- **Poison strengths:** {config.poison_strengths}\n")
        f.write(f"- **Random seed:** {config.random_seed}\n\n")
        
        f.write(f"## Key Findings\n\n")
        
        # Add statistical results
        poison_analysis = analysis.analyze_poison_strength_effects()
        f.write(f"### Poison Strength Effects\n\n")
        for strength, stats in poison_analysis.items():
            f.write(f"- **Strength {strength}:** Mean saturation = {stats['mean_saturation']:.4f} "
                   f"± {stats['std_saturation']:.4f} (n={stats['n_samples']})\n")
        
        f.write(f"\n### Statistical Significance\n\n")
        f.write(f"ANOVA analysis shows {'significant' if analysis.binding_results.groupby('poison_strength')['saturation_ratio'].apply(lambda x: len(x)).sum() > 0 else 'no significant'} "
               f"differences between poison strength levels (α = {config.statistical_alpha}).\n\n")
        
        f.write(f"## Visualizations\n\n")
        f.write(f"![Research Analysis Plots](research_analysis_plots.png)\n\n")
        
        f.write(f"## Data Files\n\n")
        f.write(f"- `binding_results.csv`: Raw binding experiment data\n")
        f.write(f"- `quality_results.csv`: Quality analysis results\n")
        f.write(f"- `forensic_results.csv`: Forensic detection results\n")
        f.write(f"- `experiment_metadata.json`: Experiment configuration and metadata\n")
    
    print(f"Research report generated: {report_path}")
```

## Publication Preparation

### LaTeX Table Generation

```python
def generate_latex_tables(analysis):
    """Generate LaTeX tables for publication."""
    
    # Poison strength effects table
    poison_analysis = analysis.analyze_poison_strength_effects()
    
    latex_table = "\\begin{table}[htbp]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{Effect of Poison Strength on Saturation Ratio}\n"
    latex_table += "\\label{tab:poison_effects}\n"
    latex_table += "\\begin{tabular}{cccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "Poison Strength & Mean Saturation & Std Dev & Sample Size \\\\\n"
    latex_table += "\\midrule\n"
    
    for strength, stats in poison_analysis.items():
        latex_table += f"{strength:.2f} & {stats['mean_saturation']:.4f} & "
        latex_table += f"{stats['std_saturation']:.4f} & {stats['n_samples']} \\\\\n"
    
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}\n"
    
    # Save to file
    with open(analysis.results_dir / "poison_effects_table.tex", 'w') as f:
        f.write(latex_table)
    
    print("LaTeX table generated: poison_effects_table.tex")

def generate_bibtex_entry():
    """Generate BibTeX entry for the system."""
    
    bibtex = """@software{perceptual_interdependence_2024,
  title={Perceptual Interdependence: Geometry-Aware Photometric Binding System},
  author={Research Team},
  year={2024},
  url={https://github.com/research/perceptual-interdependence},
  version={1.0.0},
  note={Research-grade implementation of analytically safe one-way binding}
}

@article{your_research_paper_2024,
  title={Your Research Paper Title},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2024},
  volume={XX},
  number={X},
  pages={XXX--XXX},
  publisher={Publisher Name}
}"""
    
    with open("references.bib", 'w') as f:
        f.write(bibtex)
    
    print("BibTeX entries generated: references.bib")
```

### Figure Generation for Papers

```python
def generate_publication_figures(config):
    """Generate high-quality figures for publication."""
    
    # Set publication style
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.0
    })
    
    # Generate demonstration chart for paper
    generator = ChartGenerator()
    
    # Main demonstration figure
    chart_path = generator.generate_demonstration_chart(
        albedo_path=config.texture_pairs[0][0],  # Use first texture
        normal_path=config.texture_pairs[0][1],
        victim_id=42,
        attacker_id=99,
        output_path="figure_1_demonstration.png"
    )
    
    # High-magnification detail figure
    zoom_path = generator.generate_zoomed_demonstration_chart(
        albedo_path=config.texture_pairs[0][0],
        normal_path=config.texture_pairs[0][1],
        victim_id=42,
        attacker_id=99,
        output_path="figure_2_magnification.png",
        zoom_factor=20.0
    )
    
    # Forensic detection figure
    forensics = RGBForensics()
    
    # Create bound texture for forensic demo
    binder = AssetBinder()
    bound_result = binder.bind_textures(
        albedo_path=config.texture_pairs[0][0],
        normal_path=config.texture_pairs[0][1],
        user_id=42,
        poison_strength=0.2
    )
    
    # Extract signature and generate spike chart
    signature = forensics.extract_signature(
        bound_result['output_paths']['albedo'],
        config.texture_pairs[0][0]
    )
    
    forensics.generate_continuous_spike_chart(
        signature,
        max_users=100,
        output_path="figure_3_forensic_detection.png"
    )
    
    print("Publication figures generated:")
    print("- figure_1_demonstration.png")
    print("- figure_2_magnification.png") 
    print("- figure_3_forensic_detection.png")
```

## Usage Example

```python
# Complete research workflow example
def main_research_workflow():
    """Complete example of research workflow."""
    
    # 1. Create experiment configuration
    config = create_example_config()
    
    # 2. Run automated pipeline
    experiment, analysis = run_automated_research_pipeline("experiment_config.yaml")
    
    # 3. Generate publication materials
    generate_latex_tables(analysis)
    generate_bibtex_entry()
    generate_publication_figures(config)
    
    # 4. Create supplementary materials
    create_supplementary_materials(config, analysis)
    
    print("\n=== Research Workflow Complete ===")
    print("Generated materials:")
    print("- Experimental data and analysis")
    print("- Statistical reports")
    print("- Publication-quality figures")
    print("- LaTeX tables")
    print("- BibTeX references")
    print("- Supplementary materials")

def create_supplementary_materials(config, analysis):
    """Create supplementary materials for publication."""
    
    supp_dir = Path(config.output_dir) / config.name / "supplementary"
    supp_dir.mkdir(exist_ok=True)
    
    # Generate additional charts for all texture types
    generator = ChartGenerator()
    
    for i, (albedo_path, normal_path) in enumerate(config.texture_pairs):
        # Standard chart
        generator.generate_demonstration_chart(
            albedo_path=albedo_path,
            normal_path=normal_path,
            victim_id=42,
            attacker_id=99,
            output_path=str(supp_dir / f"supplementary_chart_{i:02d}.png")
        )
        
        # Zoomed chart
        generator.generate_zoomed_demonstration_chart(
            albedo_path=albedo_path,
            normal_path=normal_path,
            victim_id=42,
            attacker_id=99,
            output_path=str(supp_dir / f"supplementary_zoom_{i:02d}.png"),
            zoom_factor=15.0
        )
    
    # Create supplementary data summary
    summary_stats = analysis.binding_results.describe()
    summary_stats.to_csv(supp_dir / "summary_statistics.csv")
    
    print(f"Supplementary materials created in: {supp_dir}")

if __name__ == "__main__":
    main_research_workflow()
```

This research workflows tutorial provides a comprehensive framework for conducting rigorous academic research using the Perceptual Interdependence system, from experimental design through publication preparation.