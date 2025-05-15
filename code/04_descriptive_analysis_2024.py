"""
Descriptive Analysis for StackOverflow HRD 2024 Research
This script performs comprehensive descriptive analysis on the preprocessed dataset
to understand the structure, distribution, and relationships within the data.

The results support the main research questions by providing context and foundation
for the more advanced analyses that follow.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import json
import pickle
from scipy import stats
import matplotlib.ticker as mtick

# Set the visual style for plots
plt.style.use('fivethirtyeight')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Paths
BASE_DIR = Path("/Volumes/External SSD/Pycharm/StackOverFlow_HRD_2024")
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "Reports_0425" / "Research Reports" / "images"
FIGURE_DIR = BASE_DIR / "Reports_0425" / "figures"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load preprocessed data
try:
    # Try to load the parquet file
    df = pd.read_parquet(DATA_DIR / "processed_survey_2024.parquet")
    print(f"Loaded preprocessed data with {df.shape[0]} rows and {df.shape[1]} columns")
except FileNotFoundError:
    # Try alternative filename
    try:
        df = pd.read_parquet(DATA_DIR / "survey_processed_2024.parquet")
        print(f"Loaded preprocessed data with {df.shape[0]} rows and {df.shape[1]} columns")
    except FileNotFoundError:
        print("Preprocessed data not found. Please ensure the processed data files exist.")
        exit(1)

# Function to save figures
def save_figure(fig, filename):
    """Save figure to both the figures directory and the report images directory"""
    fig.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filename}")

def descriptive_statistics():
    """Generate basic descriptive statistics for numerical variables"""
    # Select numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    # Basic statistics
    stats_df = df[num_cols].describe().T
    
    # Add additional statistics
    stats_df['median'] = df[num_cols].median()
    stats_df['skewness'] = df[num_cols].skew()
    stats_df['kurtosis'] = df[num_cols].kurtosis()
    stats_df['missing'] = df[num_cols].isnull().sum()
    stats_df['missing_pct'] = (df[num_cols].isnull().sum() / len(df)) * 100
    
    # Save statistics to CSV
    stats_df.to_csv(OUTPUT_DIR / "numerical_descriptive_stats.csv")
    print("Saved numerical descriptive statistics")
    
    return stats_df

def categorical_analysis():
    """Analyze categorical variables and their distributions"""
    # Select categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Create a DataFrame to store categorical statistics
    cat_stats = []
    
    for col in cat_cols:
        # Get value counts and proportions
        vc = df[col].value_counts()
        prop = df[col].value_counts(normalize=True) * 100
        
        # Create a summary for this column
        summary = {
            'variable': col,
            'unique_values': df[col].nunique(),
            'missing': df[col].isnull().sum(),
            'missing_pct': (df[col].isnull().sum() / len(df)) * 100,
            'most_common': vc.index[0] if not vc.empty else None,
            'most_common_pct': prop.iloc[0] if not prop.empty else None,
            'least_common': vc.index[-1] if not vc.empty and len(vc) > 1 else None,
            'least_common_pct': prop.iloc[-1] if not prop.empty and len(prop) > 1 else None
        }
        
        cat_stats.append(summary)
    
    # Convert to DataFrame and save
    cat_stats_df = pd.DataFrame(cat_stats)
    cat_stats_df.to_csv(OUTPUT_DIR / "categorical_descriptive_stats.csv")
    print("Saved categorical descriptive statistics")
    
    # Create visualizations for key categorical variables
    key_categories = ['DevType', 'Country', 'EdLevel', 'YearsCodePro']
    key_categories = [col for col in key_categories if col in df.columns]
    
    for col in key_categories:
        plt.figure(figsize=(14, 8))
        
        # Get the top 15 categories by count
        top_cats = df[col].value_counts().nlargest(15)
        
        # Create bar plot
        ax = sns.barplot(x=top_cats.index, y=top_cats.values)
        plt.title(f'Distribution of {col} (Top 15)')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Count')
        
        # Add percentage labels
        for i, v in enumerate(top_cats.values):
            ax.text(i, v + 0.1, f"{v/len(df)*100:.1f}%", ha='center')
        
        plt.tight_layout()
        save_figure(plt.gcf(), f"{col}_distribution.png")
        plt.close()
    
    return cat_stats_df

def compensation_analysis():
    """Analyze compensation distributions and relationships"""
    # Check if compensation column exists (name might vary)
    comp_col = None
    for col in df.columns:
        if 'comp' in col.lower() or 'salary' in col.lower():
            comp_col = col
            break
    
    if comp_col is None:
        print("No compensation column found in the dataset")
        return
    
    # Convert compensation to numeric if it's a string
    if df[comp_col].dtype == 'object':
        try:
            # Try to convert, handling currency symbols and commas
            df['comp_numeric'] = df[comp_col].str.replace(r'[^\d.]', '', regex=True).astype(float)
            comp_col = 'comp_numeric'
            print(f"Converted {comp_col} to numeric values")
        except Exception as e:
            print(f"Could not convert compensation to numeric: {e}")
            return
    
    # Create a histogram of compensation
    plt.figure(figsize=(12, 8))
    sns.histplot(df[comp_col].dropna(), kde=True, bins=50)
    plt.title(f'Distribution of {comp_col}')
    plt.xlabel(comp_col)
    plt.ylabel('Count')
    save_figure(plt.gcf(), "compensation_distribution.png")
    plt.close()
    
    # Create log-transformed compensation histogram (commonly used in salary analysis)
    plt.figure(figsize=(12, 8))
    # Add small value to avoid log(0)
    sns.histplot(np.log1p(df[comp_col].dropna().astype(float)), kde=True, bins=50)
    plt.title(f'Log-transformed Distribution of {comp_col}')
    plt.xlabel(f'Log({comp_col})')
    plt.ylabel('Count')
    save_figure(plt.gcf(), "log_compensation_distribution.png")
    plt.close()
    
    # Compensation by experience (if available)
    exp_col = None
    for col in df.columns:
        if 'years' in col.lower() and 'code' in col.lower():
            exp_col = col
            break
    
    if exp_col:
        # Create boxplot of compensation by experience
        plt.figure(figsize=(14, 8))
        
        # Convert experience to categorical bins if it's numerical
        if df[exp_col].dtype in [np.int64, np.float64]:
            # Create experience bins
            bins = [0, 3, 5, 10, 15, 20, 100]
            labels = ['0-3', '3-5', '5-10', '10-15', '15-20', '20+']
            df['exp_bins'] = pd.cut(df[exp_col], bins=bins, labels=labels)
            
            # Create boxplot
            sns.boxplot(x='exp_bins', y=comp_col, data=df)
            plt.title(f'{comp_col} by Experience')
            
        else:
            # If already categorical, just use the column directly
            sns.boxplot(x=exp_col, y=comp_col, data=df)
            plt.title(f'{comp_col} by {exp_col}')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_figure(plt.gcf(), "compensation_by_experience.png")
        plt.close()
    
    # Compensation by education level (if available)
    edu_col = None
    for col in df.columns:
        if 'ed' in col.lower() or 'education' in col.lower():
            edu_col = col
            break
    
    if edu_col:
        plt.figure(figsize=(14, 8))
        sns.boxplot(x=edu_col, y=comp_col, data=df)
        plt.title(f'{comp_col} by Education Level')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_figure(plt.gcf(), "compensation_by_education.png")
        plt.close()

def skill_distribution_analysis():
    """Analyze the distribution of technical skills in the dataset"""
    # Find columns that might contain skill information
    skill_cols = []
    
    # Common patterns in skill column names
    skill_patterns = ['language', 'tech', 'framework', 'tool', 'skill', 'platform']
    
    for col in df.columns:
        col_lower = col.lower()
        # Check if any pattern matches or if it's a binary column with mostly 0s and 1s
        if any(pattern in col_lower for pattern in skill_patterns) or (
                df[col].dtype == bool or 
                (df[col].dtype in [np.int64, np.float64] and set(df[col].unique()).issubset({0, 1, np.nan}))):
            skill_cols.append(col)
    
    # If we have too many columns, filter to binary columns that likely represent skills
    if len(skill_cols) > 100:
        skill_cols = [col for col in skill_cols if df[col].dtype == bool or 
                     (df[col].dtype in [np.int64, np.float64] and set(df[col].unique()).issubset({0, 1, np.nan}))]
    
    # Take top 30 most common skills for visualization
    if skill_cols:
        # Calculate percentage of developers with each skill
        skill_percentages = {}
        for col in skill_cols:
            # For binary columns
            if df[col].dtype in [bool, np.int64, np.float64]:
                # Count True or 1 values and calculate percentage
                skill_percentages[col] = (df[col] == 1).mean() * 100
            else:
                # For non-binary columns, count non-null values
                skill_percentages[col] = df[col].notna().mean() * 100
        
        # Convert to DataFrame for easier handling
        skill_df = pd.DataFrame(list(skill_percentages.items()), columns=['Skill', 'Percentage'])
        skill_df = skill_df.sort_values('Percentage', ascending=False).head(30)
        
        # Create horizontal bar chart
        plt.figure(figsize=(14, 10))
        ax = sns.barplot(x='Percentage', y='Skill', data=skill_df)
        
        # Add percentage labels
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            plt.text(width + 1, p.get_y() + p.get_height()/2, f'{width:.1f}%', va='center')
        
        plt.title('Percentage of Developers with Each Skill (Top 30)')
        plt.xlabel('Percentage (%)')
        plt.tight_layout()
        save_figure(plt.gcf(), "top_skills_distribution.png")
        plt.close()
        
        # Save the skill distribution data
        skill_df.to_csv(OUTPUT_DIR / "skill_distribution.csv", index=False)
        print("Saved skill distribution data")

def learning_methods_analysis():
    """Analyze the distribution and relationships of learning methods"""
    # Find columns related to learning methods
    learning_cols = []
    learning_patterns = ['learn', 'educat', 'train', 'course', 'study', 'mentor', 'bootcamp']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in learning_patterns):
            learning_cols.append(col)
    
    if learning_cols:
        # Calculate percentage for each learning method
        learning_percentages = {}
        for col in learning_cols:
            if df[col].dtype in [bool, np.int64, np.float64]:
                # Count True or 1 values
                learning_percentages[col] = (df[col] == 1).mean() * 100
            else:
                # For non-binary columns, count non-null values
                learning_percentages[col] = df[col].notna().mean() * 100
        
        # Convert to DataFrame
        learning_df = pd.DataFrame(list(learning_percentages.items()), 
                                   columns=['Learning Method', 'Percentage'])
        learning_df = learning_df.sort_values('Percentage', ascending=False)
        
        # Create horizontal bar chart
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='Percentage', y='Learning Method', data=learning_df)
        
        # Add percentage labels
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            plt.text(width + 1, p.get_y() + p.get_height()/2, f'{width:.1f}%', va='center')
        
        plt.title('Percentage of Developers Using Each Learning Method')
        plt.xlabel('Percentage (%)')
        plt.tight_layout()
        save_figure(plt.gcf(), "learning_methods_distribution.png")
        plt.close()
        
        # Save the learning methods distribution data
        learning_df.to_csv(OUTPUT_DIR / "learning_methods_distribution.csv", index=False)
        print("Saved learning methods distribution data")
        
        # If we have experience data, analyze learning methods by experience level
        exp_col = None
        for col in df.columns:
            if 'years' in col.lower() and 'code' in col.lower():
                exp_col = col
                break
        
        if exp_col and df[exp_col].dtype in [np.int64, np.float64]:
            # Create experience bins
            bins = [0, 3, 5, 10, 15, 20, 100]
            labels = ['0-3', '3-5', '5-10', '10-15', '15-20', '20+']
            df['exp_bins'] = pd.cut(df[exp_col], bins=bins, labels=labels)
            
            # Analyze top learning methods by experience
            plt.figure(figsize=(16, 10))
            
            # Take top 5 learning methods
            top_methods = learning_df.head(5)['Learning Method'].tolist()
            
            # Create data for plotting
            plot_data = []
            for method in top_methods:
                if df[method].dtype in [bool, np.int64, np.float64]:
                    for exp in labels:
                        pct = (df[df['exp_bins'] == exp][method] == 1).mean() * 100
                        plot_data.append({'Learning Method': method, 'Experience': exp, 'Percentage': pct})
            
            # Create plot
            plot_df = pd.DataFrame(plot_data)
            sns.barplot(x='Experience', y='Percentage', hue='Learning Method', data=plot_df)
            plt.title('Learning Methods by Experience Level')
            plt.xlabel('Years of Experience')
            plt.ylabel('Percentage Using Method (%)')
            plt.legend(title='Learning Method', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            save_figure(plt.gcf(), "learning_methods_by_experience.png")
            plt.close()

def correlation_analysis():
    """Analyze correlations between numerical variables"""
    # Select numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out binary columns (likely one-hot encoded)
    non_binary_cols = []
    for col in num_cols:
        if len(df[col].dropna().unique()) > 2:
            non_binary_cols.append(col)
    
    # If we have too many columns, select a manageable subset
    if len(non_binary_cols) > 15:
        # Try to find important columns
        important_patterns = ['salary', 'comp', 'years', 'age', 'experience', 'satisf', 'hours']
        selected_cols = []
        
        for pattern in important_patterns:
            matching = [col for col in non_binary_cols if pattern in col.lower()]
            selected_cols.extend(matching)
        
        # Add more columns if needed to reach around 10-15 total
        if len(selected_cols) < 10:
            remaining = [col for col in non_binary_cols if col not in selected_cols]
            selected_cols.extend(remaining[:15-len(selected_cols)])
        
        non_binary_cols = selected_cols[:15]  # Limit to 15 columns
    
    if len(non_binary_cols) > 1:
        # Calculate correlation matrix
        corr_matrix = df[non_binary_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask for upper triangle
        sns.heatmap(corr_matrix, mask=mask, cmap='viridis', annot=True, fmt='.2f', 
                    linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numerical Variables')
        plt.tight_layout()
        save_figure(plt.gcf(), "correlation_matrix.png")
        plt.close()
        
        # Save correlation matrix to CSV
        corr_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
        print("Saved correlation matrix")

def main():
    """Run all descriptive analysis functions and compile results"""
    print("Starting comprehensive descriptive analysis...")
    
    # Run all analyses
    num_stats = descriptive_statistics()
    cat_stats = categorical_analysis()
    compensation_analysis()
    skill_distribution_analysis()
    learning_methods_analysis()
    correlation_analysis()
    
    # Create a summary report
    summary = {
        "dataset_size": {
            "rows": df.shape[0],
            "columns": df.shape[1]
        },
        "numeric_variables": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_variables": len(df.select_dtypes(include=['object', 'category']).columns),
        "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d")
    }
    
    # Save summary as JSON
    with open(OUTPUT_DIR / "descriptive_analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("Descriptive analysis completed successfully!")
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
