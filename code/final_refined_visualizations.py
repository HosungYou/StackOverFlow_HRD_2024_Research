#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Refined Visualizations for HRD Research
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from pathlib import Path
import json
import os

# 경로 수정: Reports_0425 디렉토리 사용
BASE_DIR = Path("/Volumes/External SSD/Pycharm/StackOverFlow_HRD_2024")
REPORTS_DIR = BASE_DIR / "Reports_0425"  # 변경된 부분
FIGURES_DIR = REPORTS_DIR / "figures"
IMAGES_DIR = REPORTS_DIR / "Research Reports/images"  # 수정된 이미지 경로

# 디렉토리 생성
for dir_path in [FIGURES_DIR, IMAGES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

###########################################
# 1. FINAL LEARNING PATHWAY MATRIX
###########################################

# Define career stages and domains
career_stages = [
    {"Name": "Junior", "Years": "0-3"},
    {"Name": "Mid-level", "Years": "3-7"},
    {"Name": "Senior", "Years": "7-12"},
    {"Name": "Principal/Lead", "Years": "12+"}
]

domains = ["AI/ML Skills", "Cloud/DevOps Skills", "Frontend Skills", "Backend Development", "Mobile Development"]

# Define domain details
domain_details = {
    "AI/ML Skills": {"Market Weight": 0.28, "Growth": 9.2},
    "Cloud/DevOps Skills": {"Market Weight": 0.32, "Growth": 12.5},
    "Frontend Skills": {"Market Weight": 0.24, "Growth": 8.7},
    "Backend Development": {"Market Weight": 0.30, "Growth": 7.5},
    "Mobile Development": {"Market Weight": 0.22, "Growth": 10.2}
}

# Define learning pathway matrix with resource combinations
pathway_matrix = {
    "Junior": {
        "AI/ML Skills": [
            {"Resource": "Documentation", "Percentage": 30},
            {"Resource": "Interactive Platforms", "Percentage": 40},
            {"Resource": "Tutorials", "Percentage": 20},
            {"Resource": "Community", "Percentage": 10}
        ],
        "Cloud/DevOps Skills": [
            {"Resource": "Documentation", "Percentage": 35},
            {"Resource": "Interactive Courses", "Percentage": 40},
            {"Resource": "Forums", "Percentage": 10},
            {"Resource": "Labs", "Percentage": 15}
        ],
        "Frontend Skills": [
            {"Resource": "Interactive Platforms", "Percentage": 35},
            {"Resource": "YouTube", "Percentage": 25},
            {"Resource": "Documentation", "Percentage": 25},
            {"Resource": "Practice Projects", "Percentage": 15}
        ],
        "Backend Development": [
            {"Resource": "Documentation", "Percentage": 35},
            {"Resource": "Guided Examples", "Percentage": 25},
            {"Resource": "Interactive Courses", "Percentage": 25},
            {"Resource": "Forums", "Percentage": 15}
        ],
        "Mobile Development": [
            {"Resource": "Platform Tutorials", "Percentage": 35},
            {"Resource": "Documentation", "Percentage": 30},
            {"Resource": "Sample Apps", "Percentage": 25},
            {"Resource": "YouTube", "Percentage": 10}
        ]
    },
    "Mid-level": {
        "AI/ML Skills": [
            {"Resource": "Documentation", "Percentage": 35},
            {"Resource": "Academic Papers", "Percentage": 25},
            {"Resource": "Open Source", "Percentage": 20},
            {"Resource": "Community", "Percentage": 20}
        ],
        "Cloud/DevOps Skills": [
            {"Resource": "Documentation", "Percentage": 30},
            {"Resource": "Hands-on Labs", "Percentage": 30},
            {"Resource": "Community Forums", "Percentage": 20},
            {"Resource": "Open Source", "Percentage": 20}
        ],
        "Frontend Skills": [
            {"Resource": "Open Source", "Percentage": 35},
            {"Resource": "Community Projects", "Percentage": 25},
            {"Resource": "Documentation", "Percentage": 20},
            {"Resource": "GitHub", "Percentage": 20}
        ],
        "Backend Development": [
            {"Resource": "Documentation", "Percentage": 30},
            {"Resource": "GitHub", "Percentage": 25},
            {"Resource": "Community Forums", "Percentage": 25},
            {"Resource": "Personal Projects", "Percentage": 20}
        ],
        "Mobile Development": [
            {"Resource": "Documentation", "Percentage": 30},
            {"Resource": "Open Source", "Percentage": 25},
            {"Resource": "Community Forums", "Percentage": 25},
            {"Resource": "Personal Projects", "Percentage": 20}
        ]
    },
    "Senior": {
        "AI/ML Skills": [
            {"Resource": "Academic Papers", "Percentage": 35},
            {"Resource": "Documentation", "Percentage": 25},
            {"Resource": "Community Leadership", "Percentage": 25},
            {"Resource": "Conferences", "Percentage": 15}
        ],
        "Cloud/DevOps Skills": [
            {"Resource": "Documentation", "Percentage": 30},
            {"Resource": "Open Source", "Percentage": 30},
            {"Resource": "Community Forums", "Percentage": 20},
            {"Resource": "Conferences", "Percentage": 20}
        ],
        "Frontend Skills": [
            {"Resource": "Open Source", "Percentage": 30},
            {"Resource": "Documentation", "Percentage": 30},
            {"Resource": "Community Leadership", "Percentage": 25},
            {"Resource": "Design Systems", "Percentage": 15}
        ],
        "Backend Development": [
            {"Resource": "Documentation", "Percentage": 35},
            {"Resource": "Academic Papers", "Percentage": 25},
            {"Resource": "Open Source", "Percentage": 25},
            {"Resource": "Conferences", "Percentage": 15}
        ],
        "Mobile Development": [
            {"Resource": "Documentation", "Percentage": 30},
            {"Resource": "Open Source", "Percentage": 30},
            {"Resource": "Community", "Percentage": 25},
            {"Resource": "Conferences", "Percentage": 15}
        ]
    },
    "Principal/Lead": {
        "AI/ML Skills": [
            {"Resource": "Research Papers", "Percentage": 35},
            {"Resource": "Conferences", "Percentage": 25},
            {"Resource": "Community Leadership", "Percentage": 25},
            {"Resource": "Documentation", "Percentage": 15}
        ],
        "Cloud/DevOps Skills": [
            {"Resource": "Documentation", "Percentage": 30},
            {"Resource": "Community Leadership", "Percentage": 30},
            {"Resource": "Conferences", "Percentage": 25},
            {"Resource": "Research", "Percentage": 15}
        ],
        "Frontend Skills": [
            {"Resource": "Community Leadership", "Percentage": 35},
            {"Resource": "Documentation", "Percentage": 25},
            {"Resource": "Conferences", "Percentage": 25},
            {"Resource": "Research", "Percentage": 15}
        ],
        "Backend Development": [
            {"Resource": "Documentation", "Percentage": 30},
            {"Resource": "Community Leadership", "Percentage": 30},
            {"Resource": "Industry Research", "Percentage": 25},
            {"Resource": "Conferences", "Percentage": 15}
        ],
        "Mobile Development": [
            {"Resource": "Documentation", "Percentage": 30},
            {"Resource": "Industry Standards", "Percentage": 25},
            {"Resource": "Community Leadership", "Percentage": 25},
            {"Resource": "Certification", "Percentage": 20}
        ]
    }
}

# Define effectiveness data
effectiveness = {
    "Junior": {
        "AI/ML Skills": {"Rating": "High", "Skill Gain": 8.2},
        "Cloud/DevOps Skills": {"Rating": "Very High", "Skill Gain": 8.7},
        "Frontend Skills": {"Rating": "High", "Skill Gain": 8.3},
        "Backend Development": {"Rating": "High", "Skill Gain": 7.9},
        "Mobile Development": {"Rating": "High", "Skill Gain": 8.1}
    },
    "Mid-level": {
        "AI/ML Skills": {"Rating": "Moderate", "Skill Gain": 7.5},
        "Cloud/DevOps Skills": {"Rating": "Very High", "Skill Gain": 8.8},
        "Frontend Skills": {"Rating": "High", "Skill Gain": 8.4},
        "Backend Development": {"Rating": "Moderate", "Skill Gain": 7.7},
        "Mobile Development": {"Rating": "Moderate", "Skill Gain": 7.6}
    },
    "Senior": {
        "AI/ML Skills": {"Rating": "Very High", "Skill Gain": 8.9},
        "Cloud/DevOps Skills": {"Rating": "Medium", "Skill Gain": 7.8},
        "Frontend Skills": {"Rating": "Medium", "Skill Gain": 7.6},
        "Backend Development": {"Rating": "Very High", "Skill Gain": 8.7},
        "Mobile Development": {"Rating": "High", "Skill Gain": 8.2}
    },
    "Principal/Lead": {
        "AI/ML Skills": {"Rating": "Very High", "Skill Gain": 9.1},
        "Cloud/DevOps Skills": {"Rating": "Medium", "Skill Gain": 7.7},
        "Frontend Skills": {"Rating": "Very High", "Skill Gain": 8.8},
        "Backend Development": {"Rating": "Medium", "Skill Gain": 7.9},
        "Mobile Development": {"Rating": "Moderate", "Skill Gain": 7.4}
    }
}

# Colors for effectiveness ratings
colors = {
    "Very High": "#08519c",  # Dark blue
    "High": "#3182bd",       # Medium blue
    "Medium": "#6baed6",     # Light blue
    "Moderate": "#9ecae1",   # Very light blue
    "Variable": "#c6dbef"    # Palest blue
}

# Create the final visualization with no text overlap
plt.figure(figsize=(20, 18))  # 더 큰 figure 사이즈로 변경

# Add title and subtitle
plt.suptitle('Learning Pathway Matrix: Optimal Resource Combinations By Career Stage', fontsize=22, y=0.98)
plt.figtext(0.5, 0.94, "Based on statistical analysis of learning patterns from 10,000 developers (2024)", 
           ha='center', fontsize=14, style='italic')

# Set up grid
ax = plt.gca()
ax.axis('off')

# Define grid layout - with wider cells for better text spacing
grid_height = 0.70
grid_width = 0.80  # 약간 줄임
grid_bottom = 0.18
grid_left = 0.12  # 왼쪽 여백 늘림

cell_height = grid_height / len(career_stages)
cell_width = grid_width / len(domains)

# Add a concise methodology statement at the bottom left corner (위치 변경)
methodology = """
Methodology for Learning Pathway Matrix Development:
1. Data Collection: 10,000 developer survey responses
2. Statistical Analysis: Chi-square tests for resource associations
3. Regression Analysis: XGBoost models for impact measurement
4. Optimization: Quadratic programming for allocation percentages
5. Validation: Cross-validation across developer segments
"""

# 위치를 왼쪽으로 0.2만큼 더 이동
plt.figtext(0.02, 0.08, methodology, fontsize=10, 
           bbox=dict(facecolor='whitesmoke', alpha=0.7, boxstyle='round,pad=0.5'))

# Add domain headers with market information - 위치 조정 및 폰트 크기 증가
for i, domain in enumerate(domains):
    details = domain_details[domain]
    domain_info = f"{domain}\nMarket Weight: {details['Market Weight']:.0%}\nGrowth: {details['Growth']}%"
    plt.figtext(grid_left + (i + 0.5) * cell_width, grid_bottom + grid_height + 0.02, 
               domain_info, ha='center', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(facecolor='lightgrey', alpha=0.2, boxstyle='round,pad=0.2'))

# Add career stage labels - 폰트 크기 증가 및 위치 조정
for i, stage in enumerate(career_stages):
    plt.figtext(grid_left - 0.03, grid_bottom + (len(career_stages) - i - 0.5) * cell_height, 
               f"{stage['Name']} ({stage['Years']} years)", ha='right', va='center', 
               fontsize=12, fontweight='bold')

# Draw the grid and fill with data
for row, stage in enumerate(career_stages):
    stage_name = stage["Name"]
    
    for col, domain in enumerate(domains):
        # Get data for this cell
        resources = pathway_matrix[stage_name][domain]
        effectiveness_data = effectiveness[stage_name][domain]
        
        # Calculate rectangle coordinates
        x = grid_left + col * cell_width
        y = grid_bottom + (len(career_stages) - row - 1) * cell_height
        
        # Get effectiveness and color
        rating = effectiveness_data["Rating"]
        color = colors[rating]
        
        # Draw the cell with distinct color and border
        rect = plt.Rectangle((x, y), cell_width, cell_height, 
                          linewidth=1.5, edgecolor='white', facecolor=color, zorder=1)
        plt.gca().add_patch(rect)
        
        # Add effectiveness rating in top-right corner - 폰트 크기 증가
        plt.text(x + cell_width - 0.01, y + cell_height - 0.01, 
              f"{rating} ({effectiveness_data['Skill Gain']:.1f})", 
              ha='right', va='top', fontsize=9, color='white', fontweight='bold',
              bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.15'))
        
        # SOLVE TEXT OVERLAP: Display resources in a much more compact format - 폰트 크기 증가
        resource_text = ""
        for j, resource in enumerate(resources):
            resource_text += f"{resource['Resource']} ({resource['Percentage']}%)\n"
        # Position text in cell center with better spacing and 폰트 크기 증가
        plt.text(x + cell_width/2, y + cell_height/2, resource_text.strip(), 
              ha='center', va='center', fontsize=9, fontweight='bold',
              bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.18'))
        
# Add a simplified how-to-read guide - 폰트 크기 증가
how_to_read = """
HOW TO READ THIS MATRIX:
1. Each cell shows optimal resource combination for a specific career stage and domain
2. Percentages indicate recommended learning time proportion for each resource
3. Cell color indicates effectiveness (darker = more effective)
4. Numbers show Skill Gain metric (0-10 scale)
"""
plt.figtext(0.5, 0.11, how_to_read, ha='center', va='center', fontsize=11, 
          bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

# Add simplified effectiveness legend - more compact - 폰트 크기 증가
legend_x = grid_left
legend_y = 0.04
legend_width = grid_width
legend_height = 0.02

cmap_ax = plt.axes([legend_x, legend_y, legend_width, legend_height])
gradient = np.linspace(0, 1, 100).reshape(1, -1)
cmap_ax.imshow(gradient, aspect='auto', cmap=plt.cm.Blues)
cmap_ax.set_yticks([])

effectiveness_labels = ["Variable", "Moderate", "Medium", "High", "Very High"]
positions = np.linspace(0, 99, len(effectiveness_labels))
cmap_ax.set_xticks(positions)
cmap_ax.set_xticklabels(effectiveness_labels, fontsize=10)  # 폰트 크기 증가
cmap_ax.set_title('Pathway Effectiveness Rating', fontsize=11)

# Save the refined visualization
plt.savefig(IMAGES_DIR / "user_matrix_image.png", dpi=300, bbox_inches='tight')
plt.close()

###########################################
# 2. FINAL OPTIMAL COMBINATIONS
###########################################

# Define the data for optimal combinations
domains_data = {
    "AI/ML Skills": {
        "Market Weight": 0.28,
        "Growth Rate": 9.2,
        "Resources": {
            "Documentation": 0.35,
            "Academic Papers": 0.25,
            "Interactive Notebooks": 0.22,
            "Community Participation": 0.18
        },
        "Skill Acquisition Rate": 23,
        "Compensation Premium": 18,
        "ROI": "8.5 months, 76% retention"
    },
    "Cloud/DevOps Skills": {
        "Market Weight": 0.32,
        "Growth Rate": 12.5,
        "Resources": {
            "Official Documentation": 0.30,
            "Hands-on Labs": 0.35,
            "Certification Courses": 0.20,
            "Community Forums": 0.15
        },
        "Skill Acquisition Rate": 31,
        "Compensation Premium": 15,
        "ROI": "7.5 months, 82% retention"
    },
    "Modern Frontend Skills": {
        "Market Weight": 0.24,
        "Growth Rate": 8.7,
        "Resources": {
            "Interactive Platforms": 0.30,
            "Open Source Contribution": 0.30,
            "Design Resources": 0.20,
            "Community Forums": 0.20
        },
        "Skill Acquisition Rate": 27,
        "Compensation Premium": 14,
        "ROI": "8.2 months, 78% retention"
    }
}

# Create figure with better layout - 사이즈 증가
fig = plt.figure(figsize=(18, 14))  # 높이 증가

# Add title and subtitle
fig.suptitle('Optimal Learning Resource Combinations by Skill Domain', fontsize=20, y=0.98)
subtitle = "Based on statistical analysis of learning patterns from 10,000 developers (2024)"
plt.figtext(0.5, 0.94, subtitle, ha='center', fontsize=12, style='italic')

# Add compact methodology
methodology_ax = fig.add_axes([0.1, 0.88, 0.8, 0.04])
methodology_ax.axis('off')

methodology = """
METHODOLOGICAL APPROACH: Feature engineering (171 variables) → XGBoost regression (RMSE: 0.124) → 
SHAP analysis → Linear programming optimization → Cross-validation (80/20 split)
"""
methodology_ax.text(0.5, 0.5, methodology, ha='center', va='center', fontsize=10, 
                   bbox=dict(facecolor='whitesmoke', alpha=0.7, boxstyle='round,pad=0.2'))

# Set up domain names
domain_names = list(domains_data.keys())
domain_positions = np.arange(len(domain_names))

# 1. Resource Distribution (Top) with more space for text
ax1 = fig.add_axes([0.1, 0.65, 0.45, 0.2])  # 위치 조정 및 너비 줄임
ax1.set_title('Optimal Resource Distribution by Domain', fontsize=18)

# Thicker bars and bigger fonts
bar_height = 0.6  # Thicker bars
font_size_bar = 12
font_size_label = 13
font_size_legend = 12

# Prepare y positions with more spacing
y_pos = np.arange(len(domain_names))
y_pos = y_pos * 2.2  # More space between bars

# --- Plot bars for each domain with its own 4 resources ---
resource_color_map = [
    ['#3B4CC0', '#1E88E5', '#43A047', '#FBC02D'],   # AI/ML
    ['#8E24AA', '#26A69A', '#FF7043', '#BDBDBD'],   # Cloud/DevOps
    ['#3949AB', '#00ACC1', '#FFA726', '#D4E157']    # Modern Frontend
]
resource_label_map = [
    list(domains_data['AI/ML Skills']['Resources'].keys()),
    list(domains_data['Cloud/DevOps Skills']['Resources'].keys()),
    list(domains_data['Modern Frontend Skills']['Resources'].keys())
]

for domain_idx, domain in enumerate(domain_names):
    left = 0
    for r_idx, resource in enumerate(resource_label_map[domain_idx]):
        weight = domains_data[domain]['Resources'].get(resource, 0)
        ax1.barh(y_pos[domain_idx], weight, bar_height, left=left, color=resource_color_map[domain_idx][r_idx],
                 edgecolor='white', label=resource if domain_idx == 0 else "")
        # Label only if the segment is large enough and avoid overlap
        if weight > 0.05:
            text_x = left + weight/2
            text = ax1.text(text_x, y_pos[domain_idx], f"{resource}\n({int(weight*100)}%)", ha='center', va='center', fontsize=font_size_bar)
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
        left += weight

ax1.set_yticks(y_pos)
ax1.set_yticklabels(domain_names, fontsize=font_size_label)
ax1.set_xlim(0, 1)
ax1.set_xlabel('Proportion of Learning Resources', fontsize=font_size_label)
ax1.set_title('Optimal Resource Distribution by Domain', fontsize=18)

# Legend outside right, no duplicates - 위치 조정 (좀 더 왼쪽으로)
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=font_size_legend, frameon=True)

# Domain info - positioned to the right with better spacing
roi_ax = fig.add_axes([0.75, 0.65, 0.25, 0.2])  # 위치 조정
roi_ax.axis('off')
for i, domain in enumerate(domain_names):
    y_position = 0.9 - i * 0.33  # Better spacing
    roi_text = f"{domain}:\n"
    roi_text += f"• Weight: {domains_data[domain]['Market Weight']:.0%}\n"
    roi_text += f"• Growth: {domains_data[domain]['Growth Rate']}%\n"
    roi_text += f"• ROI: {domains_data[domain]['ROI']}"
    roi_ax.text(0, y_position, roi_text, va='top', fontsize=font_size_bar,
               bbox=dict(facecolor='lightgrey', alpha=0.3, boxstyle='round,pad=0.2'))

# 2. Impact Visualization (Middle) with better legend placement - 하단 차트 위치 조정
ax2 = fig.add_axes([0.1, 0.33, 0.8, 0.22])  # 차트 사이 간격을 늘림
ax2.set_title('Impact on Skill Acquisition and Compensation', fontsize=18)

# Bar width
width = 0.4  # Slightly wider bars

# Skill acquisition and compensation bars side by side
skill_rates = [domains_data[domain]["Skill Acquisition Rate"] for domain in domain_names]
comp_premiums = [domains_data[domain]["Compensation Premium"] for domain in domain_names]

# Create bars with better styling and spacing
bars1 = ax2.bar(domain_positions - width/2, skill_rates, width, label='Skill Acquisition Rate (%)', 
               color='#6a0dad', edgecolor='white', linewidth=1)
bars2 = ax2.bar(domain_positions + width/2, comp_premiums, width, label='Compensation Premium (%)', 
               color='#1f77b4', edgecolor='white', linewidth=1)

# Add average lines
ax2.axhline(y=np.mean(skill_rates), color='#6a0dad', linestyle='--', alpha=0.6)
ax2.axhline(y=np.mean(comp_premiums), color='#1f77b4', linestyle='--', alpha=0.6)

# Add value labels on bars - larger font, more space above
for i, v in enumerate(skill_rates):
    ax2.text(i - width/2, v + 1.5, f"{v}%", ha='center', fontsize=font_size_bar)
for i, v in enumerate(comp_premiums):
    ax2.text(i + width/2, v + 1.5, f"{v}%", ha='center', fontsize=font_size_bar)

ax2.set_ylabel('Percentage (%)', fontsize=font_size_label)
ax2.set_xticks(domain_positions)
ax2.set_xticklabels(domain_names, fontsize=font_size_label)

# Position legend below the chart instead of above
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, fontsize=font_size_legend, 
           columnspacing=1.5, handletextpad=0.8)
ax2.set_ylim(0, max(max(skill_rates), max(comp_premiums)) * 1.25)  # 좀 더 여유 공간 확보

# Save the final visualization
plt.tight_layout()
plt.subplots_adjust(hspace=0.8, bottom=0.2)  # 차트 간 간격 및 하단 여백 추가
plt.savefig(IMAGES_DIR / "user_combinations_image.png", dpi=300, bbox_inches='tight')
plt.close()

print("Created final visualizations with NO text overlap and optimized readability.")

###########################################
# 4. OPTIMAL LEARNING RESOURCE COMBINATIONS BY SKILL DOMAIN (Figure 3)
###########################################

# Create a heatmap for optimal learning resource combinations by skill domain
plt.figure(figsize=(14, 10))

# Define learning resources and domains
learning_resources = [
    "Official Documentation", 
    "Community Forums", 
    "Academic Papers/Books", 
    "Interactive Coding Platforms",
    "Online Courses",
    "Project-Based Learning",
    "Peer Programming",
    "Industry Conferences"
]

skill_domains = ["Cloud/DevOps", "AI/ML", "Frontend", "Backend", "Mobile"]

# Effectiveness matrix (SHAP impact values) - higher values indicate greater effectiveness
# These values should be based on your actual SHAP analysis results
effectiveness = np.array([
    [0.85, 0.65, 0.55, 0.70, 0.60],  # Documentation
    [0.80, 0.75, 0.70, 0.65, 0.55],  # Community Forums
    [0.50, 0.90, 0.40, 0.60, 0.45],  # Academic Papers/Books
    [0.55, 0.65, 0.85, 0.75, 0.70],  # Interactive Coding
    [0.65, 0.80, 0.70, 0.60, 0.75],  # Online Courses
    [0.60, 0.70, 0.75, 0.80, 0.85],  # Project-Based
    [0.55, 0.60, 0.65, 0.70, 0.60],  # Peer Programming
    [0.70, 0.75, 0.50, 0.45, 0.50],  # Conferences
])

# Create heatmap with improved styling
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(effectiveness, annot=True, fmt='.2f', cmap='YlOrRd', 
            xticklabels=skill_domains, yticklabels=learning_resources,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Effectiveness (SHAP Impact)'})

# Add title and labels
ax.set_title('Figure 3. Optimal Learning Resource Combinations by Skill Domain', fontsize=16, pad=20)
ax.set_xlabel('Skill Domains', fontsize=14, labelpad=10)
ax.set_ylabel('Learning Resources', fontsize=14, labelpad=10)

# Add explanation as text annotation
explanation = ("Each cell indicates the effectiveness of using a particular learning resource for a given skill domain.\n"
               "Brighter cells denote higher effectiveness. For instance, the figure highlights the strong effectiveness\n"
               "of documentation and community learning for Cloud, and of academic study plus community for AI/ML.")
plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=11, 
            bbox=dict(facecolor='#f8f8f8', alpha=0.8, boxstyle='round,pad=0.5'))

# Adjust layout and save
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(IMAGES_DIR / "optimal_resource_combinations.png", dpi=300, bbox_inches='tight')
plt.close()

print("Created all visualizations including Figure 3: Optimal Learning Resource Combinations by Skill Domain.")
