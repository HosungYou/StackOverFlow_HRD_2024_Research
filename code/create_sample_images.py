#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create sample images for the HRD report
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 디렉토리 설정
BASE_DIR = Path("/Volumes/External SSD/Pycharm/StackOverFlow_HRD_2024")
REPORTS_DIR = BASE_DIR / "Reports_0425"
IMAGES_DIR = REPORTS_DIR / "images"
FIGURES_DIR = REPORTS_DIR / "figures"

# 디렉토리가 없으면 생성
for dir_path in [IMAGES_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 샘플 SHAP 값 생성 (분류 모델)
def create_shap_classification_plot():
    # 플롯 스타일 설정
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    
    # 샘플 데이터 생성
    feature_names = ['교육 경험', '리더십 경험', '기술 역량', '팀워크 역량', 
                    '문제해결 능력', '의사소통 능력', '프로젝트 관리', '자격증 보유',
                    '산업 지식', '교육 수준', '경력 기간', '전문 분야 집중도']
    
    # 각 특성별 중요도 값
    importance = np.array([0.85, 0.75, 0.68, 0.63, 0.58, 0.52, 0.47, 0.43, 0.38, 0.32, 0.25, 0.18])
    
    # 각 특성별 영향 방향 (양수/음수)
    impact_direction = np.array([1, 1, 1, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, 0, -0.3, -0.5])
    
    # 플롯 생성
    colors = ['#ff4d4d' if x < 0 else '#4d94ff' for x in impact_direction]
    
    # 특성 중요도 기준으로 정렬
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    
    plt.barh(pos, importance[sorted_idx], align='center', color=[colors[i] for i in sorted_idx])
    plt.yticks(pos, [feature_names[i] for i in sorted_idx])
    plt.xlabel('SHAP Value (역량 가치 영향도)')
    plt.title('주요 HRD 역량의 경력 성장 영향력')
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "shap_summary_classification.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # figures 디렉토리에도 저장
    plt.figure(figsize=(10, 8))
    plt.barh(pos, importance[sorted_idx], align='center', color=[colors[i] for i in sorted_idx])
    plt.yticks(pos, [feature_names[i] for i in sorted_idx])
    plt.xlabel('SHAP Value (역량 가치 영향도)')
    plt.title('주요 HRD 역량의 경력 성장 영향력')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_classification_20250425.png", dpi=300, bbox_inches='tight')
    plt.close()

# 학습 경로 매트릭스 생성
def create_learning_pathway_matrix():
    plt.figure(figsize=(12, 10))
    
    # 샘플 데이터 생성
    pathways = ['기초 이론', '적용 실습', '분야 심화', '전략 통합', '리더십 개발']
    career_stages = ['초급 (0-3년)', '중급 (4-6년)', '고급 (7-10년)', '전문가 (10년+)']
    
    # 각 조합의 효과성 점수 (0-10)
    data = np.array([
        [9.5, 7.2, 4.1, 2.3], # 기초 이론
        [6.8, 9.3, 7.5, 5.2], # 적용 실습
        [3.2, 7.8, 9.4, 8.1], # 분야 심화
        [1.5, 4.6, 8.7, 9.6], # 전략 통합
        [2.3, 5.8, 8.2, 9.8]  # 리더십 개발
    ])
    
    # 히트맵 생성
    ax = plt.subplot(111)
    im = ax.imshow(data, cmap='YlGnBu')
    
    # 축 레이블 추가
    ax.set_xticks(np.arange(len(career_stages)))
    ax.set_yticks(np.arange(len(pathways)))
    ax.set_xticklabels(career_stages)
    ax.set_yticklabels(pathways)
    
    # 각 셀에 값 표시
    for i in range(len(pathways)):
        for j in range(len(career_stages)):
            text = ax.text(j, i, f"{data[i, j]:.1f}", 
                        ha="center", va="center", 
                        color="black" if data[i, j] > 7 else "white",
                        fontweight='bold')
    
    # 제목 및 컬러바
    plt.title('경력 단계별 학습 경로 효과성 매트릭스')
    plt.colorbar(im, label='효과성 점수 (0-10)')
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "user_matrix_image.png", dpi=300, bbox_inches='tight')
    plt.close()

# 투자 최적화 차트 생성
def create_investment_optimization_chart():
    plt.figure(figsize=(12, 8))
    
    # 샘플 데이터
    investment_types = ['형식적 교육', '업무 경험', '멘토링', '온라인 학습', '워크숍', '자격증']
    roi_values = [1.8, 2.3, 2.0, 1.5, 1.7, 1.3]
    effectiveness = [7.2, 9.1, 8.4, 6.8, 7.5, 6.2]
    implementation_cost = [8500, 5200, 4300, 3200, 6100, 7800]
    
    # 크기 정규화
    size_norm = [(x / max(implementation_cost)) * 1000 for x in implementation_cost]
    
    # 산점도 생성
    plt.scatter(roi_values, effectiveness, s=size_norm, alpha=0.7, 
                c=range(len(investment_types)), cmap='viridis', edgecolor='black')
    
    # 각 점에 레이블 추가
    for i, txt in enumerate(investment_types):
        plt.annotate(txt, (roi_values[i], effectiveness[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # 축 및 제목
    plt.xlabel('투자 수익률 (ROI)')
    plt.ylabel('효과성 점수 (0-10)')
    plt.title('HRD 투자 옵션 최적화 분석')
    
    # 참조선 추가
    plt.axhline(y=7.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=2.0, color='gray', linestyle='--', alpha=0.5)
    
    # 영역 표시
    plt.text(2.1, 7.6, '최적 투자 영역', fontsize=12, bbox=dict(facecolor='green', alpha=0.2))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "user_combinations_image.png", dpi=300, bbox_inches='tight')
    plt.close()

# 모든 이미지 생성 실행
if __name__ == "__main__":
    print("생성 중: SHAP 분류 시각화...")
    create_shap_classification_plot()
    
    print("생성 중: 학습 경로 매트릭스...")
    create_learning_pathway_matrix()
    
    print("생성 중: 투자 최적화 차트...")
    create_investment_optimization_chart()
    
    # SHAP 파일 경로 정보 저장
    with open(REPORTS_DIR / "Research" / "latest_shap_files.txt", "w") as f:
        f.write("classification:shap_summary_classification_20250425.png\n")
    
    print("모든 이미지가 성공적으로 생성되었습니다.")
