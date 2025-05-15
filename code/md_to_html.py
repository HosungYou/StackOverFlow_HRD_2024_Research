#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Markdown to HTML with Bootstrap styling
"""

import markdown
from pathlib import Path
import sys
import shutil
import re
import os

def convert_md_to_html(md_file, html_file=None):
    """
    Convert markdown file to HTML with Bootstrap styling
    
    Parameters:
    -----------
    md_file : str
        Path to markdown file
    html_file : str, optional
        Path to output HTML file. If None, will use the same name with .html extension
    """
    md_path = Path(md_file)
    
    if not html_file:
        html_file = md_path.with_suffix('.html')
    
    html_path = Path(html_file)
    
    # Read markdown file
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert to HTML
    html = markdown.markdown(
        md_content,
        extensions=['markdown.extensions.tables', 'markdown.extensions.toc', 'markdown.extensions.fenced_code']
    )
    
    # 최신 SHAP 파일 경로 확인
    latest_files = {}
    latest_files_path = Path(md_path).parent / "latest_shap_files.txt"
    if latest_files_path.exists():
        with open(latest_files_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    latest_files[key] = value
    
    # 이미지 디렉토리 확인 및 생성
    images_dir = html_path.parent / "images"
    if not images_dir.exists():
        images_dir.mkdir(parents=True)
    
    # 이미지 파일 복사 (figures 디렉토리에서 images 디렉토리로)
    if latest_files:
        # SHAP 파일 복사
        figures_dir = Path("/Volumes/External SSD/Pycharm/StackOverFlow_HRD_2024/Reports_0425/figures")  
        for model_type, filename in latest_files.items():
            source_file = figures_dir / filename
            target_file = images_dir / filename
            if source_file.exists() and not target_file.exists():
                shutil.copy2(source_file, target_file)
                print(f"Copied {source_file} to {target_file}")
    
    # Descriptive Analysis 이미지 파일 복사 (Research Reports/images 디렉토리에서 images 디렉토리로)
    descriptive_images_dir = Path("/Volumes/External SSD/Pycharm/StackOverFlow_HRD_2024/Reports_0425/Research Reports/images")
    if descriptive_images_dir.exists():
        for image_file in descriptive_images_dir.glob("*.png"):
            target_file = images_dir / image_file.name
            if not target_file.exists():
                shutil.copy2(image_file, target_file)
                print(f"Copied {image_file} to {target_file}")
                
        # CSV 파일도 복사
        for csv_file in descriptive_images_dir.glob("*.csv"):
            target_file = images_dir / csv_file.name
            if not target_file.exists():
                shutil.copy2(csv_file, target_file)
                print(f"Copied {csv_file} to {target_file}")
    
    # 마크다운에서 이미지 경로 수정 (절대 경로를 상대 경로로 변경)
    image_pattern = r'!\[(.*?)\]\((.*?)\)'
    def image_replacement(match):
        alt_text = match.group(1)
        image_path = match.group(2)
        
        # 절대 경로를 파일 이름만 사용하도록 변경
        if "Reports_0425/Research Reports/images" in image_path:
            image_name = os.path.basename(image_path)
            return f'![{alt_text}](images/{image_name})'
        return match.group(0)
        
    md_content = re.sub(image_pattern, image_replacement, md_content)
    
    # 수정된 마크다운으로 HTML 다시 생성
    html = markdown.markdown(
        md_content,
        extensions=['markdown.extensions.tables', 'markdown.extensions.toc', 'markdown.extensions.fenced_code']
    )
    
    # Add Bootstrap styling
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Developer Skills Market Value Analysis - 2024</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #0056b3;
            margin-top: 1.5em;
            margin-bottom: 0.75em;
        }
        h1 {
            font-size: 2.5rem;
            border-bottom: 2px solid #0056b3;
            padding-bottom: 10px;
        }
        h2 {
            font-size: 2rem;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        h3 {
            font-size: 1.5rem;
        }
        img {
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        /* 특별한 CSS 규칙: SHAP 시각화를 위한 설정 */
        img[src*="shap_summary"] {
            height: auto !important; 
            width: 100% !important;
            max-width: 1000px;
            object-fit: contain;
            object-position: top;
        }
        table {
            width: 100%;
            margin: 20px 0;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px 15px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        code {
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: 'Courier New', Courier, monospace;
        }
        pre {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', Courier, monospace;
        }
        blockquote {
            border-left: 4px solid #0056b3;
            padding-left: 15px;
            margin-left: 0;
            color: #666;
        }
        a {
            color: #0056b3;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .toc {
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }
        @media (max-width: 768px) {
            body {
                padding: 15px;
            }
            h1 {
                font-size: 2rem;
            }
            h2 {
                font-size: 1.75rem;
            }
            h3 {
                font-size: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <div class="col-12">
                {html_content}
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    
    # Write HTML file
    final_html = html_template.replace('{html_content}', html)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    print(f"Successfully converted {md_path} to {html_path}")
    return html_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md_to_html.py [markdown_file] [html_file (optional)]")
        sys.exit(1)
    
    md_file = sys.argv[1]
    html_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_md_to_html(md_file, html_file)
