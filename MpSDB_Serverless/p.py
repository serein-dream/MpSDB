import re
import subprocess
from docx import Document

# Step 1: 从 .docx 文档中提取包名
def extract_packages_from_docx(docx_path):
    import os
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"文件 {docx_path} 不存在")
    doc = Document(docx_path)
    packages = []
    package_pattern = r'\b[a-zA-Z0-9][a-zA-Z0-9-_]*\b'  # 更精确的正则表达式
    for para in doc.paragraphs:
        packages.extend(re.findall(package_pattern, para.text))
    return set(packages)
# Step 2: 获取当前 Conda 环境中的包及版本
def get_conda_packages():
    result = subprocess.run(['conda', 'list'], capture_output=True, text=True)
    conda_packages = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            package_name = parts[0]
            package_version = parts[1]
            conda_packages[package_name] = package_version
    return conda_packages

def get_pip_packages(conda_packages):
    result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
    pip_packages = conda_packages
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            package_name = parts[0]
            package_version = parts[1]
            pip_packages[package_name] = package_version
    return pip_packages

# Step 3: 匹配文档中的包名和 Conda 环境中的包
def match_packages(docx_path):
    docx_packages = extract_packages_from_docx(docx_path)

    # print(docx_packages)
    # print(f"\n\nso:\n")
    conda_packages = get_conda_packages()
    all_packages = get_pip_packages(conda_packages)
    # for i in all_packages:
    #     print(f"{i} == {all_packages[i]}")
    # raise
    # print(all_packages)
    matched_packages = {}
    
    for package in docx_packages:
        if package in all_packages:
            matched_packages[package] = all_packages[package]
    
    return matched_packages

# 示例运行
docx_path = 'ADD_by_yourself/FaaS_FL2023/ruanzhu24.12.2/code.docx'
matched = match_packages(docx_path)

# 输出匹配的包及版本
for package, version in matched.items():
    print(f'{package}: {version}')


