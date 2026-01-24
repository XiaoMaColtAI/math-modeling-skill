#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
附录表格分析工具

用于查看数学建模题目中"附录"文件夹内的表格文件的完整内容。
支持 .xlsx, .xls, .csv 格式文件。

使用方法：
    python analyze_appendix.py                    # 分析当前目录下的 附录/ 文件夹
    python analyze_appendix.py --path ./data      # 分析指定路径
    python analyze_appendix.py --file 附录1.xlsx  # 分析单个文件

作者：数学建模 Skill
版本：1.0.0
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

try:
    import pandas as pd
except ImportError:
    print("❌ 错误：未安装 pandas 库")
    print("请运行：pip install pandas openpyxl xlrd")
    sys.exit(1)


def find_appendix_files(base_path: str = None) -> List[str]:
    """
    查找附录文件夹中的所有表格文件

    参数：
        base_path: 基础路径，默认为当前工作目录

    返回：
        文件路径列表
    """
    if base_path is None:
        base_path = os.getcwd()

    # 尝试常见的附录文件夹名称
    appendix_names = ['附录', 'appendix', 'Appendix', '附件', 'data', 'Data']
    found_files = []

    for appendix_name in appendix_names:
        appendix_path = os.path.join(base_path, appendix_name)

        if os.path.exists(appendix_path) and os.path.isdir(appendix_path):
            print(f"✅ 找到附录文件夹: {appendix_path}\n")

            # 查找所有表格文件
            for filename in sorted(os.listdir(appendix_path)):
                if filename.endswith(('.xlsx', '.xls', '.csv')):
                    found_files.append(os.path.join(appendix_path, filename))

            if found_files:
                return found_files

    # 如果没找到，直接在当前目录查找
    print("⚠️  未找到'附录'文件夹，在当前目录查找表格文件...\n")
    for filename in sorted(os.listdir(base_path)):
        if filename.endswith(('.xlsx', '.xls', '.csv')):
            found_files.append(os.path.join(base_path, filename))

    return found_files


def print_dataframe(df: pd.DataFrame, max_cols: int = None):
    """
    完整打印 DataFrame 的所有内容

    参数：
        df: pandas DataFrame
        max_cols: 最大列数显示限制（None表示显示所有列）
    """
    # 设置 pandas 显示选项，显示所有行和列
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # 打印完整表格
    print(df.to_string())


def analyze_excel_file(filepath: str) -> Dict[str, Any]:
    """
    分析并完整输出 Excel 文件内容

    参数：
        filepath: Excel 文件路径

    返回：
        分析结果字典
    """
    result = {
        'filename': os.path.basename(filepath),
        'type': 'Excel',
        'sheets': {}
    }

    try:
        # 读取所有 sheet
        excel_file = pd.ExcelFile(filepath)
        sheet_names = excel_file.sheet_names

        for sheet_name in sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)

            result['sheets'][sheet_name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'data': df
            }

    except Exception as e:
        result['error'] = str(e)

    return result


def analyze_csv_file(filepath: str) -> Dict[str, Any]:
    """
    分析并完整输出 CSV 文件内容

    参数：
        filepath: CSV 文件路径

    返回：
        分析结果字典
    """
    result = {
        'filename': os.path.basename(filepath),
        'type': 'CSV',
    }

    try:
        # 尝试不同的编码
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                result['encoding'] = encoding
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise Exception("无法识别文件编码")

        result['sheets'] = {'Sheet1': {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'data': df
        }}

    except Exception as e:
        result['error'] = str(e)

    return result


def display_full_content(result: Dict[str, Any]):
    """
    完整显示表格内容

    参数：
        result: 分析结果字典
    """
    print("=" * 120)
    print(f"📄 文件名: {result['filename']}")
    print(f"📋 类型: {result['type']}")

    if 'error' in result:
        print(f"❌ 错误: {result['error']}")
        print("=" * 120)
        return

    if 'encoding' in result:
        print(f"🔤 编码: {result['encoding']}")

    print("")

    for sheet_name, sheet_data in result['sheets'].items():
        print("-" * 120)
        print(f"📊 工作表: {sheet_name}")
        print(f"📏 维度: {sheet_data['rows']} 行 × {sheet_data['columns']} 列")
        print("")

        # 列名信息
        print("📌 列名:")
        for i, col in enumerate(sheet_data['column_names'], 1):
            print(f"   {i:2d}. {col}")

        print("")
        print("📋 完整数据内容:")
        print("=" * 120)

        # 完整输出数据
        print_dataframe(sheet_data['data'])

        print("")
        print("=" * 120)
        print("")


def save_full_content(results: List[Dict[str, Any]], output_path: str):
    """
    保存完整内容到文件

    参数：
        results: 分析结果列表
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write("=" * 120 + "\n")
            f.write(f"📄 文件名: {result['filename']}\n")
            f.write(f"📋 类型: {result['type']}\n")

            if 'error' in result:
                f.write(f"❌ 错误: {result['error']}\n")
                f.write("=" * 120 + "\n\n")
                continue

            if 'encoding' in result:
                f.write(f"🔤 编码: {result['encoding']}\n")

            f.write("\n")

            for sheet_name, sheet_data in result['sheets'].items():
                f.write("-" * 120 + "\n")
                f.write(f"📊 工作表: {sheet_name}\n")
                f.write(f"📏 维度: {sheet_data['rows']} 行 × {sheet_data['columns']} 列\n")
                f.write("\n")

                # 列名信息
                f.write("📌 列名:\n")
                for i, col in enumerate(sheet_data['column_names'], 1):
                    f.write(f"   {i:2d}. {col}\n")

                f.write("\n")
                f.write("📋 完整数据内容:\n")
                f.write("=" * 120 + "\n")

                # 完整输出数据
                f.write(sheet_data['data'].to_string())

                f.write("\n")
                f.write("=" * 120 + "\n\n")

    print(f"✅ 完整内容已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='完整输出附录表格文件的所有内容',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python analyze_appendix.py
  python analyze_appendix.py --path ./data
  python analyze_appendix.py --file 附录1.xlsx
  python analyze_appendix.py --save full_content.txt
        """
    )

    parser.add_argument(
        '--path', '-p',
        type=str,
        default=None,
        help='指定要分析的路径（默认为当前目录）'
    )

    parser.add_argument(
        '--file', '-f',
        type=str,
        default=None,
        help='分析单个文件'
    )

    parser.add_argument(
        '--save', '-s',
        type=str,
        default=None,
        help='保存完整内容到文件'
    )

    args = parser.parse_args()

    # 分析单个文件
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ 错误：文件不存在 - {args.file}")
            sys.exit(1)

        print(f"\n🔍 分析文件: {args.file}\n")

        if args.file.endswith(('.xlsx', '.xls')):
            result = analyze_excel_file(args.file)
        elif args.file.endswith('.csv'):
            result = analyze_csv_file(args.file)
        else:
            print(f"❌ 错误：不支持的文件格式 - {args.file}")
            sys.exit(1)

        display_full_content(result)

        if args.save:
            save_full_content([result], args.save)

        sys.exit(0)

    # 查找并分析所有文件
    base_path = args.path if args.path else os.getcwd()

    if not os.path.exists(base_path):
        print(f"❌ 错误：路径不存在 - {base_path}")
        sys.exit(1)

    print(f"\n🔍 在路径中查找附录文件: {base_path}\n")

    files = find_appendix_files(base_path)

    if not files:
        print("❌ 未找到任何表格文件（.xlsx, .xls, .csv）")
        sys.exit(1)

    print(f"✅ 找到 {len(files)} 个文件\n")

    # 分析所有文件
    results = []
    for filepath in files:
        print(f"🔍 正在分析: {os.path.basename(filepath)}")

        if filepath.endswith(('.xlsx', '.xls')):
            result = analyze_excel_file(filepath)
        elif filepath.endswith('.csv'):
            result = analyze_csv_file(filepath)
        else:
            continue

        results.append(result)

    # 输出结果
    print("\n" + "=" * 120)
    print("📊 完整内容输出")
    print("=" * 120 + "\n")

    for result in results:
        display_full_content(result)

    # 保存报告
    if args.save:
        save_full_content(results, args.save)

    print("✅ 分析完成！")


if __name__ == '__main__':
    main()