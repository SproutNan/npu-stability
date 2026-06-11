#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Profiling批量分析工具 - 完整版
支持批量处理、生成两个算子对比表和低精度倍率表
"""

import sys
import pandas as pd
from pathlib import Path
import subprocess
import argparse


class ProfilingBatchAnalyzer:
    """Profiling批量分析器"""
    
    def __init__(self, a3_pros_path, a5_pros_path, mstt_path, output_path, mode='both'):
        # 确保所有路径都是绝对路径
        self.a3_pros_path = Path(a3_pros_path).resolve()
        self.a5_pros_path = Path(a5_pros_path).resolve()
        self.mstt_path = Path(mstt_path).resolve()
        self.output_path = Path(output_path).resolve()
        self.compare_tools_path = self.mstt_path / "profiler" / "msprof_analyze" / "compare_tools"
        self.mode = mode  # operator_only, ratio_only, both
    
    def infer_model_name_for_flat_layout(self, profiles_path):
        """为平铺场景目录推断模型名"""
        if profiles_path.name in ["A3", "A5"] and profiles_path.parent.name == "profiles":
            return profiles_path.parent.parent.name
        if profiles_path.parent.name == "profiles":
            return profiles_path.parent.parent.name
        return profiles_path.name

    def discover_model_scenarios(self, profiles_path, pretrain_type):
        """从目录中发现模型和场景，兼容两种布局"""
        discovered = {}
        if not profiles_path.exists():
            print(f"Error: path does not exist: {profiles_path}")
            return discovered

        direct_scenarios = {}
        for item in profiles_path.iterdir():
            if not item.is_dir():
                continue

            nested_pretrain_dir = item / "profiles" / pretrain_type
            if nested_pretrain_dir.exists():
                nested_scenarios = {}
                for scenario_dir in nested_pretrain_dir.iterdir():
                    if scenario_dir.is_dir():
                        nested_scenarios[scenario_dir.name] = scenario_dir
                if nested_scenarios:
                    discovered[item.name] = nested_scenarios
                continue

            if self.find_ascend_pt_dir(item):
                direct_scenarios[item.name] = item

        if discovered:
            return discovered

        if direct_scenarios:
            model_name = self.infer_model_name_for_flat_layout(profiles_path)
            discovered[model_name] = direct_scenarios

        return discovered
    
    def find_ascend_pt_dir(self, scenario_path):
        """查找_ascend_pt目录"""
        for item in scenario_path.rglob("*"):
            if item.is_dir() and "_ascend_pt" in item.name:
                return item
        return None
    
    def find_op_statistic_files(self, scenario_path):
        """查找op_statistic.csv文件"""
        files = {}

        for item in scenario_path.rglob("*.csv"):
            if "op_statistic" in item.name:
                # 判断是bf16还是fp8 - 在整个路径中查找
                path_str = str(item).lower()
                if "bf16" in path_str:
                    # 确保返回绝对路径
                    files["bf16"] = item.resolve()
                    print(f"Found BF16 op_statistic.csv: {item.resolve()}")
                elif "fp8" in path_str:
                    # 确保返回绝对路径
                    files["fp8"] = item.resolve()
                    print(f"Found FP8 op_statistic.csv: {item.resolve()}")

        return files
    
    def get_quant_time(self, fp8_op_stat_path):
        """从FP8的op_statistic.csv中获取quant耗时"""
        try:
            df = pd.read_csv(fp8_op_stat_path)

            # 计算三个算子的总和
            quant_time = 0

            # DynamicMxQuantWithDualAxis
            dual_axis_mask = df['OP Type'].astype(str).str.contains('DynamicMxQuantWithDualAxis', case=False, na=False)
            dual_axis_rows = df[dual_axis_mask]
            if len(dual_axis_rows) > 0:
                quant_time += float(dual_axis_rows.iloc[0]['Total Time(us)'])

            # DynamicMxQuant
            dynamic_mask = df['OP Type'].astype(str).str.contains('DynamicMxQuant', case=False, na=False)
            dynamic_rows = df[dynamic_mask]
            if len(dynamic_rows) > 0:
                quant_time += float(dynamic_rows.iloc[0]['Total Time(us)'])

            # GroupedDynamicMxQuant
            grouped_mask = df['OP Type'].astype(str).str.contains('GroupedDynamicMxQuant', case=False, na=False)
            grouped_rows = df[grouped_mask]
            if len(grouped_rows) > 0:
                quant_time += float(grouped_rows.iloc[0]['Total Time(us)'])

            print(f"Quant time (DynamicMxQuantWithDualAxis + DynamicMxQuant + GroupedDynamicMxQuant): {quant_time} us")
            return quant_time
        except Exception as e:
            print(f"Warning: Error getting quant time: {e}")
            return 0

    def create_op_type_comparison(self, a3_op_stat_path, a5_op_stat_path, model_name, scenario_name, output_path):
        """创建计算算子对比表（基于OP Type）"""
        print(f"\nGenerating OP Type comparison table...")

        try:
            # 读取A3和A5的op_statistic.csv
            a3_df = pd.read_csv(a3_op_stat_path)
            a5_df = pd.read_csv(a5_op_stat_path)

            # 按OP Type分组，合并相同算子的耗时
            a3_grouped = a3_df.groupby('OP Type').agg({
                'Core Type': 'first',
                'Count': 'sum',
                'Total Time(us)': 'sum',
                'Ratio(%)': 'sum'
            }).reset_index()

            a5_grouped = a5_df.groupby('OP Type').agg({
                'Core Type': 'first',
                'Count': 'sum',
                'Total Time(us)': 'sum',
                'Ratio(%)': 'sum'
            }).reset_index()

            # 创建对比数据
            comparison_data = []

            # 获取所有OP Type（A3和A5的并集）
            all_op_types = set(a3_grouped['OP Type']) | set(a5_grouped['OP Type'])

            for op_type in sorted(all_op_types):
                # A3数据
                a3_row = a3_grouped[a3_grouped['OP Type'] == op_type]
                a3_time = float(a3_row['Total Time(us)'].iloc[0]) if len(a3_row) > 0 else 0
                a3_count = int(a3_row['Count'].iloc[0]) if len(a3_row) > 0 else 0
                a3_core = str(a3_row['Core Type'].iloc[0]) if len(a3_row) > 0 else ''
                a3_ratio = float(a3_row['Ratio(%)'].iloc[0]) if len(a3_row) > 0 else 0

                # A5数据
                a5_row = a5_grouped[a5_grouped['OP Type'] == op_type]
                a5_time = float(a5_row['Total Time(us)'].iloc[0]) if len(a5_row) > 0 else 0
                a5_count = int(a5_row['Count'].iloc[0]) if len(a5_row) > 0 else 0
                a5_core = str(a5_row['Core Type'].iloc[0]) if len(a5_row) > 0 else ''
                a5_ratio = float(a5_row['Ratio(%)'].iloc[0]) if len(a5_row) > 0 else 0

                # 计算A3/A5比值
                if a3_time > 0 and a5_time > 0:
                    a3_a5_ratio = a3_time / a5_time
                elif a3_time > 0:
                    a3_a5_ratio = float('inf')
                elif a5_time > 0:
                    a3_a5_ratio = 0
                else:
                    a3_a5_ratio = 0

                comparison_data.append({
                    'OP_Type': op_type,
                    'A3_Core_Type': a3_core,
                    'A3_Count': a3_count,
                    'A3_Total_Time_us': a3_time,
                    'A3_Ratio_percent': a3_ratio,
                    'A5_Core_Type': a5_core,
                    'A5_Count': a5_count,
                    'A5_Total_Time_us': a5_time,
                    'A5_Ratio_percent': a5_ratio,
                    'A3_A5_Ratio': a3_a5_ratio
                })

            # 按A3 Total Time降序排序
            comparison_data.sort(key=lambda x: x['A3_Total_Time_us'], reverse=True)

            # 保存为CSV
            csv_file = output_path / f"{model_name}_{scenario_name}_op_type_comparison.csv"
            df = pd.DataFrame(comparison_data)
            df.to_csv(csv_file, index=False, encoding='utf-8')

            print(f"OP Type comparison table generated: {csv_file}")
            return str(csv_file)

        except Exception as e:
            print(f"Error generating OP Type comparison: {e}")
            return None


    def run_mstt_comparison(self, a3_path, a5_path, output_dir):
        """运行mstt对比"""
        # 确保所有路径都是绝对路径
        a3_path = Path(a3_path).resolve()
        a5_path = Path(a5_path).resolve()
        output_dir = Path(output_dir).resolve()

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(self.compare_tools_path / "performance_compare.py"),
            str(a3_path),
            str(a5_path),
            "-o",
            str(output_dir)
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.compare_tools_path),
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                print(f"Command execution failed: {result.stderr}")
                return None
            
            # 查找生成的Excel文件
            excel_files = list(output_dir.rglob("*.xlsx"))
            if not excel_files:
                print("警告: 未找到Excel文件")
                return None
            
            return str(excel_files[0])
        except Exception as e:
            print(f"Execution exception: {e}")
            return None
    
    def extract_overall_metrics(self, excel_path):
        """从OverallMetrics提取数据"""
        df = pd.read_excel(excel_path, sheet_name='OverallMetrics', header=2)
        
        index_col = 'Index'
        duration_cols = [col for col in df.columns if 'Duration(ms)' in str(col)]
        
        data = {}
        
        def get_value(row_name, col_index=0):
            try:
                mask = df[index_col].astype(str).str.contains(row_name, case=False, na=False, regex=True)
                matched_rows = df[mask]
                
                if len(matched_rows) > 0:
                    if len(duration_cols) > col_index:
                        value = matched_rows.iloc[0][duration_cols[col_index]]
                        return float(value) if pd.notna(value) else 0
                return 0
            except:
                return 0
        
        # 提取各项数据
        data['A3_Computing_Time'] = get_value('Computing Time', 0)
        data['A5_Computing_Time'] = get_value('Computing Time', 1)
        
        fa_forward_a3 = get_value('Flash Attention.*Forward', 0)
        fa_backward_a3 = get_value('Flash Attention.*Backward', 0)
        data['A3_FA_Time'] = fa_forward_a3 + fa_backward_a3
        
        fa_forward_a5 = get_value('Flash Attention.*Forward', 1)
        fa_backward_a5 = get_value('Flash Attention.*Backward', 1)
        data['A5_FA_Time'] = fa_forward_a5 + fa_backward_a5
        
        data['A3_Cube_Time'] = get_value('Matmul', 0)
        data['A5_Cube_Time'] = get_value('Matmul', 1)
        
        # Vector精确匹配
        vector_mask = df[index_col].astype(str) == '\tVector'
        if vector_mask.any():
            data['A3_Vector_Time'] = df[vector_mask].iloc[0][duration_cols[0]]
            data['A5_Vector_Time'] = df[vector_mask].iloc[0][duration_cols[1]]
        else:
            data['A3_Vector_Time'] = 0
            data['A5_Vector_Time'] = 0
        
        data['A3_SDMA_Time'] = get_value('SDMA.*Tensor Move', 0)
        data['A5_SDMA_Time'] = get_value('SDMA.*Tensor Move', 1)
        
        data['A3_Others_Time'] = get_value('Others', 0)
        data['A5_Others_Time'] = get_value('Others', 1)
        
        data['A3_Uncovered_Comm_Time'] = get_value('Uncovered Communication Time', 0)
        data['A5_Uncovered_Comm_Time'] = get_value('Uncovered Communication Time', 1)
        
        data['A3_Free_Time'] = get_value('Free Time', 0)
        data['A5_Free_Time'] = get_value('Free Time', 1)
        
        data['A3_E2E_Time'] = get_value('E2E Time', 0)
        data['A5_E2E_Time'] = get_value('E2E Time', 1)
        
        return data
    
    def extract_communication_compare(self, excel_path):
        """从CommunicationCompare提取通信算子数据"""
        try:
            df = pd.read_excel(excel_path, sheet_name='CommunicationCompare', header=None, skiprows=2)
            df.columns = ['Order Id', 'Communication OP Name', 'Task Name', 'Calls', 'Total Duration(us)',
                          'Avg Duration(us)', 'Max Duration(us)', 'Min Duration(us)',
                          'Communication OP Name.1', 'Task Name.1', 'Calls.1', 'Total Duration(us).1',
                          'Avg Duration(us).1', 'Max Duration(us).1', 'Min Duration(us).1',
                          'Diff Duration(us)', 'Diff Ratio']
        except:
            return {}
        
        op_name_col = 'Communication OP Name'
        duration_cols = ['Total Duration(us)', 'Total Duration(us).1']
        
        data = {}
        
        def get_comm_value(op_name, col_index=0):
            try:
                mask = df[op_name_col].astype(str).str.contains(op_name, case=False, na=False)
                matched_rows = df[mask]
                if len(matched_rows) > 0:
                    return float(matched_rows.iloc[0][duration_cols[col_index]]) if pd.notna(matched_rows.iloc[0][duration_cols[col_index]]) else 0
                return 0
            except:
                return 0
        
        data['A3_allgather_Time'] = get_comm_value('allgather', 0)
        data['A5_allgather_Time'] = get_comm_value('allgather', 1)
        data['A3_allreduce_Time'] = get_comm_value('allreduce', 0)
        data['A5_allreduce_Time'] = get_comm_value('allreduce', 1)
        data['A3_reducescatter_Time'] = get_comm_value('reducescatter', 0)
        data['A5_reducescatter_Time'] = get_comm_value('reducescatter', 1)
        data['A3_alltoallv_Time'] = get_comm_value('alltoallv', 0)
        data['A5_alltoallv_Time'] = get_comm_value('alltoallv', 1)
        
        return data
    
    def create_operator_comparison(self, overall_data, comm_data, model_name, scenario_name, output_path, fp8_op_stat_path=None):
        """创建算子对比表"""
        all_data = {**overall_data, **comm_data}

        # 如果是FP8场景，需要特殊处理Cube Time和Vector Time
        quant_time = 0
        if fp8_op_stat_path and 'FP8' in scenario_name:
            print(f"\nFP8 scenario detected, extracting quant time...")
            quant_time = self.get_quant_time(fp8_op_stat_path)

            # 修改A5的Cube Time和Vector Time
            original_cube_time = all_data.get('A5_Cube_Time', 0)
            original_vector_time = all_data.get('A5_Vector_Time', 0)

            # Cube Time = 原Cube Time + quant耗时
            all_data['A5_Cube_Time'] = original_cube_time + quant_time
            print(f"A5 Cube Time adjusted: {original_cube_time} + {quant_time} = {all_data['A5_Cube_Time']}")

            # Vector Time = 原Vector Time - quant耗时
            all_data['A5_Vector_Time'] = max(0, original_vector_time - quant_time)
            print(f"A5 Vector Time adjusted: {original_vector_time} - {quant_time} = {all_data['A5_Vector_Time']}")
        
        comparison_data = [
            ('Computing Time', 'A3_Computing_Time', 'A5_Computing_Time'),
            ('FA Time', 'A3_FA_Time', 'A5_FA_Time'),
            ('Cube Time', 'A3_Cube_Time', 'A5_Cube_Time'),
            ('Vector Time', 'A3_Vector_Time', 'A5_Vector_Time'),
            ('SDMA Time', 'A3_SDMA_Time', 'A5_SDMA_Time'),
            ('Others Time', 'A3_Others_Time', 'A5_Others_Time'),
            ('Uncovered Communication Time', 'A3_Uncovered_Comm_Time', 'A5_Uncovered_Comm_Time'),
            ('FREE Time', 'A3_Free_Time', 'A5_Free_Time'),
            ('E2E Time', 'A3_E2E_Time', 'A5_E2E_Time'),
            ('allgather Time', 'A3_allgather_Time', 'A5_allgather_Time'),
            ('allreduce Time', 'A3_allreduce_Time', 'A5_allreduce_Time'),
            ('reducescatter Time', 'A3_reducescatter_Time', 'A5_reducescatter_Time'),
            ('alltoallv Time', 'A3_alltoallv_Time', 'A5_alltoallv_Time'),
        ]
        
        # 计算通信算子总耗时（用于计算占比）
        comm_operators = ['allgather Time', 'allreduce Time', 'reducescatter Time', 'alltoallv Time']
        a3_comm_total = sum([all_data.get(f'A3_{op.replace(" Time", "_Time")}', 0) for op in comm_operators])
        a5_comm_total = sum([all_data.get(f'A5_{op.replace(" Time", "_Time")}', 0) for op in comm_operators])

        result_data = []
        for item_name, a3_key, a5_key in comparison_data:
            a3_value = all_data.get(a3_key, 0)
            a5_value = all_data.get(a5_key, 0)
            ratio = a3_value / a5_value if a5_value > 0 else 0

            # 计算通信算子占比
            a3_ratio_percent = 0
            a5_ratio_percent = 0
            if item_name in comm_operators:
                if a3_comm_total > 0:
                    a3_ratio_percent = (a3_value / a3_comm_total) * 100
                if a5_comm_total > 0:
                    a5_ratio_percent = (a5_value / a5_comm_total) * 100

            result_data.append({
                'Item': item_name,
                'A3_Time': a3_value,
                'A5_Time': a5_value,
                'A3/A5_Ratio': ratio,
                'A3_Ratio_percent': a3_ratio_percent,
                'A5_Ratio_percent': a5_ratio_percent
            })
        
        df = pd.DataFrame(result_data)
        csv_file = output_path / f"{model_name}_{scenario_name}_operator_comparison.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"Operator comparison table generated: {csv_file}")
        return str(csv_file)
    
    def calculate_low_precision_ratio(self, bf16_csv, fp8_csv, model_name, scenario_name, output_path):
        """计算低精度优化倍率"""
        bf16_df = pd.read_csv(bf16_csv)
        fp8_df = pd.read_csv(fp8_csv)
        
        def get_op_time(df, op_names):
            total_time = 0
            for op_name in op_names:
                matched = df[df['OP Type'] == op_name]
                if len(matched) > 0:
                    total_time += matched['Total Time(us)'].sum()
            return total_time
        
        # matmul
        matmul_bf16_time = get_op_time(bf16_df, ['MatMulV3', 'GemmV3'])
        matmul_bf16_unreplaced_time = get_op_time(fp8_df, ['MatMulV3', 'GemmV3'])
        matmul_replaced_time = matmul_bf16_time - matmul_bf16_unreplaced_time
        matmul_quant_time = get_op_time(fp8_df, ['QuantBatchMatmulV3'])
        matmul_ratio = matmul_replaced_time / matmul_quant_time if matmul_quant_time > 0 else 0
        
        # gemm
        gemm_bf16_time = get_op_time(bf16_df, ['GroupedMatmul'])
        gemm_replaced_time = gemm_bf16_time
        gemm_quant_time = get_op_time(fp8_df, ['GroupedMatmul'])
        gemm_ratio = gemm_replaced_time / gemm_quant_time if gemm_quant_time > 0 else 0
        
        # quant
        quant_time = get_op_time(fp8_df, ['DynamicMxQuant', 'GroupedDynamicMxQuant'])
        
        # 总体
        high_precision_time = matmul_replaced_time + gemm_replaced_time
        low_precision_time = matmul_quant_time + gemm_quant_time + quant_time
        final_ratio = high_precision_time / low_precision_time if low_precision_time > 0 else 0
        
        ratio_data = [
            {'Item': 'matmul_bf16_time', 'Value(us)': matmul_bf16_time},
            {'Item': 'matmul_bf16_unreplaced_time', 'Value(us)': matmul_bf16_unreplaced_time},
            {'Item': 'matmul_replaced_time', 'Value(us)': matmul_replaced_time},
            {'Item': 'matmul_quant_time', 'Value(us)': matmul_quant_time},
            {'Item': 'matmul_ratio', 'Value': matmul_ratio},
            {'Item': '', 'Value': ''},
            {'Item': 'gemm_bf16_time', 'Value(us)': gemm_bf16_time},
            {'Item': 'gemm_replaced_time', 'Value(us)': gemm_replaced_time},
            {'Item': 'gemm_quant_time', 'Value(us)': gemm_quant_time},
            {'Item': 'gemm_ratio', 'Value': gemm_ratio},
            {'Item': '', 'Value': ''},
            {'Item': 'quant_time', 'Value(us)': quant_time},
            {'Item': '', 'Value': ''},
            {'Item': 'high_precision_time', 'Value(us)': high_precision_time},
            {'Item': 'low_precision_time', 'Value(us)': low_precision_time},
            {'Item': 'final_ratio', 'Value': final_ratio},
        ]
        
        df = pd.DataFrame(ratio_data)
        csv_file = output_path / f"{model_name}_{scenario_name}_low_precision_ratio.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"Low precision ratio table generated: {csv_file}")
        return str(csv_file)
    
    def process_scenario(self, model_name, a3_scenario_name, a5_scenario_bf16_name, a5_scenario_fp8_name,
                        a3_ascend_pt, a5_bf16_ascend_pt, a5_fp8_ascend_pt,
                        a5_bf16_path, a5_fp8_path):
        """处理单个场景"""
        print(f"\n{'='*80}")
        print(f"处理场景: {model_name} - {a3_scenario_name}")
        print(f"运行模式: {self.mode}")
        print(f"{'='*80}")
        print(f"A5 BF16路径: {a5_bf16_path}")
        print(f"A5 FP8路径: {a5_fp8_path}")

        # 初始化变量
        operator_csv_bf16 = None
        operator_csv_fp8 = None
        ratio_csv = None
        op_type_csv_bf16 = None
        op_type_csv_fp8 = None
        
        # 提取场景前缀和后缀
        scenario_parts = a3_scenario_name.split('_')
        prefix = scenario_parts[0] if scenario_parts else ''
        suffix = scenario_parts[2] if len(scenario_parts) > 2 else ''
        
        # 创建输出目录
        scenario_output = self.output_path / model_name
        scenario_output.mkdir(parents=True, exist_ok=True)
        
        # 1. 运行A3 BF16 vs A5 BF16对比
        print(f"\n步骤1: 运行A3 BF16 vs A5 BF16对比")
        excel_bf16 = self.run_mstt_comparison(
            a3_ascend_pt, a5_bf16_ascend_pt,
            scenario_output / f"{a3_scenario_name}_vs_{a5_scenario_bf16_name}_mstt"
        )
        
        if not excel_bf16:
            print("A3 vs A5 BF16对比失败")
            return
        
        # 生成第一个算子对比表
        print(f"\n步骤2: 生成算子对比表 (A3 BF16 vs A5 BF16)")
        overall_data_bf16 = self.extract_overall_metrics(excel_bf16)
        comm_data_bf16 = self.extract_communication_compare(excel_bf16)
        
        operator_csv_bf16 = self.create_operator_comparison(
            overall_data_bf16, comm_data_bf16,
            model_name, f"A3_BF16_vs_A5_BF16_{prefix}_{suffix}",
            scenario_output
        )
        
        # 2. 运行A3 BF16 vs A5 FP8对比
        print(f"\n步骤3: 运行A3 BF16 vs A5 FP8对比")
        excel_fp8 = self.run_mstt_comparison(
            a3_ascend_pt, a5_fp8_ascend_pt,
            scenario_output / f"{a3_scenario_name}_vs_{a5_scenario_fp8_name}_mstt"
        )
        
        operator_csv_fp8 = None
        if excel_fp8:
            # 生成第二个算子对比表
            print(f"\n步骤4: 生成算子对比表 (A3 BF16 vs A5 FP8)")
            overall_data_fp8 = self.extract_overall_metrics(excel_fp8)
            comm_data_fp8 = self.extract_communication_compare(excel_fp8)
            
            operator_csv_fp8 = self.create_operator_comparison(
                overall_data_fp8, comm_data_fp8,
                model_name, f"A3_BF16_vs_A5_FP8_{prefix}_{suffix}",
                scenario_output
            )
        else:
            print("A3 vs A5 FP8对比失败")
        
        if self.mode in ['ratio_only', 'both']:
            # 3. 生成低精度倍率表
            print(f"\n步骤5: 生成低精度倍率表 (A5 BF16 vs A5 FP8)")
        a5_bf16_op_stat = self.find_op_statistic_files(a5_bf16_path)
        a5_fp8_op_stat = self.find_op_statistic_files(a5_fp8_path)
        
        ratio_csv = None
        if "bf16" in a5_bf16_op_stat and "fp8" in a5_fp8_op_stat:
            ratio_csv = self.calculate_low_precision_ratio(
                str(a5_bf16_op_stat["bf16"]),
                str(a5_fp8_op_stat["fp8"]),
                model_name, f"A5_BF16_vs_A5_FP8_{prefix}_{suffix}",
                scenario_output
            )
        else:
            print("警告: 未找到op_statistic.csv文件")

        # 4. 生成计算算子对比表
        if self.mode in ['operator_only', 'both']:
            print(f"\n步骤6: 生成计算算子对比表")
            # 查找A3的op_statistic.csv
            a3_op_stat = self.find_op_statistic_files(a3_ascend_pt.parent)

            # 生成A3 BF16 vs A5 BF16的计算算子对比表
            if "bf16" in a3_op_stat and "bf16" in a5_bf16_op_stat:
                op_type_csv_bf16 = self.create_op_type_comparison(
                    str(a3_op_stat["bf16"]),
                    str(a5_bf16_op_stat["bf16"]),
                    model_name, f"A3_vs_A5_BF16_{prefix}_{suffix}",
                    scenario_output
                )
            else:
                print("Warning: A3 or A5 BF16 op_statistic.csv not found for OP Type comparison")

            # 生成A3 BF16 vs A5 FP8的计算算子对比表
            if "bf16" in a3_op_stat and "fp8" in a5_fp8_op_stat:
                op_type_csv_fp8 = self.create_op_type_comparison(
                    str(a3_op_stat["bf16"]),
                    str(a5_fp8_op_stat["fp8"]),
                    model_name, f"A3_vs_A5_FP8_{prefix}_{suffix}",
                    scenario_output
                )
            else:
                print("Warning: A3 or A5 FP8 op_statistic.csv not found for OP Type comparison")

        print(f"\n场景处理完成！")
        return {
            'model': model_name,
            'scenario': a3_scenario_name,
            'operator_csv_bf16': operator_csv_bf16,
            'operator_csv_fp8': operator_csv_fp8,
            'ratio_csv': ratio_csv,
            'op_type_csv_bf16': op_type_csv_bf16,
            'op_type_csv_fp8': op_type_csv_fp8
        }
    
    def run(self):
        """运行批量分析"""
        print("=" * 80)
        print("Profiling批量分析工具")
        print("=" * 80)
        
        a3_models = self.discover_model_scenarios(self.a3_pros_path, "A3")
        a5_models = self.discover_model_scenarios(self.a5_pros_path, "A5")

        if not a3_models:
            print(f"未找到A3场景: {self.a3_pros_path}")
            return
        if not a5_models:
            print(f"未找到A5场景: {self.a5_pros_path}")
            return

        common_models = sorted(set(a3_models.keys()) & set(a5_models.keys()))
        if not common_models and len(a3_models) == 1 and len(a5_models) == 1:
            common_models = [next(iter(a3_models.keys()))]
            a5_models[common_models[0]] = next(iter(a5_models.values()))

        print(f"\nA3模型: {sorted(a3_models.keys())}")
        print(f"A5模型: {sorted(a5_models.keys())}")
        print(f"匹配模型: {common_models}")

        if not common_models:
            print("未找到可匹配的模型")
            return
        
        # 处理每个模型
        all_results = {}
        for model_name in common_models:
            print(f"\n{'='*80}")
            print(f"处理模型: {model_name}")
            print(f"{'='*80}")
            
            a3_scenarios = a3_models.get(model_name, {})
            a5_scenarios = a5_models.get(model_name, {})
            
            print(f"A3场景: {list(a3_scenarios.keys())}")
            print(f"A5场景: {list(a5_scenarios.keys())}")
            
            # 匹配场景
            results = []
            for a3_name, a3_path in a3_scenarios.items():
                # 提取前缀和后缀
                a3_parts = a3_name.split('_')
                a3_prefix = a3_parts[0] if a3_parts else ''
                a3_suffix = a3_parts[2] if len(a3_parts) > 2 else ''
                
                # 查找对应的A5 BF16场景
                for a5_name, a5_path in a5_scenarios.items():
                    if 'bf16' in a5_name.lower():
                        a5_parts = a5_name.split('_')
                        a5_prefix = a5_parts[0] if a5_parts else ''
                        a5_suffix = a5_parts[2] if len(a5_parts) > 2 else ''
                        
                        # 基于前缀和后缀匹配
                        if a3_prefix == a5_prefix and a3_suffix == a5_suffix:
                            # 查找对应的A5 FP8场景
                            a5_fp8_name = a5_name.replace('_bf16', '_fp8')
                            if a5_fp8_name in a5_scenarios:
                                a5_fp8_path = a5_scenarios[a5_fp8_name]
                                
                                # 查找ascend_pt目录
                                a3_ascend = self.find_ascend_pt_dir(a3_path)
                                a5_bf16_ascend = self.find_ascend_pt_dir(a5_path)
                                a5_fp8_ascend = self.find_ascend_pt_dir(a5_fp8_path)
                                
                                if a3_ascend and a5_bf16_ascend and a5_fp8_ascend:
                                    result = self.process_scenario(
                                        model_name, a3_name, a5_name, a5_fp8_name,
                                        a3_ascend, a5_bf16_ascend, a5_fp8_ascend,
                                        a5_path, a5_fp8_path
                                    )
                                    if result:
                                        results.append(result)
            
            all_results[model_name] = results
        
        # 生成汇总报告
        self.generate_summary_report(all_results)
        
        print("\n" + "=" * 80)
        print("批量分析完成！")
        print("=" * 80)
        print(f"\n输出目录: {self.output_path}")
    
    def generate_summary_report(self, all_results):
        """生成汇总报告"""
        print("\n生成汇总报告...")
        
        summary_data = []
        for model_name, results in all_results.items():
            for result in results:
                summary_data.append({
                    'Model': model_name,
                    'Scenario': result['scenario'],
                    'Operator_Comparison_A3_vs_A5_BF16': result.get('operator_csv_bf16', ''),
                    'Operator_Comparison_A3_vs_A5_FP8': result.get('operator_csv_fp8', ''),
                    'Low_Precision_Ratio': result.get('ratio_csv', ''),
                    'OP_Type_Comparison_A3_vs_A5_BF16': result.get('op_type_csv_bf16', ''),
                    'OP_Type_Comparison_A3_vs_A5_FP8': result.get('op_type_csv_fp8', '')
                })
        
        df = pd.DataFrame(summary_data)
        summary_file = self.output_path / "analysis_summary.csv"
        df.to_csv(summary_file, index=False, encoding='utf-8')
        
        print(f"Summary report generated: {summary_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Profiling批量分析工具')
    parser.add_argument('--a3-pros-path', type=str, required=True,
                       help='A3 pros文件夹路径')
    parser.add_argument('--a5-pros-path', type=str, required=True,
                       help='A5 pros文件夹路径')
    parser.add_argument('--mstt-path', type=str, required=True,
                       help='mstt仓库路径')
    parser.add_argument('--output-path', type=str, required=True,
                       help='输出路径')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['operator_only', 'ratio_only', 'both'],
                       help='运行模式: operator_only(只生成算子对比表), ratio_only(只生成低精度倍率表), both(两者都生成，默认)')
    
    args = parser.parse_args()
    
    try:
        analyzer = ProfilingBatchAnalyzer(
            args.a3_pros_path,
            args.a5_pros_path,
            args.mstt_path,
            args.output_path,
            args.mode
        )
        
        analyzer.run()
        
        return 0
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
