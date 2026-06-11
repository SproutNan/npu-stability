import os
import csv
import argparse
import numpy as np
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Generate summary from log files')
    parser.add_argument('--ci-path', type=str, help='path to ci logs')
    parser.add_argument('--baseline-path', type=str, help='path to baseline logs')
    parser.add_argument('--cidate', type=str)
    return parser.parse_args()


def parse_log_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        times, loss, mem = [], [], []
        world_size, global_batch_size, seq_length = 0, 0, 0
        for line in f:
            if 'elapsed time per iteration (ms):' in line:
                start = line.index('elapsed time per iteration (ms):') + len('elapsed time per iteration (ms):')
                time = line[start:].split()[0]
                times.append(float(time))
            if 'lm loss:' in line:
                start = line.index('lm loss:') + len('lm loss:')
                lmloss = line[start:].split()[0]
                loss.append(float(lmloss))
            if ' world_size ' in line:
                world_size = int(line.split()[-1])
            if ' global_batch_size ' in line:
                global_batch_size = int(line.split()[-1])
            if ' seq_length ' in line:
                seq_length = int(line.split()[-1])
            if ' Memory Usage: ' in line:
                mem = int(line.split()[-1])
        if not world_size:
            return ['No world_size found! Please check the ST log!']
        if not global_batch_size:
            return ['No global_batch_size found! Please check the ST log!']
        if not seq_length:
            return ['No seq_length found! Please check the ST log!']
        if not times:
            return ['No iteration time found! Please check the ST log!']
        if not mem:
            return ['No memory usage found! Please check the ST log!']
        if not loss:
            return ['No loss found! Please check the ST log!']
        avg_time = sum(times[-50:]) / len(times[-50:])
        tps = int(global_batch_size * seq_length / avg_time / world_size * 1000 + 0.5)
        return [tps, mem] + loss


def compare_csv_file(baseline_path, ci_path, summary_path, st_name, cidate):
    detail_path = os.path.join(baseline_path, 'detail.csv')
    his_path = os.path.join(baseline_path, 'history.csv')
    newbaseflag = 0
    if 'detail.csv' not in os.listdir(args.baseline_path):
        newbaseflag = 1
        shutil.copy(ci_path, baseline_path)
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['st'] + st_name)
    l_t_m = [cidate]
    t_relative = ['tps compared to baseline']
    m_relative = ['mem compared to baseline(MB)']
    alarm = ['alarm']
    with open(ci_path, 'r', encoding='utf-8') as file1:
        reader1 = csv.reader(file1)
        next(reader1)
        for row1 in reader1:
            while "" in row1:
                row1.remove("")
            if len(row1) == 2:
                l_t_m.append(row1[1])
                t_relative.append('')
                m_relative.append('')
                alarm.append('Attention!!')
                continue
            with open(detail_path, 'r', encoding='utf-8') as file2:
                reader2 = csv.reader(file2)
                next(reader2)
                compflag = 0
                for row2 in reader2:
                    if row1[0] != row2[0]:
                        continue
                    while "" in row2:
                        row2.remove("")
                    compflag = 1
                    if len(row1) != len(row2):
                        l_t_m.append('Number of skipped iteration is different compared to baseline!')
                        t_relative.append('')
                        m_relative.append('')
                        alarm.append('Attention!!')
                        break
                    ci = list(map(float, row1[1:]))
                    base = list(map(float, row2[1:]))
                    tps_diff = (ci[0] - base[0]) / base[0]
                    mem_diff = ci[1] - base[1]
                    ci_loss, base_loss = np.array(ci[2:]), np.array(base[2:])
                    mean_err = np.mean(np.abs(ci_loss - base_loss) / base_loss)
                    if newbaseflag:
                        l_t_m[0] += ' 基线'
                        l_t_m.append('tps: ' + row1[1] + ' | mem: ' + row1[2])
                    else:
                        l_t_m.append(
                            'loss mre: ' + '{:.4f}'.format(mean_err) + ' | tps: ' + row1[1] + ' | mem: ' + row1[2])
                    t_relative.append('{:.4%}'.format(tps_diff))
                    m_relative.append(int(mem_diff))
                    if tps_diff < -0.05:
                        alarm.append('Attention!!')
                        l_t_m[-1] += ' --tps alarm!'
                    elif mem_diff > 5000:
                        alarm.append('Attention!!')
                        l_t_m[-1] += ' --mem alarm!'
                    elif mean_err > 0.01:
                        alarm.append('Attention!!')
                        l_t_m[-1] += ' --loss alarm!'
                    else:
                        alarm.append('')
                    break
                if not compflag:
                    l_t_m.append('Info not found in baseline detail.csv')
                    t_relative.append('')
                    m_relative.append('')
                    alarm.append('Attention!!')
        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(l_t_m)
            writer.writerow(t_relative)
            writer.writerow(m_relative)
            writer.writerow(alarm)
        with open(his_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if newbaseflag:
                writer.writerow(['ST'] + st_name)
            writer.writerow(l_t_m)


if __name__ == '__main__':
    args = parse_args()
    if not args.ci_path:
        raise Exception('ci log path not found! please use --ci-path')
    st_name = sorted(os.listdir(args.ci_path))
    if 'result.log' in st_name:
        st_name.remove('result.log')
    detail_path = os.path.join(args.ci_path, 'detail.csv')
    with open(detail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['st', 'tps', 'memory usage', 'loss'])
    for file_path in st_name:
        if file_path.endswith('.log'):
            full_path = os.path.join(args.ci_path, file_path)
            row = [file_path] + parse_log_file(full_path)
            with open(detail_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
    if args.baseline_path:
        summary_path = os.path.join(args.ci_path, 'summary.csv')
        compare_csv_file(args.baseline_path, detail_path, summary_path, st_name, args.cidate[:5])
