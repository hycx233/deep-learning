#!/usr/bin/env python3
"""#9：固定候选的去重召回、WT 文件校验及双重复 O/E 验证。"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.window_splits import overlaps, window_segments

EXPECTED = {
    'WT_rep1': ('GSE272159_37C_rep1.mapq_30.10.cool', 482254398,
                '0256e10da36abfe0238bf16952bc43bc78365f43c201e6e3ba0c49bd243549a8', 368695972),
    'WT_rep2': ('GSE272159_37C_rep2.mapq_30.10.cool', 468345755,
                'ede9b4f7c0c6b40c8f098e43cdd0abbc0a8987f05e8b37051a5e3cbf143ad106', 358452621),
}
SEED = 20261002


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def segments(row):
    return [[int(s['start']), int(s['end'])] for s in json.loads(row['segments'])]


def candidate_audit():
    candidates = read_csv(ROOT / 'docs/results/discovery/candidates.csv')
    clusters = read_csv(ROOT / 'docs/results/clustering/candidate_clusters.csv')
    structures = read_csv(ROOT / 'data/structures.csv')
    assert len(candidates) == len(clusters) == 47, '固定的 47 个候选输入发生变化'
    assert {r['window_id'] for r in candidates} == {r['window_id'] for r in clusters}
    assert len({s['ID'] for s in structures}) == 344
    grouped = defaultdict(list)
    hits = set()
    for row in clusters:
        assert row['sample_id'] == 'WT_rep1' and int(row['length_bp']) == 24000
        assert abs(float(row['candidate_threshold']) - 0.04633597284555435) < 1e-14
        assert float(row['reconstruction_error']) >= float(row['candidate_threshold'])
        matched = {s['ID'] for s in structures if row['chrom'] == s['chrom'] and
                   overlaps(segments(row), [[int(s['start']), int(s['end'])]])}
        supplied = set(filter(None, row['known_structure_ids'].split(';')))
        assert matched == supplied, f"{row['candidate_id']}：坐标重算与原已知关系不一致"
        hits.update(matched)
        grouped[row['independent_locus_id']].append(row)
    assert len(grouped) == 10, '固定独立位置数发生变化'
    # 核对已保存分组确实是重叠连通分量；当前候选不跨原点。
    from src.candidate_clustering import assign_independent_loci
    import pandas as pd
    table = pd.DataFrame(clusters)
    table[['start', 'end']] = table[['start', 'end']].astype(int)
    assert (assign_independent_loci(table).to_numpy() == table['independent_locus_id'].to_numpy()).all()
    loci = []
    for locus, members in sorted(grouped.items()):
        # 只用 rep1 的最高重建误差窗口；并列时按稳定 ID，绝不查看 rep2 后挑窗口。
        representative = min(members, key=lambda r: (-float(r['reconstruction_error']), r['candidate_id']))
        loci.append({
            'independent_locus_id': locus, 'candidate_id': representative['candidate_id'],
            'window_id': representative['window_id'], 'chrom': representative['chrom'],
            'start': int(representative['start']), 'end': int(representative['end']),
            'center': int(representative['center']), 'length_bp': 24000,
            'segments': representative['segments'], 'member_window_count': len(members),
            'member_candidate_ids': ';'.join(r['candidate_id'] for r in members),
            'locus_union_start': min(int(r['start']) for r in members),
            'locus_union_end': max(int(r['end']) for r in members),
            'known_structure_ids': ';'.join(sorted({k for r in members for k in r['known_structure_ids'].split(';') if k})),
            'new_structure_status': 'not_eligible_under_fixed_cluster_rule',
            'rep2_window_id': representative['window_id'].replace('WT_rep1__', 'WT_rep2__'),
            'correlation_type': 'Pearson', 'replicate_correlation': '',
            'valid_upper_pixels': '', 'excluded_upper_pixels': '',
            'reproducibility_status': 'blocked_missing_raw',
            'reason': 'WT 原始矩阵尚未取得；不能判定复现，也不作生物学排除',
        })
    recall = []
    for kind, expected_hit, expected_total in [('OPCID', 10, 68), ('CHIN', 10, 250), ('CHID', 1, 26)]:
        ids = {s['ID'] for s in structures if s['type'] == kind}
        found = ids & hits
        assert (len(found), len(ids)) == (expected_hit, expected_total)
        recall.append({'type': kind, 'recalled_structures': len(found), 'total_structures': len(ids),
                       'recall': len(found) / len(ids), 'recalled_structure_ids': ';'.join(sorted(found)),
                       'scope': 'fixed_rep1_candidates_before_replicate_validation',
                       'coordinate_rule': '0-based half-open, any positive overlap; unique structure ID'})
    return clusters, structures, loci, recall


def controls(structures, clusters):
    rows = []
    for kind in ('OPCID', 'CHIN', 'CHID'):
        s = min((s for s in structures if s['type'] == kind), key=lambda s: int(s['ID'].split('_')[-1]))
        start, end, spans = window_segments(int(s['center']), 4641652, 24000)
        rows.append({'control_id': s['ID'], 'type': kind, 'chrom': s['chrom'],
                     'center': int(s['center']), 'start': start, 'end': end,
                     'segments': json.dumps([dict(start=a, end=b) for a, b in spans]),
                     'selection': 'first annotation ID per class; fixed before rep2'})
    centers = list(range(0, 4641652, 1000))
    random.Random(SEED).shuffle(centers)
    excluded = [[[int(s['start']), int(s['end'])]] for s in structures] + [segments(c) for c in clusters]
    background_spans = []
    for center in centers:
        start, end, spans = window_segments(center, 4641652, 24000)
        if any(overlaps(spans, other) for other in excluded + background_spans):
            continue
        background_spans.append(spans)
        rows.append({'control_id': f'background_{len(background_spans):02d}', 'type': 'background',
                     'chrom': 'NC_000913.3', 'center': center, 'start': start, 'end': end,
                     'segments': json.dumps([dict(start=a, end=b) for a, b in spans]),
                     'selection': f'1 kb grid, seed={SEED}, excludes annotations/candidates/other background'})
        if len(background_spans) == 3:
            break
    assert len(rows) == 6
    return rows


def wt_manifest(source):
    samples = [r for r in read_csv(source) if r['sample_id'] in EXPECTED]
    assert len(samples) == 2 and {r['sample_id'] for r in samples} == set(EXPECTED)
    samples.sort(key=lambda r: r['sample_id'])
    for row in samples:
        name, size, _, _ = EXPECTED[row['sample_id']]
        assert row['condition'] == 'WT' and int(row['replicate']) == int(row['sample_id'][-1])
        assert row['path'] == f'data/{name}' and int(row['file_size_bytes']) == size
        assert int(row['bin_size']) == 10 and int(row['genome_length']) == 4641652
    return samples


def verify_raw(samples):
    records = []
    for sample in samples:
        name, size, digest, count_sum = EXPECTED[sample['sample_id']]
        path = ROOT / sample['path']
        record = {'sample_id': sample['sample_id'], 'path': sample['path'],
                  'expected_size_bytes': size, 'expected_sha256': digest,
                  'observed_size_bytes': None, 'observed_sha256': None, 'status': 'missing'}
        records.append(record)
        if not path.is_file():
            continue
        record['observed_size_bytes'] = path.stat().st_size
        record['observed_sha256'] = sha256(path)
        record['status'] = 'invalid'
        if path.name != name or record['observed_size_bytes'] != size or record['observed_sha256'] != digest:
            record['reason'] = '文件名、字节数或 SHA-256 不匹配；停止使用'
            continue
        import cooler
        import h5py
        import numpy as np
        if not cooler.fileops.is_cooler(str(path)):
            record['reason'] = '不是 Cooler 格式；停止使用'
            continue
        c = cooler.Cooler(str(path))
        with h5py.File(path, 'r') as h:
            dtype = str(h['pixels/count'].dtype)
            has_weight = 'weight' in h['bins']
            # 在完整哈希已相符的条件下核对 counts 元数据及首尾像素。
            actual = np.r_[h['pixels/count'][:1000], h['pixels/count'][-1000:]]
        record['metadata'] = {'bin_size': int(c.binsize) if c.binsize is not None else None,
                              'chroms': {str(k): int(v) for k, v in c.chromsizes.items()},
                              'storage_mode': c.storage_mode, 'nbins': int(c.info['nbins']),
                              'nnz': int(c.info['nnz']), 'sum': int(c.info['sum']),
                              'count_dtype': dtype, 'has_weight': has_weight}
        good = (c.binsize == 10 and dict(c.chromsizes) == {'NC_000913.3': 4641652}
                and c.storage_mode == 'symmetric-upper' and dtype == sample['count_dtype']
                and not has_weight and int(c.info['nbins']) == int(sample['n_bins'])
                and int(c.info['nnz']) == int(sample['n_pixels']) and int(c.info['sum']) == count_sum
                and np.isfinite(actual).all() and (actual >= 0).all())
        record['status'] = 'verified' if good else 'invalid'
        if not good:
            record['reason'] = 'Cooler 分辨率、染色体或 counts 元数据不匹配；停止使用'
    return records


def paired_statistics(left, right):
    import numpy as np
    if not (np.array_equal(left['bin_ids'], right['bin_ids']) and
            np.array_equal(left['window_edges'], right['window_edges']) and
            left['oe'].shape == right['oe'].shape):
        raise ValueError('两个重复的实际读取坐标/bin 不一致')
    common = left['mask'] & right['mask'] & np.isfinite(left['oe']) & np.isfinite(right['oe'])
    upper = np.triu(np.ones(common.shape, dtype=bool), k=1)
    keep = common & upper
    n = int(keep.sum())
    result = {'correlation_type': 'Pearson', 'replicate_correlation': '',
              'valid_upper_pixels': n, 'excluded_upper_pixels': int(upper.sum()) - n,
              'reproducibility_status': 'unassessable', 'reason': ''}
    x, y = left['oe'][keep].astype(float), right['oe'][keep].astype(float)
    if n < 3:
        result['reason'] = '共同有效上三角像素少于 3'
    elif np.ptp(x) == 0 or np.ptp(y) == 0:
        result['reason'] = '至少一个重复的共同有效 O/E 为常量，Pearson 无定义'
    else:
        result.update(replicate_correlation=float(np.corrcoef(x, y)[0, 1]),
                      reproducibility_status='measured_pending_review',
                      reason='已测量；相关系数不是结构存在检验，须结合已知/背景及配对图审阅')
    return result, common


def plot_pair(left, right, common, label, path):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), layout='constrained')
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#bbbbbb')
    for ax, window, sample in zip(axes, (left, right), ('WT_rep1', 'WT_rep2')):
        edges = window['window_edges'] / 1000
        im = ax.pcolormesh(edges, edges, np.ma.array(window['oe'], mask=~common),
                           cmap=cmap, vmin=0, vmax=10, shading='flat', rasterized=True)
        ax.set_aspect('equal')
        ax.set_title(sample)
        ax.set_xlabel('Offset from shared start (kb)')
        ax.set_ylabel('Offset (kb)')
    fig.colorbar(im, ax=axes, label='O/E (fixed 0–10; gray = common invalid)')
    fig.suptitle(f"{label}\nNC_000913.3 [{left['start']}, {left['end']}) bp", fontsize=10)
    # PDF 保留矢量文字，避免本机 Agg 连续绘图的字形缓存缺失。
    fig.savefig(path.with_suffix('.pdf'), dpi=140)
    plt.close(fig)


def review_results(outdir):
    """将逐图审阅记录接回已实测表，不按 rep2 相关调阈值。"""
    import numpy as np
    loci = read_csv(outdir / 'candidate_loci_review.csv')
    reviews = read_csv(outdir / 'visual_review.csv')
    by_id = {r['candidate_id']: r for r in reviews}
    assert len(loci) == len(reviews) == len(by_id) == 10
    assert set(by_id) == {r['candidate_id'] for r in loci}
    for row in loci:
        assert row['replicate_correlation'] != '' and int(row['valid_upper_pixels']) >= 3
        note = by_id[row['candidate_id']]
        assert int(note['center']) == int(row['center'])
        row.update({k: note[k] for k in ['reproducibility_status', 'review_decision', 'visual_observation']})
        row['whole_window_assessable'] = int(row['excluded_upper_pixels']) == 0
        row['paired_plot'] = f"paired_oe/{row['candidate_id']}.pdf"
        row['reason'] = note['visual_observation'] + '；不作新结构证据'
    write_csv(outdir / 'candidate_loci_review.csv', loci)
    by_locus = {r['independent_locus_id']: r for r in loci}
    windows = read_csv(outdir / 'candidate_windows_review.csv')
    for row in windows:
        measured = by_locus[row['independent_locus_id']]
        for k in ['reproducibility_status', 'review_decision', 'visual_observation',
                  'whole_window_assessable', 'reason', 'paired_plot']:
            row[k] = measured[k]
    write_csv(outdir / 'candidate_windows_review.csv', windows)
    retained_ids = {i for r in loci if r['review_decision'].startswith('retain')
                    for i in r['known_structure_ids'].split(';') if i}
    structures = read_csv(ROOT / 'data/structures.csv')
    recall = read_csv(outdir / 'recall_by_class.csv')
    for row in recall:
        ids = {s['ID'] for s in structures if s['type'] == row['type']}
        row['retained_loci_recalled_structures'] = len(ids & retained_ids)
        row['retained_loci_recall'] = len(ids & retained_ids) / len(ids)
        row['retained_scope'] = 'reviewed contact patterns, including shared-region-only; annotation overlap, not per-structure replication test'
    write_csv(outdir / 'recall_by_class.csv', recall)
    controls = read_csv(outdir / 'controls_review.csv')
    r = [float(x['replicate_correlation']) for x in loci]
    summary = {
        'candidate_windows': 47, 'independent_loci': 10, 'measured_loci': len(r),
        'candidate_pearson_range': [min(r), max(r)], 'candidate_pearson_median': float(np.median(r)),
        'valid_upper_pixel_range': [min(int(x['valid_upper_pixels']) for x in loci), max(int(x['valid_upper_pixels']) for x in loci)],
        'retained_contact_patterns': sum(x['review_decision'].startswith('retain') for x in loci),
        'shared_region_only_loci': sum(int(x['excluded_upper_pixels']) > 0 for x in loci),
        'unassessable_loci': 0, 'clear_whole_pattern_failure_loci': 0,
        'known_overlap_loci': sum(bool(x['known_structure_ids']) for x in loci),
        'unannotated_loci': sum(not x['known_structure_ids'] for x in loci),
        'retained_new_structures': 0, 'new_structure_exclusion_rule': 'original clustering criteria unchanged and not satisfied',
        'control_pearson': {x['control_id']: float(x['replicate_correlation']) for x in controls},
        'correlation_not_an_independent_structure_test': True,
        'whole_locus_union_not_tested': True,
    }
    (outdir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    run = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'command': shlex.join(sys.argv),
           'source_sha256': sha256(Path(__file__)), 'visual_review_sha256': sha256(outdir / 'visual_review.csv'),
           'analysis_run_sha256': sha256(outdir / 'run_analyze.json'),
           'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'review_basis': 'all 10 candidate pairs and 6 preset controls visually inspected; no Pearson pass cutoff',
           'summary': summary}
    (outdir / 'run_review.json').write_text(json.dumps(run, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def plot_examples(states, loci, control_rows, outdir):
    """最高 rep1 误差候选、首条已知参照和首个固定背景，共用显示尺度。"""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.preprocessing import extract_window
    rows = [next(r for r in loci if r['candidate_id'] == 'candidate_035'),
            next(r for r in control_rows if r['control_id'] == 'OPCID_1'),
            next(r for r in control_rows if r['control_id'] == 'background_01')]
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 9.5), layout='constrained')
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#bbbbbb')
    for i, row in enumerate(rows):
        pair = [extract_window(state, int(row['center'])) for state in states]
        _, common = paired_statistics(*pair)
        label = row.get('candidate_id', row.get('control_id'))
        for j, window in enumerate(pair):
            edges = window['window_edges'] / 1000
            im = axes[i, j].pcolormesh(edges, edges, np.ma.array(window['oe'], mask=~common),
                                     vmin=0, vmax=10, cmap=cmap, shading='flat', rasterized=True)
            axes[i, j].set_aspect('equal')
            axes[i, j].set_title(f"{label} | WT_rep{j+1}\n[{window['start']}, {window['end']}) bp", fontsize=9)
            axes[i, j].set_xlabel('Offset (kb)', fontsize=9)
            axes[i, j].set_ylabel('Offset (kb)', fontsize=9)
            axes[i, j].tick_params(labelsize=8)
        axes[i, 0].text(0.02, 0.97, f"r={float(row['replicate_correlation']):.3f}; n={row['valid_upper_pixels']}",
                        transform=axes[i, 0].transAxes, ha='left', va='top', fontsize=8, color='white',
                        bbox={'facecolor': 'black', 'alpha': 0.65, 'edgecolor': 'none'})
    fig.colorbar(im, ax=axes, label='O/E (fixed 0-10; gray = shared invalid)', shrink=0.8)
    fig.savefig(outdir / 'paired_examples.pdf', dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['audit', 'verify', 'analyze', 'review'], default='audit')
    p.add_argument('--samples', type=Path, default=ROOT / 'data/samples.csv')
    p.add_argument('--cache-dir', type=Path, default=ROOT / 'data/cache/100bp')
    p.add_argument('--outdir', type=Path, default=ROOT / 'docs/results/validation')
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.stage == 'review':
        review_results(args.outdir)
        return
    if args.stage == 'audit' and (args.outdir / 'candidate_loci_review.csv').is_file():
        prior = read_csv(args.outdir / 'candidate_loci_review.csv')
        if any(r.get('replicate_correlation', '') != '' for r in prior):
            raise SystemExit('审阅表已有实测结果；请指定新的 --outdir，避免覆盖')
    clusters, structures, loci, recall = candidate_audit()
    control_rows = controls(structures, clusters)
    samples = wt_manifest(args.samples)
    write_csv(args.outdir / 'samples-wt.csv', samples)
    raw = verify_raw(samples)
    ready = all(r['status'] == 'verified' for r in raw)
    if args.stage == 'analyze' and ready:
        from src.preprocessing import load_sample, extract_window, BIN_SIZE
        states = [load_sample(args.cache_dir, s['sample_id']) for s in samples]
        for state, sample in zip(states, samples):
            identity = state['metadata']['cache_identity']
            path = ROOT / sample['path']
            assert identity['source_path'] == str(path.resolve())
            assert (identity['source_size'], identity['source_mtime_ns']) == (path.stat().st_size, path.stat().st_mtime_ns)
            assert state['metadata']['bin_size'] == BIN_SIZE == 100
            assert state['metadata']['total_counts'] == EXPECTED[sample['sample_id']][3]
        (args.outdir / 'paired_oe').mkdir(exist_ok=True)
        for row in loci + control_rows:
            pair = [extract_window(state, int(row['center'])) for state in states]
            assert all((w['start'], w['end']) == (int(row['start']), int(row['end'])) for w in pair)
            result, common = paired_statistics(*pair)
            row.update(result)
            label = row.get('candidate_id', row.get('control_id'))
            row['paired_plot'] = f'paired_oe/{label}.pdf'
            plot_pair(*pair, common, label, args.outdir / row['paired_plot'])
        write_csv(args.outdir / 'controls_review.csv', control_rows)
        plot_examples(states, loci, control_rows, args.outdir)
    if args.stage == 'audit':
        # 独立准备模式；有原始文件时也不会把未分析记录写成已测量。
        for r in loci:
            r['reproducibility_status'] = 'pending_analysis' if ready else 'blocked_raw_data'
            r['reason'] = '未执行双重复分析' if ready else 'WT 原始矩阵缺失或校验失败；无法判定复现'
    if args.stage == 'audit' or (args.stage == 'analyze' and ready):
        write_csv(args.outdir / 'candidate_loci_review.csv', loci)
        write_csv(args.outdir / 'recall_by_class.csv', recall)
        write_csv(args.outdir / 'controls_manifest.csv', controls(structures, clusters))
        by_locus = {r['independent_locus_id']: r for r in loci}
        review = []
        for original in clusters:
            row = dict(original)
            measured = by_locus[row['independent_locus_id']]
            for k in ('replicate_correlation', 'correlation_type', 'valid_upper_pixels', 'reproducibility_status', 'reason'):
                row[k] = measured[k]
            row['rep2_window_id'] = measured['rep2_window_id']
            row['correlation_scope'] = 'locus_representative_not_this_window'
            row['representative_candidate_id'] = measured['candidate_id']
            review.append(row)
        write_csv(args.outdir / 'candidate_windows_review.csv', review)
    def git(*arguments):
        return subprocess.check_output(['git', *arguments], cwd=ROOT, text=True).strip()
    run = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'command': shlex.join(sys.argv),
           'stage': args.stage, 'git_revision': git('rev-parse', 'HEAD'),
           'git_status': git('status', '--porcelain=v1'), 'seed': SEED,
           'raw_files': raw, 'analysis_executed': args.stage == 'analyze' and ready,
           'candidate_windows': 47, 'independent_loci': 10,
           'candidate_rule': 'rep1 top 1%, threshold=0.04633597284555435; unchanged',
           'representative_rule': 'highest rep1 reconstruction_error per saved locus; stable ID tie-break',
           'correlation': 'Pearson on shared finite native 100 bp O/E pixels, strict upper triangle k=1; no clipping',
           'oe_rule': 'shared preprocessing counts/expected_raw[circular distance], zeros included in background; no pseudocount',
           'plot_scale': 'raw O/E, fixed 0–10, common invalid pixels gray; clipping only for display',
           'retention_rule': 'no numeric pass threshold specified; measurement plus paired review; no new structure claim',
           'measured_loci': sum(r['reproducibility_status'] == 'measured_pending_review' for r in loci),
           'unassessable_loci': sum(r['reproducibility_status'] == 'unassessable' for r in loci),
           'pending_loci': sum(r['reproducibility_status'] in ('blocked_raw_data', 'blocked_missing_raw', 'pending_analysis') for r in loci),
           'source_sha256': {f: sha256(ROOT / f) for f in ['scripts/validate_wt_candidates.py', 'src/preprocessing.py', 'src/window_splits.py', 'src/candidate_clustering.py']},
           'input_sha256': {f: sha256(ROOT / f) for f in ['data/samples.csv', 'data/structures.csv', 'docs/results/discovery/candidates.csv', 'docs/results/clustering/candidate_clusters.csv']},
           'versions': {name: version(name) for name in ['numpy', 'cooler', 'h5py', 'matplotlib', 'pandas']},
           'python_version': sys.version,
           'cache_preparation': 'separate scripts/prepare_data.py --stage cache; consult its run_cache.json',
           'cache_metadata_sha256': {str(p): sha256(p) for p in args.cache_dir.glob('WT_rep[12].json')},
           'download_attempt_record': 'download_attempt.json (historical; no download performed by this script)'}
    (args.outdir / f'run_{args.stage}.json').write_text(json.dumps(run, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: run[k] for k in ['stage', 'analysis_executed', 'candidate_windows', 'independent_loci', 'measured_loci', 'unassessable_loci', 'pending_loci']}, ensure_ascii=False))
    if args.stage in ('verify', 'analyze') and not ready:
        raise SystemExit('WT 文件缺失或校验失败；停止缓存/跨重复分析。详见运行记录。')


if __name__ == '__main__':
    main()
