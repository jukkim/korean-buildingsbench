"""Residential vs Commercial: 예측 vs 실측 시각화

논문 best 체크포인트로 대표 건물의 시계열을 비교한다.
- Commercial: BDG-2/Electricity에서 median 부근 건물 2개
- Residential: LCL/IDEAL에서 median 부근 건물 2개
- 부하 스케일/패턴 차이를 명확히 보여줌
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import tomllib as tomli
except ModuleNotFoundError:
    import tomli
import torch

from src.models.transformer import model_factory

PROJECT_DIR = Path(__file__).parent.parent
BB_DATA_DIR = PROJECT_DIR / 'external' / 'BuildingsBench_data'


def load_model_and_boxcox(checkpoint_path, config_path):
    with open(config_path, 'rb') as f:
        config = tomli.load(f)
    model_args = config['model']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_factory(model_args)
    model.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()

    bc_path = PROJECT_DIR / 'data' / 'korean_bb' / 'metadata' / 'transforms'
    sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
    from evaluate_bb import BBBoxCoxTransform
    boxcox = BBBoxCoxTransform(bc_path)

    return model, boxcox, model_args, device


def get_building_predictions(model, boxcox, building_info, model_args, device, max_windows=20):
    """건물 1개의 context+prediction 시계열을 추출"""
    sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
    from evaluate_bb import BBBuildingDataset

    context_len = model_args['context_len']
    pred_len = model_args['pred_len']
    btype_int = 0 if building_info['building_type'] == 'residential' else 1

    dataset = BBBuildingDataset(
        series=building_info['series'],
        latlon=building_info['latlon'],
        building_type_int=btype_int,
        context_len=context_len,
        pred_len=pred_len,
        boxcox_transform=boxcox,
        timestamps=building_info.get('timestamps'),
    )

    if len(dataset) == 0:
        return None

    from torch.utils.data import DataLoader
    n_use = min(len(dataset), max_windows)
    indices = np.linspace(0, len(dataset) - 1, n_use, dtype=int)

    all_ctx_orig = []
    all_pred_orig = []
    all_tgt_orig = []

    model.eval()
    for idx in indices:
        batch = dataset[idx]
        batch_gpu = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            predictions, dist_params = model.predict(batch_gpu)

        ctx_bc = batch_gpu['load'][:, :context_len]
        tgt_bc = batch_gpu['load'][:, context_len:]

        ctx_orig = boxcox.inverse_transform(ctx_bc).squeeze().cpu().numpy()
        pred_orig = boxcox.inverse_transform(predictions).squeeze().cpu().numpy()
        tgt_orig = boxcox.inverse_transform(tgt_bc).squeeze().cpu().numpy()

        all_ctx_orig.append(ctx_orig)
        all_pred_orig.append(pred_orig)
        all_tgt_orig.append(tgt_orig)

    return {
        'context': all_ctx_orig,
        'prediction': all_pred_orig,
        'target': all_tgt_orig,
        'context_len': context_len,
        'pred_len': pred_len,
    }


def pick_median_buildings(metrics_csv, dataset_name, n=1):
    """CVRMSE median 부근 건물 선택"""
    df = pd.read_csv(metrics_csv)
    if 'cvrmse' in df.columns:
        sub = df[df['dataset'] == dataset_name].copy()
        col = 'cvrmse'
    else:
        sub = df[(df['dataset'] == dataset_name) & (df['metric'] == 'cvrmse')].copy()
        col = 'value'

    if len(sub) == 0:
        return []

    sub = sub.sort_values(col)
    mid = len(sub) // 2
    start = max(0, mid - n // 2)
    return sub.iloc[start:start + n]['building_id'].tolist()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/TransformerWithGaussian-M-v3-3k_ms_n50_s18000_revin_on_seed42_best.pt')
    parser.add_argument('--config', type=str,
                        default='configs/model/TransformerWithGaussian-M-v3-3k.toml')
    parser.add_argument('--output', type=str, default='results/residential_gap_analysis.png')
    args = parser.parse_args()

    print("Loading model...")
    model, boxcox, model_args, device = load_model_and_boxcox(args.checkpoint, args.config)

    print("Parsing BB buildings...")
    sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
    from evaluate_bb import parse_bb_buildings
    all_buildings = parse_bb_buildings()
    bld_map = {b['building_id']: b for b in all_buildings}

    metrics_csv = PROJECT_DIR / 'results' / 'bb_metrics_TransformerWithGaussian-M-v3-3k.csv'
    use_full = False
    if not metrics_csv.exists():
        metrics_csv = PROJECT_DIR / 'results' / 'bb_metrics_TransformerWithGaussian-M-v3-final.csv'
        use_full = True

    # ---- Pick representative buildings ----
    targets = [
        ('BDG-2', 'commercial'),
        ('Electricity', 'commercial'),
        ('LCL', 'residential'),
        ('IDEAL', 'residential'),
    ]

    selected = []
    for ds, btype in targets:
        bids = pick_median_buildings(str(metrics_csv), ds, n=1)
        if bids:
            bid = bids[0]
            if bid in bld_map:
                selected.append((ds, btype, bid, bld_map[bid]))
                print(f"  {ds} ({btype}): {bid}")

    if len(selected) < 4:
        for ds, btype in targets:
            candidates = [b for b in all_buildings if b['dataset'] == ds]
            if candidates and not any(s[0] == ds for s in selected):
                b = candidates[len(candidates) // 2]
                selected.append((ds, btype, b['building_id'], b))
                print(f"  {ds} ({btype}, fallback): {b['building_id']}")

    # ---- Get predictions ----
    print("\nGenerating predictions...")
    results = {}
    for ds, btype, bid, bld_info in selected:
        print(f"  {ds}/{bid}...", end="", flush=True)
        res = get_building_predictions(model, boxcox, bld_info, model_args, device, max_windows=15)
        if res:
            results[(ds, btype, bid)] = res
            print(f" OK ({len(res['context'])} windows)")
        else:
            print(" SKIP (no data)")

    # ---- Compute per-building CVRMSE for these buildings ----
    for key, res in results.items():
        all_se = []
        all_gt = []
        for pred, tgt in zip(res['prediction'], res['target']):
            all_se.append((pred - tgt) ** 2)
            all_gt.append(np.abs(tgt))
        se = np.concatenate(all_se)
        gt = np.concatenate(all_gt)
        res['cvrmse'] = float(np.sqrt(np.mean(se)) / np.mean(gt)) * 100

    # ============================================================
    # Figure 1: 4-panel time series (2 commercial + 2 residential)
    # ============================================================
    print("\nPlotting Figure 1: Time series comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Korean-700 Predictions: Commercial vs Residential Buildings',
                 fontsize=14, fontweight='bold', y=0.98)

    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    for idx, ((ds, btype, bid), res) in enumerate(results.items()):
        ax = axes[idx // 2][idx % 2]
        ctx_len = res['context_len']
        pred_len = res['pred_len']

        win_idx = len(res['context']) // 2
        ctx = res['context'][win_idx]
        pred = res['prediction'][win_idx]
        tgt = res['target'][win_idx]

        hours_ctx = np.arange(ctx_len)
        hours_pred = np.arange(ctx_len, ctx_len + pred_len)

        ax.plot(hours_ctx, ctx, color='#2196F3', alpha=0.6, linewidth=0.8, label='Context (actual)')
        ax.plot(hours_pred, tgt, color='#2196F3', linewidth=1.5, label='Target (actual)')
        ax.plot(hours_pred, pred, color='#FF5722', linewidth=1.5, linestyle='--', label='Prediction')
        ax.axvline(x=ctx_len, color='gray', linestyle=':', alpha=0.5)

        unit = 'kWh'
        mean_load = np.mean(np.abs(tgt))
        if mean_load < 1:
            unit = 'Wh'
            ctx = ctx * 1000
            pred_scaled = pred * 1000
            tgt_scaled = tgt * 1000
            ax.clear()
            ax.plot(hours_ctx, ctx, color='#2196F3', alpha=0.6, linewidth=0.8, label='Context (actual)')
            ax.plot(hours_pred, tgt_scaled, color='#2196F3', linewidth=1.5, label='Target (actual)')
            ax.plot(hours_pred, pred_scaled, color='#FF5722', linewidth=1.5, linestyle='--', label='Prediction')
            ax.axvline(x=ctx_len, color='gray', linestyle=':', alpha=0.5)

        cvrmse = res['cvrmse']
        type_label = 'COMMERCIAL' if btype == 'commercial' else 'RESIDENTIAL'
        ax.set_title(f"{panel_labels[idx]} {ds} — {type_label}\n"
                     f"CVRMSE={cvrmse:.1f}%, mean={mean_load:.3f} kWh/h",
                     fontsize=11)
        ax.set_xlabel('Hour')
        ax.set_ylabel(f'Load ({unit})')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out1 = args.output
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out1}")
    plt.close()

    # ============================================================
    # Figure 2: Distribution comparison (violin/box)
    # ============================================================
    print("Plotting Figure 2: Error distribution + load scale...")

    fig2 = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # (a) CVRMSE distribution by dataset
    ax1 = fig2.add_subplot(gs[0, 0])
    df_full = None
    for csv_candidate in [
        'results/bb_metrics_TransformerWithGaussian-M-v3-3k.csv',
        'results/bb_metrics_TransformerWithGaussian-M-v3-final.csv',
    ]:
        try:
            tmp = pd.read_csv(csv_candidate)
            if 'residential' in tmp.get('building_type', pd.Series()).values:
                if 'cvrmse' in tmp.columns:
                    df_full = tmp
                elif 'metric' in tmp.columns:
                    df_full = tmp[tmp['metric'] == 'cvrmse'].rename(columns={'value': 'cvrmse'})
                break
        except:
            continue

    if df_full is not None:
        datasets_order = ['BDG-2', 'Electricity', 'Borealis', 'LCL', 'IDEAL']
        colors_map = {
            'BDG-2': '#4CAF50', 'Electricity': '#8BC34A',
            'Borealis': '#FF9800', 'LCL': '#F44336', 'IDEAL': '#E91E63',
        }
        box_data = []
        labels = []
        colors_list = []
        for ds in datasets_order:
            sub = df_full[df_full['dataset'] == ds]
            if len(sub) > 0:
                vals = sub['cvrmse'].clip(upper=2.0)
                box_data.append(vals.values * 100)
                n = len(sub)
                med = sub['cvrmse'].median() * 100
                labels.append(f'{ds}\n(n={n}, med={med:.1f}%)')
                colors_list.append(colors_map.get(ds, '#999'))

        bp = ax1.boxplot(box_data, labels=labels, patch_artist=True, widths=0.6,
                         showfliers=False, medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% CVRMSE')
        ax1.set_ylabel('CVRMSE (%)', fontsize=11)
        ax1.set_title('(a) CVRMSE by Dataset (capped at 200%)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')

    # (b) Load scale comparison
    ax2 = fig2.add_subplot(gs[0, 1])
    if df_full is not None:
        for ds in datasets_order:
            sub = df_full[df_full['dataset'] == ds]
            if len(sub) > 0:
                col = 'mean_actual'
                loads = sub[col].values
                ax2.scatter(loads, sub['cvrmse'].values * 100,
                           alpha=0.3, s=15, label=f'{ds} (n={len(sub)})',
                           color=colors_map.get(ds, '#999'))

        ax2.set_xscale('log')
        ax2.set_xlabel('Mean Hourly Load (kWh) — log scale', fontsize=11)
        ax2.set_ylabel('CVRMSE (%)', fontsize=11)
        ax2.set_title('(b) Load Scale vs Error', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 250)
        ax2.legend(fontsize=8, loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
        ax2.annotate('1 kWh/h\n(residential/commercial\nboundary)',
                    xy=(1.0, 220), fontsize=8, color='red', ha='center')

    # (c) Weekly pattern: commercial example
    ax3 = fig2.add_subplot(gs[1, 0])
    com_key = [k for k in results.keys() if k[1] == 'commercial']
    if com_key:
        res_c = results[com_key[0]]
        ctx = res_c['context'][0]
        n_show = min(168, len(ctx))
        ax3.plot(range(n_show), ctx[:n_show], color='#4CAF50', linewidth=1.0)
        ax3.set_title(f'(c) Commercial Weekly Pattern — {com_key[0][0]}',
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Hour')
        ax3.set_ylabel('Load (kWh)')
        ax3.grid(True, alpha=0.3)
        for d in range(0, n_show, 24):
            ax3.axvline(x=d, color='gray', alpha=0.15)

    # (d) Weekly pattern: residential example
    ax4 = fig2.add_subplot(gs[1, 1])
    res_key = [k for k in results.keys() if k[1] == 'residential']
    if res_key:
        res_r = results[res_key[0]]
        ctx = res_r['context'][0]
        n_show = min(168, len(ctx))
        vals = ctx[:n_show]
        if np.mean(np.abs(vals)) < 1:
            vals = vals * 1000
            ylabel = 'Load (Wh)'
        else:
            ylabel = 'Load (kWh)'
        ax4.plot(range(n_show), vals, color='#F44336', linewidth=1.0)
        ax4.set_title(f'(d) Residential Weekly Pattern — {res_key[0][0]}',
                     fontsize=12, fontweight='bold')
        ax4.set_xlabel('Hour')
        ax4.set_ylabel(ylabel)
        ax4.grid(True, alpha=0.3)
        for d in range(0, n_show, 24):
            ax4.axvline(x=d, color='gray', alpha=0.15)

    out2 = args.output.replace('.png', '_distribution.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out2}")
    plt.close()

    # ============================================================
    # Figure 3: Multi-window overlay (best visual for gap)
    # ============================================================
    print("Plotting Figure 3: Multi-window prediction overlay...")
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 5))

    for col_idx, btype in enumerate(['commercial', 'residential']):
        ax = axes3[col_idx]
        keys = [k for k in results.keys() if k[1] == btype]
        if not keys:
            continue
        res = results[keys[0]]
        ds_name = keys[0][0]
        cvrmse = res['cvrmse']

        n_win = min(8, len(res['context']))
        cmap = plt.cm.viridis(np.linspace(0.2, 0.9, n_win))

        for i in range(n_win):
            pred = res['prediction'][i]
            tgt = res['target'][i]
            scale = 1000 if np.mean(np.abs(tgt)) < 1 else 1
            hours = np.arange(len(tgt))
            ax.plot(hours, tgt * scale, color=cmap[i], alpha=0.5, linewidth=1.0)
            ax.plot(hours, pred * scale, color=cmap[i], alpha=0.8, linewidth=1.0, linestyle='--')

        unit = 'Wh' if btype == 'residential' and np.mean(np.abs(res['target'][0])) < 1 else 'kWh'
        type_upper = btype.upper()
        ax.set_title(f'{type_upper} — {ds_name} (CVRMSE={cvrmse:.1f}%)\n'
                     f'Solid=actual, Dashed=predicted ({n_win} windows)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Prediction Hour (24h)')
        ax.set_ylabel(f'Load ({unit})')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out3 = args.output.replace('.png', '_overlay.png')
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out3}")
    plt.close()

    # ---- Summary stats ----
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for (ds, btype, bid), res in results.items():
        mean_load = np.mean([np.mean(np.abs(t)) for t in res['target']])
        print(f"  {ds} ({btype}): CVRMSE={res['cvrmse']:.1f}%, mean_load={mean_load:.4f} kWh/h, bid={bid}")

    print(f"\nOutput files:")
    print(f"  {out1}")
    print(f"  {out2}")
    print(f"  {out3}")


if __name__ == '__main__':
    main()
