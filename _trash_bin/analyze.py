import json

with open('results.json') as f:
    res = json.load(f)

print('=== Current Results ===')
print(f'Accuracy: {res["accuracy"]}')
print(f'AUC: {res["auc"]}')
print(f'F1 - Tampered: {res["f1_tampered"]}, Authentic: {res["f1_authentic"]}')
print(f'Precision: {res["precision"]}, Recall: {res["recall"]}')
print()
print('Ablation:')
for k, v in res.get('ablation', {}).items():
    print(f'  {k}: {v}')
print()
print('Robustness:')
for k, v in res.get('robustness', {}).items():
    print(f'  {k}: {v}')
print()
print('Per-class:')
for k, v in res.get('per_class', {}).items():
    print(f'  {k}: {v}')

print()
print('=== Generalization Analysis ===')
acc = res['accuracy']
auc = res['auc']

# Check if model generalizes well across attack types
robustness = res.get('robustness', {})
original_acc = robustness.get('Original', 0)
drops = {}
for k, v in robustness.items():
    if k != 'Original':
        drops[k] = original_acc - v

print(f'Drop from original on attacks:')
for k, v in drops.items():
    print(f'  {k}: {v:.4f} ({v/original_acc*100:.1f}%)')

# Per-class analysis
per_class = res.get('per_class', {})
print(f'\nPer-class breakdown:')
for k, v in per_class.items():
    gap = acc - v
    print(f'  {k}: {v} (gap from overall accuracy: {gap:.4f})')

# Check if copy-move is the weakest link
if 'copy_move' in per_class and 'splicing' in per_class:
    print(f'\nCopy-move vs Splicing gap: {per_class["splicing"] - per_class["copy_move"]:.4f}')
    if per_class['copy_move'] < per_class['splicing']:
        print('NOTE: Copy-move forgery detection is the weaker component')