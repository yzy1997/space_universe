"""
run_inference.py  —  GENERAL inference script
────────────────────────────────────────────────────────────────
Supports ANY model and ANY preprocessing pipeline.

HOW TO SAVE YOUR MODEL (two options — pick one):
─────────────────────────────────────────────────
  Option A — Full model (recommended, no class needed at inference):
      torch.save(model, 'MyModel.pth')

  Option B — State dict only (class definition required):
      torch.save(model.state_dict(), 'MyModel.pth')
      → pass --model_py and --model_class so the script can rebuild it

HOW TO SAVE YOUR TRANSFORMS (optional):
─────────────────────────────────────────
  In your training notebook, create a file  my_transforms.py:
      from torchvision import transforms
      inference_transform = transforms.Compose([
          transforms.ToPILImage(),
          transforms.ToTensor(),
      ])
  Then pass --transform_py my_transforms.py --transform_name inference_transform

USAGE EXAMPLES (in Jupyter):
──────────────────────────────
  # Option A — full model, with external transform file
  %run run_inference.py --test_h5 Galaxy10_DECals_20pct.h5 \\
      --model_pth MyModel.pth \\
      --transform_py my_transforms.py --transform_name inference_transform

  # Option B — state dict, class loaded from file
  %run run_inference.py --test_h5 Galaxy10_DECals_20pct.h5 \\
      --model_pth MyModel.pth \\
      --model_py my_model.py --model_class MyModelClass \\
      --transform_py my_transforms.py --transform_name inference_transform

  # Minimal — full model, no preprocessing
  %run run_inference.py --test_h5 Galaxy10_DECals_20pct.h5 --model_pth MyModel.pth
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import argparse, importlib.util, os, sys
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='General Galaxy10 inference script')

# Data
parser.add_argument('--test_h5',        type=str, default='Galaxy10_DECals_20pct.h5')
parser.add_argument('--label_key',      type=str, default='ans',
                    help='HDF5 key for labels (default: ans)')
parser.add_argument('--image_key',      type=str, default='images',
                    help='HDF5 key for images (default: images)')
# Model
parser.add_argument('--model_pth',      type=str, default='MyModel.pth',
                    help='Path to .pth file (full model OR state dict)')
parser.add_argument('--model_py',       type=str, default=None,
                    help='[state dict only] Path to .py file containing model class')
parser.add_argument('--model_class',    type=str, default=None,
                    help='[state dict only] Class name inside --model_py')
# Transforms
parser.add_argument('--transform_py',   type=str, default=None,
                    help='Path to .py file containing the transform object')
parser.add_argument('--transform_name', type=str, default='inference_transform',
                    help='Variable name of the transform in --transform_py (default: inference_transform)')
# Inference
parser.add_argument('--batch_size',     type=int, default=50)
parser.add_argument('--out_npy',        type=str, default='y_pred_test.npy')
parser.add_argument('--num_classes',    type=int, default=10)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_module_from_file(filepath, module_name='_user_module'):
    """Dynamically import a .py file as a module."""
    spec   = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def describe_transform(t):
    """Return a readable pipeline string for any transforms.Compose."""
    if t is None:
        return 'None (raw numpy uint8 array passed directly)'
    if hasattr(t, 'transforms'):
        return ' -> '.join(type(s).__name__ for s in t.transforms)
    return str(type(t).__name__)

# ── Scope summary ─────────────────────────────────────────────────────────────
print('=' * 64)
print('  Galaxy10 Inference — Scope')
print('=' * 64)
print(f'  Test HDF5       : {args.test_h5}')
print(f'  Model weights   : {args.model_pth}')
print(f'  Model source    : {"state dict  (--model_py " + args.model_py + ")" if args.model_py else "full model (torch.save(model, ...))"}')
print(f'  Transform file  : {args.transform_py or "none — ToTensor fallback will be used"}')
print(f'  Batch size      : {args.batch_size}')
print(f'  Output file     : {args.out_npy}')
print(f'  Device          : {device}')
print('=' * 64)

# ── [1/4] Load test data ──────────────────────────────────────────────────────
print('\n[1/4] Loading test data ...')
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

with h5py.File(args.test_h5, 'r') as f:
    print(f'  HDF5 keys available : {list(f.keys())}')
    X_test = f[args.image_key][:]
    y_test = f[args.label_key][:]

print(f'  Images : {X_test.shape}  dtype={X_test.dtype}')
print(f'  Labels : {y_test.shape}  unique={np.unique(y_test)}')

# ── [2/4] Transforms ──────────────────────────────────────────────────────────
print('\n[2/4] Resolving inference transform ...')

if args.transform_py:
    # Load transform object from the user-supplied file
    user_module        = load_module_from_file(args.transform_py)
    inference_transform = getattr(user_module, args.transform_name)
    print(f'  Source   : {args.transform_py}  ->  {args.transform_name}')
    print(f'  Pipeline : {describe_transform(inference_transform)}')
else:
    # Sensible fallback: ToPILImage + ToTensor (works for uint8 HxWxC arrays)
    inference_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    print('  No --transform_py given.')
    print(f'  Fallback pipeline : {describe_transform(inference_transform)}')
    print('  TIP: create a transforms file and pass --transform_py to match training exactly.')

# ── [3/4] Dataset & DataLoader ────────────────────────────────────────────────
class GalaxyDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img

print('\n[3/4] Building DataLoader ...')
test_dataset = GalaxyDataset(X_test, labels=None, transform=inference_transform)
test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
print(f'  Samples : {len(test_dataset)}   Batches : {len(test_loader)}')

# ── [4/4] Load model ──────────────────────────────────────────────────────────
print(f'\n[4/4] Loading model from "{args.model_pth}" ...')

checkpoint = torch.load(args.model_pth, map_location=device)

if isinstance(checkpoint, dict) and 'model_source' in checkpoint:
    # ── Enriched dict: state_dict + model_source + model_class + num_classes ──
    print('  Detected: enriched checkpoint (state_dict + model_source)')
    ns = {}
    exec(checkpoint['model_source'], ns)
    ModelClass = ns[checkpoint['model_class']]
    num_classes  = checkpoint.get('num_classes', args.num_classes)
    model        = ModelClass(num_classes=num_classes)
    model.load_state_dict(checkpoint['state_dict'])
    print(f'  Class       : {checkpoint["model_class"]}  (loaded from checkpoint source)')
    print(f'  Num classes : {num_classes}')

elif isinstance(checkpoint, dict):
    # ── Plain state dict: needs --model_py and --model_class ─────────────────
    print('  Detected: plain state dict')
    if not args.model_py or not args.model_class:
        raise ValueError(
            '\n  This .pth file contains only weights (plain state dict).\n'
            '  Re-run with:\n'
            '    --model_py   path/to/your_model.py\n'
            '    --model_class YourModelClassName\n'
            '  OR re-save using the enriched format:\n'
            '    torch.save({"state_dict": model.state_dict(),\n'
            '                "model_source": "<class as string>",\n'
            '                "model_class": model.__class__.__name__,\n'
            '                "num_classes": 10}, "MyModel.pth")'
        )
    user_module  = load_module_from_file(args.model_py)
    ModelClass   = getattr(user_module, args.model_class)
    model        = ModelClass(num_classes=args.num_classes)
    model.load_state_dict(checkpoint)
    print(f'  Class    : {args.model_class}  (from {args.model_py})')

else:
    # ── Full model object (torch.save(model, ...)) ────────────────────────────
    print('  Detected: full model object')
    model = checkpoint

model.to(device)
model.eval()
print(f'  Architecture : {model.__class__.__name__}')
print(f'  Parameters   : {sum(p.numel() for p in model.parameters()):,}')
print('  Status       : eval mode')

# ── Inference ─────────────────────────────────────────────────────────────────
print('\n-- Running inference -----------------------------------------------')
y_pred_list   = []
y_scores_list = []

with torch.no_grad():
    for batch_idx, imgs in enumerate(test_loader):
        imgs    = imgs.to(device)
        outputs = model(imgs)
        probs   = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
        y_pred_list.extend(predicted.cpu().numpy())
        y_scores_list.extend(probs.cpu().numpy())

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
            done = min((batch_idx + 1) * args.batch_size, len(test_dataset))
            print(f'  Processed {done}/{len(test_dataset)} ...')

y_pred   = np.array(y_pred_list)
y_scores = np.array(y_scores_list)
y_true   = y_test

np.save(args.out_npy, y_pred)
np.save(args.out_npy.replace('.npy', '_probs.npy'), y_scores)

# ── Scores & report ───────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Disturbed', 'Merging', 'Round Smooth', 'In-between Round',
    'Cigar Shaped', 'Barred Spiral', 'Unbarred Tight', 'Unbarred Loose',
    'Edge-on w/o Bulge', 'Edge-on w/ Bulge',
]

acc = accuracy_score(y_true, y_pred)

print('\n' + '=' * 64)
print(f'  Model              : {model.__class__.__name__}')
print(f'  Transform pipeline : {describe_transform(inference_transform)}')
print(f'  Test samples       : {len(y_true)}')
print(f'  Overall Accuracy   : {acc * 100:.2f}%')

unique_pred = np.unique(y_pred)
if len(unique_pred) == 1:
    print(f'\n  WARNING: Model predicts only class {unique_pred[0]} for every sample.')
    print('  The model likely collapsed during training.')
    print('  Suggestions: more epochs, lower lr, check class balance.')
elif len(unique_pred) < 5:
    print(f'\n  WARNING: Model uses only {len(unique_pred)} out of {args.num_classes} classes.')
    print(f'  Predicted classes: {unique_pred}')

print('\n  Classification Report:')
print(classification_report(y_true, y_pred,
      labels=list(range(args.num_classes)),
      target_names=CLASS_NAMES[:args.num_classes], digits=4))

print('  Confusion Matrix (rows=true, cols=predicted):')
cm     = confusion_matrix(y_true, y_pred, labels=list(range(args.num_classes)))
header = '        ' + '  '.join(f'{i:>3}' for i in range(args.num_classes))
print(header)
for i, row in enumerate(cm):
    print(f'  [{i:2}]  ' + '  '.join(f'{v:>3}' for v in row))
print('\n  Class legend: ' + ' | '.join(f'{i}={n}' for i, n in enumerate(CLASS_NAMES)))

print('=' * 64)
print(f'\n  Saved: {args.out_npy}  (class predictions)')
print(f'  Saved: {args.out_npy.replace(".npy", "_probs.npy")}  (softmax probabilities)')
print('\nSubmit y_pred_test.npy together with your notebook.')
