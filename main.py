import os
import argparse
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ExitHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)


def _count_conv2d_flops(module: nn.Conv2d, inp: torch.Tensor, out: torch.Tensor) -> float:
    batch_size = out.shape[0]
    out_channels = out.shape[1]
    out_h = out.shape[2]
    out_w = out.shape[3]
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
    output_elements = batch_size * out_channels * out_h * out_w
    bias_ops = 1 if module.bias is not None else 0
    return float(output_elements * (2.0 * kernel_ops + bias_ops))


def _count_linear_flops(module: nn.Linear, inp: torch.Tensor, out: torch.Tensor) -> float:
    batch_size = out.shape[0]
    add_ops = 1 if module.bias is not None else 0
    return float(batch_size * module.in_features * (2 * module.out_features + add_ops))


def _module_flops(module: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> float:
    if isinstance(module, nn.Conv2d):
        return _count_conv2d_flops(module, inp, out)
    if isinstance(module, nn.Linear):
        return _count_linear_flops(module, inp, out)
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        return float(out.numel() * 2)
    if isinstance(module, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Sigmoid, nn.GELU, nn.Hardswish, nn.LeakyReLU)):
        return float(out.numel())
    return 0.0


def _run_with_flops(modules: List[nn.Module], inp: torch.Tensor) -> Tuple[torch.Tensor, float]:
    hooks = []
    module_flops: List[float] = [0.0]

    def _hook(module: nn.Module, module_in: Tuple[torch.Tensor], module_out: torch.Tensor):
        x = module_in[0]
        y = module_out
        if isinstance(y, (tuple, list)):
            y = y[0]
        if not torch.is_tensor(y):
            return
        if not torch.is_tensor(x):
            return
        if not x.is_floating_point():
            return
        module_flops[0] += _module_flops(module, x, y)

    for mod in modules:
        for layer in mod.modules():
            if layer is mod:
                continue
            if len(list(layer.children())) > 0:
                continue
            hooks.append(layer.register_forward_hook(_hook))

    with torch.no_grad():
        out = inp
        for mod in modules:
            out = mod(out)

    for hook in hooks:
        hook.remove()

    return out, module_flops[0]


def _format_flops(flops: float) -> str:
    if flops >= 1_000_000_000:
        return f"{flops / 1_000_000_000:.2f} GFLOPs"
    if flops >= 1_000_000:
        return f"{flops / 1_000_000:.2f} MFLOPs"
    if flops >= 1_000:
        return f"{flops / 1_000:.2f} KFLOPs"
    return f"{flops:.0f} FLOPs"


def print_exit_flops(model: nn.Module, sample_size: int = 224, device: torch.device = torch.device("cpu")):
    model = model.eval()
    sample = torch.randn(1, 3, sample_size, sample_size, device=device)

    if hasattr(model, "block1"):
        cum_flops = 0.0
        x, f = _run_with_flops([model.block1], sample)
        cum_flops += f
        _, f = _run_with_flops([model.exit1], x)
        cum_flops += f
        exit1_flops = cum_flops

        x, f = _run_with_flops([model.block2], x)
        cum_flops += f
        _, f = _run_with_flops([model.exit2], x)
        cum_flops += f
        exit2_flops = cum_flops

        x, f = _run_with_flops([model.block3], x)
        cum_flops += f
        _, f = _run_with_flops([model.exit3], x)
        cum_flops += f
        exit3_flops = cum_flops

        x, f = _run_with_flops([model.block4, model.avgpool], x)
        cum_flops += f
        x = torch.flatten(x, 1)
        classifier_seq = nn.Sequential(model.classifier)
        _, f = _run_with_flops([model.classifier], x)
        cum_flops += f
        final_flops = cum_flops

        print("[FLOPS] cumulative FLOPs per exit:")
        print(f"  Exit 1: {_format_flops(exit1_flops)}")
        print(f"  Exit 2: {_format_flops(exit2_flops)}")
        print(f"  Exit 3: {_format_flops(exit3_flops)}")
        print(f"  Final: {_format_flops(final_flops)}")
    else:
        print("[FLOPS] model does not expose early-exit blocks; unable to compute per-exit FLOPs.")


class EarlyExitEfficientNetB0(nn.Module):
    """
    3 early exits + 1 final classifier.

    This implementation uses torchvision EfficientNet-B0 and splits its
    `features` Sequential into chunks.

    Exit placement used here:
    - exit1 after features[0:5]
    - exit2 after features[5:6]
    - exit3 after features[6:8]
    - final after features[8:] + original head

    This is a practical mapping for the Stage 4 / Stage 5 / Stage 7 plan.
    """
    def __init__(self, num_classes: int, pretrained: bool = True, sample_size: int = 224):
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b0(weights=weights)

        feats = base.features  # Sequential backbone

        # Chunk the backbone
        self.block1 = nn.Sequential(*feats[:5])   # up to first early exit
        self.block2 = nn.Sequential(*feats[5:6])  # up to second early exit
        self.block3 = nn.Sequential(*feats[6:8])  # up to third early exit
        self.block4 = nn.Sequential(*feats[8:])   # final feature block

        with torch.no_grad():
            dummy = torch.zeros(1, 3, sample_size, sample_size)
            ex1 = self.block1(dummy)
            ex2 = self.block2(ex1)
            ex3 = self.block3(ex2)
            c1 = ex1.shape[1]
            c2 = ex2.shape[1]
            c3 = ex3.shape[1]

        self.exit1 = ExitHead(in_channels=c1, num_classes=num_classes, dropout=0.0)
        self.exit2 = ExitHead(in_channels=c2, num_classes=num_classes, dropout=0.0)
        self.exit3 = ExitHead(in_channels=c3, num_classes=num_classes, dropout=0.0)

        # Reuse the official final head structure
        final_in = base.classifier[1].in_features
        if hasattr(base, "avgpool"):
            self.avgpool = base.avgpool
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(final_in, num_classes),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []

        x = self.block1(x)
        outputs.append(self.exit1(x))

        x = self.block2(x)
        outputs.append(self.exit2(x))

        x = self.block3(x)
        outputs.append(self.exit3(x))

        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        outputs.append(self.classifier(x))

        return outputs


def multi_exit_loss(
    outputs: List[torch.Tensor],
    targets: torch.Tensor,
    weights: List[float] = None,
) -> Tuple[torch.Tensor, List[float]]:
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]

    assert len(outputs) == len(weights)

    losses = []
    for out in outputs:
        losses.append(F.cross_entropy(out, targets))

    total = 0.0
    for w, loss in zip(weights, losses):
        total += w * loss

    return total, [loss.item() for loss in losses]


def vprint(enabled: bool, msg: str):
    if enabled:
        print(msg)


def format_time(seconds: float) -> str:
    seconds = float(seconds)
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    verbose: bool = False,
    log_every: int = 25,
):
    model.eval()

    correct = [0, 0, 0, 0]
    total = 0
    loss_sum = 0.0
    running_correct = [0, 0, 0, 0]
    running_total = 0

    for step, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss, _ = multi_exit_loss(outputs, targets)
        loss_sum += loss.item() * images.size(0)

        total += targets.size(0)
        running_total += targets.size(0)
        for i, out in enumerate(outputs):
            preds = out.argmax(dim=1)
            correct[i] += (preds == targets).sum().item()
            running_correct[i] += (preds == targets).sum().item()

        if verbose and running_total > 0 and (step + 1) % max(1, log_every) == 0:
            batch_accs = [c / running_total for c in running_correct]
            vprint(
                verbose,
                f"  [EVAL] step={step+1}/{len(loader)} "
                f"batch_loss={loss.item():.4f} "
                f"accs={batch_accs[0]:.4f},{batch_accs[1]:.4f},{batch_accs[2]:.4f},{batch_accs[3]:.4f}",
            )
            running_total = 0
            running_correct = [0, 0, 0, 0]

    accs = [c / total for c in correct]
    vprint(verbose, f"  [EVAL] completed: samples={total}, loss={loss_sum / total:.4f}, accs={accs}")
    return loss_sum / total, accs


def train_one_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    verbose: bool = False,
    log_every: int = 25,
):
    model.train()
    running_loss = 0.0
    total = 0
    start = time.perf_counter()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss, _ = multi_exit_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        if verbose and (step + 1) % max(1, log_every) == 0:
            now = time.perf_counter()
            elapsed = now - start
            vprint(
                verbose,
                f"  [TRAIN] step={step+1}/{len(loader)} running_loss={running_loss / total:.4f} "
                f"throughput={total / max(1e-9, elapsed):.1f} img/s "
                f"elapsed={format_time(elapsed)}",
            )

    epoch_time = time.perf_counter() - start
    return running_loss / total, epoch_time


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    verbose: bool = False,
):
    rng = random.Random(seed)

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    vprint(verbose, f"[DATA] loading dataset from {data_root}")
    full_ds = datasets.ImageFolder(data_root)
    if len(full_ds.classes) == 0:
        raise ValueError(f"No class subdirectories found in data_root: {data_root}")
    vprint(verbose, f"[DATA] found {len(full_ds)} files, {len(full_ds.classes)} classes")
    for cls_idx, cls_name in enumerate(full_ds.classes):
        cls_count = sum(1 for _, target in full_ds.samples if target == cls_idx)
        vprint(verbose, f"[DATA] class={cls_name} count={cls_count}")
    class_to_indices: Dict[int, List[int]] = {}
    for idx, target in enumerate(full_ds.targets):
        if target not in class_to_indices:
            class_to_indices[target] = []
        class_to_indices[target].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for target in sorted(class_to_indices.keys()):
        inds = class_to_indices[target][:]
        rng.shuffle(inds)

        n = len(inds)
        if n == 1:
            n_train, n_val, n_test = 1, 0, 0
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)
            n_test = n - n_train - n_val

            if n_val == 0:
                n_val = 1
            if n_test == 0:
                if n_train > 1:
                    n_train -= 1
                else:
                    n_val -= 1
                n_test = 1

            while n_train + n_val + n_test > n:
                if n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    n_train -= 1

            while n_train + n_val + n_test < n:
                n_train += 1

        train_indices.extend(inds[:n_train])
        val_indices.extend(inds[n_train : n_train + n_val])
        test_indices.extend(inds[n_train + n_val :])
        vprint(
            verbose,
            f"[SPLIT] class={full_ds.classes[target]} -> total={n}, train={n_train}, val={n_val}, test={n_test}",
        )

    train_ds = datasets.ImageFolder(data_root, transform=train_tfms)
    eval_ds = datasets.ImageFolder(data_root, transform=val_tfms)
    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(eval_ds, val_indices)
    test_subset = Subset(eval_ds, test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    vprint(verbose, f"[DATA] train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}")
    return train_loader, val_loader, test_loader, full_ds.classes


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-exit EfficientNet-B0 on ImageFolder data.")
    parser.add_argument("--data-root", type=str, required=True, help="Path containing class-wise image folders.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per process.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image resize dimension.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for reproducible stratified split.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    pin_args = parser.add_mutually_exclusive_group()
    pin_args.add_argument("--pin-memory", action="store_true", help="Enable pin_memory in DataLoader.")
    pin_args.add_argument("--no-pin-memory", action="store_false", dest="pin_memory", help="Disable pin_memory in DataLoader.")
    parser.set_defaults(pin_memory=True)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    verbosity.add_argument("--quiet", action="store_true", help="Disable verbose logging.")
    parser.set_defaults(verbose=True)
    parser.add_argument("--log-every", type=int, default=25, help="Log every N batches when verbose.")
    pretrained = parser.add_mutually_exclusive_group()
    pretrained.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained EfficientNet-B0.")
    pretrained.add_argument("--not-pretrained", action="store_false", dest="pretrained", help="Disable pretrained backbone.")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated CUDA device indices to use, e.g. '0,1'.")
    parser.add_argument("--multi-gpu", action="store_true", help="Enable DataParallel when multiple GPUs are available.")
    parser.add_argument("--save-path", type=str, default="early_exit_efficientnet_b0_plantvillage.pt", help="Checkpoint output path.")
    parser.add_argument("--disable-cuda", action="store_true", help="Force CPU training.")
    return parser.parse_args()


def select_visible_gpus(gpus: str) -> List[int]:
    if not gpus:
        return list(range(torch.cuda.device_count()))
    ids = [x.strip() for x in gpus.split(",") if x.strip() != ""]
    if len(ids) == 0:
        return list(range(torch.cuda.device_count()))
    selected = []
    for g in ids:
        selected.append(int(g))
    return selected


def main():
    args = parse_args()
    verbose = args.verbose and not args.quiet
    if args.quiet:
        args.verbose = False
    set_seed(args.seed)
    vprint(verbose, f"[SETUP] args={vars(args)}")
    vprint(verbose, f"[SEED] set to {args.seed}")

    pin_memory = bool(args.pin_memory)

    gpu_ids = select_visible_gpus(args.gpus)
    if args.disable_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        parallel_enabled = False
        vprint(verbose, "[DEVICE] using CPU")
    else:
        if args.gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            gpu_ids = list(range(len(gpu_ids)))
        if not gpu_ids:
            gpu_ids = [0]
        device = torch.device(f"cuda:{gpu_ids[0]}")
        torch.cuda.set_device(device)
        parallel_enabled = args.multi_gpu and len(gpu_ids) > 1
        vprint(
            verbose,
            f"[DEVICE] available CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}",
        )
        vprint(verbose, f"[DEVICE] selected device {device}")
        if parallel_enabled:
            vprint(verbose, f"[DEVICE] DataParallel enabled on {gpu_ids}")

    train_loader, val_loader, test_loader, classes = make_loaders(
        args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        seed=args.seed,
        verbose=verbose,
    )
    num_classes = len(classes)
    vprint(verbose, f"[DATA] classes={classes}")

    vprint(verbose, "[MODEL] building EarlyExitEfficientNetB0")
    model = EarlyExitEfficientNetB0(
        num_classes=num_classes,
        pretrained=args.pretrained,
        sample_size=args.image_size,
    ).to(device)
    print_exit_flops(model, sample_size=args.image_size, device=device)

    if parallel_enabled:
        model = nn.DataParallel(model)
        vprint(verbose, f"Using DataParallel across {len(gpu_ids)} CUDA devices: {gpu_ids}")
        vprint(verbose, f"Using first device as anchor: {device}")

    vprint(verbose, f"[TRAIN] optimizer=AdamW(lr={args.lr}, weight_decay={args.weight_decay})")
    vprint(verbose, f"[TRAIN] scheduler=CosineAnnealingLR(T_max={args.epochs})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_final_acc = 0.0
    save_path = Path(args.save_path)
    final_checkpoint = str(save_path.with_name(f"{save_path.stem}_final{save_path.suffix}"))
    last_epoch_acc = 0.0
    cumulative_train_time = 0.0
    ckpt_dir = save_path.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        vprint(verbose, f"[TRAIN] starting epoch {epoch + 1}/{args.epochs}")
        train_loss, epoch_train_time = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            verbose=verbose,
            log_every=args.log_every,
        )
        cumulative_train_time += epoch_train_time
        remaining_epochs = args.epochs - (epoch + 1)
        avg_epoch_time = cumulative_train_time / float(epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        vprint(verbose, f"[VAL] evaluating epoch {epoch + 1}/{args.epochs}")
        val_loss, val_accs = evaluate(
            model,
            val_loader,
            device,
            verbose=verbose,
            log_every=args.log_every,
        )
        last_epoch_acc = val_accs[3]
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"epoch_time={format_time(epoch_train_time)} | "
            f"elapsed_train={format_time(cumulative_train_time)} | "
            f"eta={format_time(eta_seconds)} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"exit1={val_accs[0]:.4f} exit2={val_accs[1]:.4f} "
            f"exit3={val_accs[2]:.4f} final={val_accs[3]:.4f}"
        )

        if val_accs[3] > best_final_acc:
            best_final_acc = val_accs[3]
            torch.save(
                {
                    "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                    "classes": classes,
                    "best_final_acc": best_final_acc,
                    "multi_gpu": parallel_enabled,
                },
                args.save_path,
            )
        epoch_ckpt = ckpt_dir / f"{save_path.stem}_epoch_{epoch+1:03d}{save_path.suffix}"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "classes": classes,
                "val_loss": val_loss,
                "val_accs": val_accs,
                "best_final_acc": best_final_acc,
                "multi_gpu": parallel_enabled,
            },
            str(epoch_ckpt),
        )

    torch.save(
        {
            "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            "classes": classes,
            "best_final_acc": best_final_acc,
            "last_epoch_acc": last_epoch_acc,
            "multi_gpu": parallel_enabled,
            "epochs_trained": args.epochs,
        },
        final_checkpoint,
    )
    vprint(verbose, f"[CHECKPOINT] saved final checkpoint to {final_checkpoint}")

    print(f"Best final accuracy: {best_final_acc:.4f}")

    test_loss, test_accs = evaluate(model, test_loader, device, verbose=verbose, log_every=args.log_every)
    print(
        f"Final test | "
        f"test_loss={test_loss:.4f} | "
        f"exit1={test_accs[0]:.4f} exit2={test_accs[1]:.4f} "
        f"exit3={test_accs[2]:.4f} final={test_accs[3]:.4f}"
    )


if __name__ == "__main__":
    main()
