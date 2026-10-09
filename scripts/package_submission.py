#!/usr/bin/env python3
"""检查正式模型交接文件，并打包课程代码 ZIP（报告 PDF 和视频单独提交）。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
CNN_PATH = Path("outputs/classifier/b_retrain_seed20260918")
AE_PATH = Path("outputs/discovery/autoencoder")
CNN_RECORD = Path("docs/results/classifier/test_evaluation")
AE_RECORD = Path("docs/results/discovery/autoencoder_run.json")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8")


def source_files():
    files = {}
    for name in git("ls-files", "-z").split("\0"):
        path = Path(name)
        if not name:
            continue
        if path.parts[0] not in {"scripts", "src", "tests", "docs", "data"}:
            if name not in {"README.md", "AGENTS.md", "requirements.txt", ".gitignore"}:
                continue
        if any(part in {"build", "cache", "raw", "parts", "__pycache__"}
               for part in path.parts):
            continue
        if path.suffix.lower() in {
            ".cool", ".mcool", ".scool", ".tar", ".gz", ".zip", ".7z",
            ".pt", ".pth", ".ckpt", ".mp4", ".mkv", ".webm",
        }:
            continue
        files[name] = ROOT / path
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-dir", type=Path, default=ROOT / CNN_PATH,
                        help="B 的正式 CNN 目录，包含 best.pt、run.json、splits.csv")
    parser.add_argument("--ae-dir", type=Path, default=ROOT / AE_PATH,
                        help="C 的正式自编码器目录，包含 best.pt、run.json")
    parser.add_argument("--ae-sha256", help="可选：C 原有交接清单中的正式 AE 权重 SHA-256")
    parser.add_argument("--output-zip", type=Path, default=ROOT / "dist/code-submission.zip")
    parser.add_argument("--check-only", action="store_true", help="只检查；缺文件或校验不符时退出码为 1")
    args = parser.parse_args()
    if args.output_zip.suffix.lower() != ".zip":
        parser.error("--output-zip 应以 .zip 结尾")
    if args.ae_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.ae_sha256):
        parser.error("--ae-sha256 应为 64 位十六进制哈希")

    files = source_files()
    for directory, destination, names in (
        (args.cnn_dir, CNN_PATH, ("best.pt", "run.json", "splits.csv")),
        (args.ae_dir, AE_PATH, ("best.pt", "run.json")),
    ):
        files.update({(destination / name).as_posix(): directory / name for name in names})
    errors = [f"缺少文件：{path}" for path in files.values() if not path.is_file()]
    analysis = read_json(ROOT / CNN_RECORD / "analysis_run.json")
    expected_hashes = {
        (CNN_PATH / "best.pt").as_posix(): analysis["checkpoint_sha256"],
        (CNN_PATH / "splits.csv").as_posix(): analysis["split_file_sha256"],
    }
    if args.ae_sha256:
        expected_hashes[(AE_PATH / "best.pt").as_posix()] = args.ae_sha256.lower()

    for name, expected in expected_hashes.items():
        if files[name].is_file() and sha256(files[name]) != expected:
            errors.append(f"SHA-256 与正式记录不符：{files[name]}")
    for actual, saved in (
        (args.cnn_dir / "run.json", ROOT / CNN_RECORD / "training_run.json"),
        (args.ae_dir / "run.json", ROOT / AE_RECORD),
    ):
        if actual.is_file():
            received, recorded = read_json(actual), read_json(saved)
            # 目录与耗时可能不同，只核对决定此次实验身份的字段。
            keys = {
                "seed": None,
                "model": ("name", "parameters", "dropout", "class_weighted",
                          "latent_dim", "diagonal_exclusion"),
                "training": ("epochs_requested", "best_epoch", "best_val_macro_f1",
                             "best_val_loss", "batch_size", "lr", "weight_decay", "patience"),
            }
            for section, names in keys.items():
                if names is None:
                    mismatch = received.get(section) != recorded.get(section)
                else:
                    mismatch = any(received.get(section, {}).get(name) != value
                                   for name, value in recorded.get(section, {}).items()
                                   if name in names)
                if mismatch:
                    errors.append(f"运行记录的 {section} 与正式记录不符：{actual}；对照 {saved}")

    if errors:
        print("代码包尚未就绪：\n" + "\n".join(f"- {error}" for error in errors), file=sys.stderr)
        print("请接收 B/C 的正式产物；不使用 A 的历史 CNN，也不为打包重训。", file=sys.stderr)
        return 1
    print(f"检查通过：{len(files)} 个文件；CNN/划分 SHA-256 及两份运行记录关键参数一致。")
    if args.ae_sha256:
        print("AE 权重 SHA-256 与指定交接清单值一致。")
    else:
        print("AE 权重将记录实际 SHA-256；来源须以 C 的正式交接包确认。")
    if args.check_only:
        return 0
    if args.output_zip.exists():
        parser.error(f"输出已存在，请更换文件名：{args.output_zip}")

    manifest = {
        "git_commit": git("rev-parse", "HEAD").strip(),
        "working_tree_has_changes": bool(git("status", "--porcelain").strip()),
        "model_records": {
            "cnn": (CNN_RECORD / "analysis_run.json").as_posix(),
            "cnn_training": (CNN_RECORD / "training_run.json").as_posix(),
            "autoencoder_training": AE_RECORD.as_posix(),
            "autoencoder_expected_sha256": args.ae_sha256,
            "autoencoder_handoff": "C 的 mask 修订后正式产物；包名及收件核验见交付记录",
        },
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in sorted(files.items())
        },
    }
    readme = """# 课程代码提交包

解压后从本目录运行。安装依赖：`python -m pip install -r requirements.txt`。
正式运行顺序、命令和结果位置见 `docs/运行与交付清单.md`，数据来源见 `data/README.md`。

- B 的正式分类权重、划分与运行记录：`outputs/classifier/b_retrain_seed20260918/`。
- C 的正式自编码器权重与运行记录：`outputs/discovery/autoencoder/`。
- 既有实验结果与代表图：`docs/results/`；报告与演示编辑源：`docs/report/`、`docs/slides/`。
- 本包不含原始矩阵、缓存、扫描数组、全部 465 张轨道或视频；按运行说明生成需要的中间输入。
- 报告 PDF 与讲解视频单独提交。保留正式指标，不把其他 checkpoint 的成绩混入当前结果。

`SUBMISSION_MANIFEST.json` 保存打包时的代码版本、逐文件 SHA-256 及模型依据。
该清单不表示所有命令已在本机复跑；实际验证范围以交付清单中的记录为准。
"""
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output_zip, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr("SUBMISSION_README.md", readme)
        archive.writestr("SUBMISSION_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"已生成：{args.output_zip}（{args.output_zip.stat().st_size:,} 字节）")
    print(f"ZIP SHA-256：{sha256(args.output_zip)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
