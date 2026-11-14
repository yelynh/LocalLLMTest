#!/usr/bin/env python3
"""Download the three requested AI Hub datasets with an API key kept out of argv.

Usage: python3 tools/download_aihub.py [71896 [71902 [71903]]]
Optional: AIHUB_KEY_FILE=/path/to/key AIHUB_DATA_DIR=/path/to/data
"""

import os
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path


DATASETS = {
    "71896": ("agent-action-capability", "에이전트 행위 능력 데이터"),
    "71902": ("high-quality-multiturn", "고품질 멀티턴 데이터 집합"),
    "71903": ("synthetic-instruction", "Synthetic & Instruction 데이터 집합"),
}
ROOT = Path(os.environ.get("AIHUB_DATA_DIR", Path(__file__).resolve().parents[1] / "data/raw/aihub"))
KEY_FILE = Path(os.environ.get("AIHUB_KEY_FILE", "/home/iris/workspace/key/aihub.key"))
PART_RE = re.compile(r"^(.*)\.part([0-9]+)$")


def record(dataset_id, state, detail=""):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with (ROOT / "download-status.tsv").open("a", encoding="utf-8") as out:
        out.write(f"{dataset_id}\t{DATASETS[dataset_id][0]}\t{state}\t{now}\t{detail}\n")
    print(f"{dataset_id}: {state}" + (f" ({detail})" if detail else ""), flush=True)


def download(dataset_id, key, target):
    url = f"https://api.aihub.or.kr/down/0.6/{dataset_id}.do?fileSn=all"
    archive = target / "download.tar"
    partial = target / "download.tar.part"
    if archive.exists():
        return archive

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"apikey": key}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=90)
    except urllib.error.HTTPError as error:
        message = error.read(512).decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"HTTP {error.code}: {message}") from None

    with response:
        status = response.status
        if status not in (200, 206):
            raise RuntimeError(f"unexpected HTTP {status}")
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise RuntimeError(f"unexpected Content-Range: {content_range}")
        mode = "ab" if status == 206 and offset else "wb"
        target.mkdir(parents=True, exist_ok=True)
        with partial.open(mode) as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
    partial.rename(archive)
    return archive


def extract_and_check(archive, target):
    staging = target / ".extracting"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        with tarfile.open(archive) as tar:
            tar.extractall(staging, filter="data")

        groups = {}
        for path in staging.rglob("*"):
            if path.is_file() and (match := PART_RE.match(path.name)):
                groups.setdefault((path.parent, match.group(1)), []).append((int(match.group(2)), path))
        for (parent, name), parts in groups.items():
            parts.sort()
            next_offset = 0
            for offset, part in parts:
                if offset != next_offset:
                    raise RuntimeError(f"missing archive part: {name} at byte {next_offset}")
                next_offset += part.stat().st_size
            output = parent / name
            if output.exists():
                raise RuntimeError(f"assembled file already exists: {output}")
            with output.open("wb") as merged:
                for _, part in parts:
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, merged, 1024 * 1024)
            for _, part in parts:
                part.unlink()

        zips = list(staging.rglob("*.zip"))
        if not zips:
            raise RuntimeError("archive contains no ZIP files")
        for path in zips:
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
                if bad:
                    raise RuntimeError(f"ZIP integrity failed: {path.name}, member {bad}")
        for item in staging.iterdir():
            destination = target / item.name
            if destination.exists():
                raise RuntimeError(f"destination already exists: {destination}")
            item.rename(destination)
        return zips
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def write_summary():
    latest = {}
    status_file = ROOT / "download-status.tsv"
    if status_file.exists():
        for line in status_file.read_text(encoding="utf-8").splitlines():
            columns = line.split("\t", 4)
            if len(columns) == 5 and columns[0] in DATASETS:
                latest[columns[0]] = columns
    lines = ["# AI Hub dataset status", "", "| ID | 데이터셋 | 상태 | 확인 시각 | 사유 |", "| --- | --- | --- | --- | --- |"]
    for dataset_id, (slug, name) in DATASETS.items():
        row = latest.get(dataset_id)
        state, when, detail = (row[2], row[3], row[4]) if row else ("not started", "", "")
        detail = detail.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {dataset_id} | {name} | {state} | {when} | {detail} |")
    lines.extend(["", "재시도: `python3 tools/download_aihub.py`", "저장 위치: `data/raw/aihub/<ID>-<slug>/`", ""])
    (ROOT / "DATASET_STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ids = sys.argv[1:] or list(DATASETS)
    if any(dataset_id not in DATASETS for dataset_id in ids):
        sys.exit("usage: download_aihub.py [71896] [71902] [71903]")
    key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        sys.exit(f"empty API key file: {KEY_FILE}")
    ROOT.mkdir(parents=True, exist_ok=True)
    failed = False
    for dataset_id in ids:
        slug, _ = DATASETS[dataset_id]
        target = ROOT / f"{dataset_id}-{slug}"
        if (target / ".complete").exists() and list(target.rglob("*.zip")):
            record(dataset_id, "complete", "already downloaded")
            continue
        record(dataset_id, "downloading")
        try:
            archive = download(dataset_id, key, target)
            zips = extract_and_check(archive, target)
            archive.unlink()
            (target / ".complete").write_text(datetime.now().astimezone().isoformat() + "\n")
            record(dataset_id, "complete", f"{len(zips)} ZIP file(s), integrity checked")
        except Exception as error:
            failed = True
            record(dataset_id, "failed", str(error).replace("\n", " ")[:500])
    write_summary()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
