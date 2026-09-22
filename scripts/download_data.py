"""Download public official archives with provenance and safe extraction.

Archives are large. No archive is bundled or downloaded by the test suite.
If a provider moves an archive, use --url with its replacement official URL.
"""

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

URLS = {
    "wisdm": "https://archive.ics.uci.edu/static/public/507/wisdm%2Bsmartphone%2Band%2Bsmartwatch%2Bactivity%2Band%2Bbiometrics%2Bdataset.zip",
    "pamap2": "https://archive.ics.uci.edu/static/public/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring.zip",
    "realworld": "https://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip",
}


def safe_target(root, name):
    root = Path(root).resolve()
    target = (root / name).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"unsafe archive path: {name}")
    return target


def extract(archive, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target = safe_target(output, member.filename)
                if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError("archive symlinks are not supported")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            for member in tar:
                target = safe_target(output, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    raise ValueError("archive links and special entries are not supported")
    else:
        raise ValueError(f"not a ZIP or TAR archive: {archive}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=URLS)
    p.add_argument("--output", default="data/raw")
    p.add_argument("--url")
    p.add_argument("--sha256", help="Optional published expected SHA-256")
    p.add_argument("--extract", action="store_true")
    p.add_argument(
        "--nested", action="store_true", help="Also extract archives inside the main archive"
    )
    a = p.parse_args()
    root = Path(a.output) / a.dataset
    root.mkdir(parents=True, exist_ok=True)
    url = a.url or URLS[a.dataset]
    archive = root / "download.archive"
    tmp = root / "download.part"
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as src, tmp.open("wb") as dst:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
            dst.write(block)
    checksum = digest.hexdigest()
    if a.sha256 and checksum.lower() != a.sha256.lower():
        tmp.unlink()
        raise ValueError("SHA-256 mismatch")
    tmp.replace(archive)
    (root / "download.json").write_text(
        json.dumps(dict(url=url, sha256=checksum, verified_expected_hash=bool(a.sha256)), indent=2)
        + "\n"
    )
    if a.extract or a.nested:
        extract(archive, root / "extracted")
    if a.nested:
        # Process finite archive tree once per distinct path, without modifying source files.
        visited = set()
        while True:
            candidates = [
                f
                for f in (root / "extracted").rglob("*")
                if f.is_file()
                and f.suffix.lower() in (".zip", ".tar", ".gz", ".tgz")
                and f not in visited
            ]
            if not candidates:
                break
            for f in candidates:
                visited.add(f)
                extract(f, f.parent / (f.name + "_contents"))
    print(root)


if __name__ == "__main__":
    main()
