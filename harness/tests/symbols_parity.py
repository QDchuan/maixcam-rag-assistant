"""一次性对拍：Python 校验器 与 JS 移植版 对同一批样本是否给同一个答案。

不是长期测试（那个是 harness/tests/symbols.test.mjs），只是移植之后的当场核对。
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from maixrag.evaluation.harness import check_symbols  # noqa: E402

SAMPLES = {
    "good": """from maix import camera, display, image, app

cam = camera.Camera(320, 240)
disp = display.Display()

while not app.need_exit():
    img = cam.read()
    blobs = img.find_blobs([[0, 80, 40, 80, 10, 80]], pixels_threshold=500)
    for blob in blobs:
        img.draw_rect(blob[0], blob[1], blob[2], blob[3], image.COLOR_GREEN)
    disp.show(img)
""",
    "hallucinated": """from maix import camera

cam = camera.Camera()
cam.super_zoom()
maix.image.Image.magic_method()
""",
    "alias": "import maix.image as im\nim.Image()\n",
    "enum_member": "maix.image.Format.FMT_RGB888\n",
    "class_without_method": "maix.camera.Camera.super_zoom\n",
    "no_maix": 'print("hello")\n',
    "from_import": "from maix.image import Image\nImage()\n",
}

ROSTER = REPO / "corpus" / "processed" / "api_roster.txt"
JS_MODULE = REPO / "harness" / "dsh-maixcam-shell" / "lib" / "symbols.js"


def python_results() -> dict[str, dict]:
    roster = {line.strip() for line in ROSTER.read_text(encoding="utf-8").splitlines() if line.strip()}
    out = {}
    for name, code in SAMPLES.items():
        unknown, checked = check_symbols(code, roster, assume_code=True)
        out[name] = {"unknown": list(unknown), "checked": checked}
    return out


def js_results() -> dict[str, dict]:
    script = f"""
import {{ readFileSync }} from 'node:fs'
import {{ checkSymbols }} from {json.dumps(JS_MODULE.as_uri())}
const samples = JSON.parse(readFileSync(process.argv[2], 'utf8'))
const roster = new Set(readFileSync({json.dumps(str(ROSTER))}, 'utf8').split('\\n').map(s => s.trim()).filter(Boolean))
const out = {{}}
for (const [name, code] of Object.entries(samples)) {{
  const r = checkSymbols(code, roster, {{ assumeCode: true }})
  out[name] = {{ unknown: r.unknown, checked: r.checked }}
}}
console.log(JSON.stringify(out))
"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        (tmp_path / "samples.json").write_text(json.dumps(SAMPLES), encoding="utf-8")
        (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
        proc = subprocess.run(
            ["node", str(tmp_path / "run.mjs"), str(tmp_path / "samples.json")],
            capture_output=True, text=True, encoding="utf-8",
        )
    if proc.returncode != 0:
        raise SystemExit(f"node 失败：{proc.stderr}")
    return json.loads(proc.stdout)


def main() -> int:
    py = python_results()
    js = js_results()
    bad = 0
    for name in SAMPLES:
        a, b = py[name], js[name]
        same = a == b
        if not same:
            bad += 1
        mark = "一致" if same else "**不一致**"
        print(f"  {mark}  {name:<22} python={a} js={b}")
    print()
    print(f"{len(SAMPLES)} 个样本，{bad} 个不一致")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
