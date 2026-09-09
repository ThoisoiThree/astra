#!/usr/bin/env python3
"""Build Astra.app with the icon generated from logo/astra_icon.svg."""

import json
import pathlib
import plistlib
import shutil
import subprocess
import sys


def main():
    if sys.platform != "darwin":
        raise SystemExit("Run this script on macOS.")
    root = pathlib.Path(__file__).resolve().parents[1]
    binary = None
    icon = None
    package_id = None
    outputs = {}
    process = subprocess.Popen(
        ["cargo", "build", "--release", "--bin", "astra", "--message-format=json-render-diagnostics"],
        cwd=root,
        stdout=subprocess.PIPE,
        text=True,
    )
    for line in process.stdout:
        message = json.loads(line)
        if message["reason"] == "compiler-message":
            sys.stderr.write(message["message"].get("rendered") or "")
        elif message["reason"] == "build-script-executed":
            outputs[message["package_id"]] = pathlib.Path(message["out_dir"])
        elif (
            message["reason"] == "compiler-artifact"
            and message["target"]["name"] == "astra"
            and message.get("executable")
        ):
            binary = pathlib.Path(message["executable"])
            package_id = message["package_id"]
    if process.wait():
        raise SystemExit(process.returncode)
    if binary is None or package_id not in outputs:
        raise SystemExit("Cargo did not report the Astra executable and icon output directory.")
    icon = outputs[package_id] / "astra.icns"
    # Derive the version from Cargo rather than duplicating it in a plist.
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=root, text=True
    ))
    package = next(p for p in metadata["packages"] if p["id"] == package_id)
    bundle = binary.parent / "Astra.app"
    contents = bundle / "Contents"
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    (contents / "Resources").mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, contents / "MacOS" / "astra")
    shutil.copy2(icon, contents / "Resources" / "astra.icns")
    with (contents / "Info.plist").open("wb") as stream:
        plistlib.dump({
            "CFBundleName": "Astra",
            "CFBundleDisplayName": "Astra",
            "CFBundleExecutable": "astra",
            "CFBundleIdentifier": "io.thoisoithree.astra",
            "CFBundlePackageType": "APPL",
            "CFBundleIconFile": "astra.icns",
            "CFBundleShortVersionString": package["version"],
            "CFBundleVersion": package["version"],
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": "Copyright Thoisoi Three. AGPL-3.0-only.",
        }, stream)
    # Signing comes after copying all resources so the seal covers the icon.
    subprocess.run(["codesign", "--force", "--sign", "-", str(bundle)], check=True)
    print(bundle)


if __name__ == "__main__":
    main()
