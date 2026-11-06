# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Host-only regression checks for launcher paths after the core/apps split."""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="kiwi-layout-")
        self.addCleanup(self.tmp.cleanup)
        self.env = {
            "PATH": os.environ["PATH"],
            "HOME": self.tmp.name,
            "DRY_RUN": "1",
            "OUTPUT_DIR": self.tmp.name,
            "LOG_DIR": self.tmp.name,
            "PASS_INFINIBAND": "0",
        }

    def run_script(self, path, *args, **env):
        return subprocess.check_output(
            ["bash", str(ROOT / path), *args],
            cwd=self.tmp.name,
            env={**self.env, **env},
            text=True,
            stderr=subprocess.PIPE,
        )

    def test_container_mounts_whole_checkout_and_sets_root(self):
        for mount in ("/workspace/kiwi", "/custom path/kiwi"):
            with self.subTest(mount=mount):
                env = {} if mount == "/workspace/kiwi" else {"CONTAINER_KIWI_DIR": mount}
                output = self.run_script(
                    "scripts/ep_bench/run_container.sh", "--dry-run", "--", "true", **env
                )
                args = shlex.split(output.removeprefix("+ "))
                mounts = [args[i + 1] for i, arg in enumerate(args) if arg == "-v"]
                self.assertIn(f"{ROOT}:{mount}:rw", mounts)
                self.assertFalse(any("/queue:" in value for value in mounts))
                self.assertIn(f"KIWI_REPO={mount}", args)
                self.assertEqual(args[args.index("-w") + 1], mount)

    def test_ep_smoke_explicitly_enables_ep_from_root(self):
        output = self.run_script(
            "scripts/ep_bench/run_kiwi_ep_smoke_correctness.sh", "--dry-run", "--no-python"
        )
        commands = [shlex.split(line[2:]) for line in output.splitlines() if line.startswith("+ ")]
        configure = commands[0]
        self.assertEqual(configure[configure.index("-S") + 1], str(ROOT))
        self.assertIn("-DKIWI_BUILD_EP=ON", configure)
        self.assertIn("-DKIWI_BUILD_TESTS=ON", configure)
        self.assertIn("-DKIWI_BUILD_PYTHON=OFF", configure)
        self.assertEqual(commands[-1][0], "ctest")

    def test_mori_wrapper_uses_ep_sources_and_binding_directory(self):
        output = self.run_script("scripts/ep_bench/run_mori_ep_compat.sh", "--dry-run")
        args = shlex.split(output.removeprefix("+ "))
        source = ROOT / "apps/ep/tests/test_official_mori_compat.py"
        self.assertTrue(source.is_file())
        self.assertIn(str(source), args)
        self.assertIn(f"PYTHONPATH={ROOT}/build/ep-bench/apps/ep/python", args)

    def test_comparison_wrapper_finds_root_from_nested_app(self):
        output = self.run_script(
            "apps/ep/benchmarks/python/run_ep_mori_compare.sh",
            RUN_MODE="bare", RUN_MORI_V1="0", RUN_MORI_V1LL="0",
        )
        self.assertIn(f"KIWI repo host path: {ROOT}", output)
        self.assertIn(str(ROOT / "apps/ep/benchmarks/python/bench_kiwi_ep.py"), output)
        self.assertIn(str(ROOT / "build/apps/ep/python"), output)

    def test_shell_syntax(self):
        scripts = list((ROOT / "scripts").rglob("*.sh"))
        scripts += list((ROOT / "apps").rglob("*.sh"))
        for script in scripts:
            with self.subTest(script=script.relative_to(ROOT)):
                subprocess.run(["bash", "-n", str(script)], check=True)


if __name__ == "__main__":
    unittest.main()
