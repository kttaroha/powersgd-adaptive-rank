# utils/network_limiter.py
import os
import shutil
import subprocess
from typing import Dict, Any

class NetworkLimiter:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("network_limits", {})
        self.enabled = bool(self.cfg.get("enable", False))

        # tc_apply.sh をこのファイル位置からの相対パスで解決（プロジェクト直下想定）
        self.script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "macvlan_run", "tc_apply.sh")
        )
        if not os.path.exists(self.script_path):
            # 旧配置(カレント直下)も一応見る
            alt = os.path.abspath("tc_apply.sh")
            if os.path.exists(alt):
                self.script_path = alt

        # NIC
        self.interface = self._resolve_interface(self.cfg.get("interface", "auto"))

        # 帯域: egress_mbps/ingress_mbps が無ければ bandwidth_mbps をフォールバック
        bw = self.cfg.get("bandwidth_mbps", None)
        self.egress = int(self.cfg.get("egress_mbps", bw if bw is not None else 0))
        self.ingress = int(self.cfg.get("ingress_mbps", self.egress))

        # 遅延・ジッタ・損失
        self.latency = int(self.cfg.get("latency_ms", 0))
        self.jitter  = int(self.cfg.get("jitter_ms", 0))
        self.loss    = float(self.cfg.get("loss_pct", 0))

        self.auto_cleanup = bool(self.cfg.get("auto_cleanup", True))

        # 事前チェック（あると安心）
        self._preflight_check()

    def _preflight_check(self):
        if not self.enabled:
            return
        for bin_name in ("tc", "ip", "bash"):
            if shutil.which(bin_name) is None:
                raise RuntimeError(f"[NetworkLimiter] '{bin_name}' not found in PATH")

        if os.geteuid() != 0:
            # rootでない場合、権限不足で必ず失敗する
            raise PermissionError("[NetworkLimiter] Need root (or --cap-add NET_ADMIN) inside container")

        if not os.path.exists(self.script_path):
            raise FileNotFoundError(f"[NetworkLimiter] tc_apply.sh not found at {self.script_path}")

        if not os.path.exists(f"/sys/class/net/{self.interface}"):
            raise RuntimeError(f"[NetworkLimiter] Interface '{self.interface}' not found in container")

    def _resolve_interface(self, iface: str) -> str:
        if iface != "auto":
            return iface
        # eth0 を最優先
        if os.path.exists("/sys/class/net/eth0"):
            return "eth0"

        # 候補一覧（lo / docker* / ifb* / veth* / br-* / tailscale* は除外）
        bad_prefix = ("lo", "docker", "ifb", "veth", "br-", "tailscale")
        nets = [
            n for n in os.listdir("/sys/class/net")
            if not any(n.startswith(p) for p in bad_prefix)
        ]
        # eth*, en* を優先
        for pref in ("eth", "en"):
            cand = [n for n in nets if n.startswith(pref)]
            if cand:
                return cand[0]
        # 残りのどれか
        return nets[0] if nets else "eth0"

    def apply_limits(self):
        if not self.enabled:
            print("[NetworkLimiter] Disabled → skip")
            return

        env = os.environ.copy()
        env["IF"]        = self.interface
        env["EGRESS"]    = str(max(0, self.egress))
        env["INGRESS"]   = str(max(0, self.ingress))
        env["DELAY_MS"]  = str(max(0, self.latency))
        env["JITTER_MS"] = str(max(0, self.jitter))
        env["LOSS_PCT"]  = str(max(0.0, self.loss))

        print(f"[NetworkLimiter] Applying via tc_apply.sh:")
        print(f"  IF={env['IF']}  EGRESS={env['EGRESS']}M  INGRESS={env['INGRESS']}M  "
              f"DELAY={env['DELAY_MS']}ms  JITTER={env['JITTER_MS']}ms  LOSS={env['LOSS_PCT']}%")

        # ここで例外を上げる方が不具合に気づきやすい
        subprocess.run(["bash", self.script_path], env=env, check=True)

    def cleanup(self):
        if not self.auto_cleanup:
            print("[NetworkLimiter] auto_cleanup=False → skip cleanup")
            return

        print(f"[NetworkLimiter] Cleanup tc on {self.interface}...")
        cmds = [
            ["tc", "qdisc", "del", "dev", self.interface, "root"],
            ["tc", "qdisc", "del", "dev", self.interface, "ingress"],
            ["ip", "link", "del", "ifb0"],
        ]
        for cmd in cmds:
            try:
                subprocess.run(cmd, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        print("[NetworkLimiter] Cleanup done.")
