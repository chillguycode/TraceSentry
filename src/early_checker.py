"""
src/early_checker.py
Stateful Gateway Early Checker & Policy Engine 
"""

from __future__ import annotations
import time
from typing import Any, Dict, Iterable, List, Optional, Set
from pydantic import BaseModel, Field

# Defensive imports supporting both local package layout and interactive notebooks
try:
    from src.graph_builder import DiamondFeatureExtractor
    from src.schema import SwarmTracesRawRecord
except ModuleNotFoundError:
    from graph_builder import DiamondFeatureExtractor
    from schema import SwarmTracesRawRecord


class TaintEntry(BaseModel):
    """
    Forensic record of a contaminated asset, tracking origin attribution and rules.
    """
    asset_id: str
    asset_type: str  # "credential" or "checksum"
    tainted_at_seq: int
    originator_row_id: str
    origin_score: float
    trigger_rules: List[str] = Field(default_factory=list)


class EarlyCheckNotice(BaseModel):
    """
    Structured security notice emitted for every evaluated swarm event.
    """
    row_id: str
    seq_index: int
    depth: int = 0
    suspicion_score: float = Field(..., ge=0.0, le=1.0)
    escalate: bool
    taint_committed: bool = False
    tripped_rules: List[str] = Field(default_factory=list)
    inherited_taint_provenance: Optional[Dict[str, Any]] = None
    mode: str = "FULL_DPI"


class GatewayEarlyChecker:
    """
    In-line wire-speed security gateway monitor with dynamic taint provenance.
    """

    ESCALATION_THRESHOLD = 0.20
    TAINT_COMMIT_THRESHOLD = 0.50

    # Static C2 Dead-Drop Markers 
    KNOWN_C2_MARKERS = ("G23619", "D64LINK", "LOOT", "OTS92", "m47push2", "[SERVICE 3 URL")

    def __init__(
        self,
        mode: str = "FULL_DPI",
        manifest_allowed: Optional[Set[str]] = None
    ) -> None:
        if mode not in ("FULL_DPI", "EGRESS_ONLY"):
            raise ValueError(f"Invalid mode '{mode}'. Choose 'FULL_DPI' or 'EGRESS_ONLY'.")

        self.mode = mode

        # Table 1: Approved Task Manifest (M_allowed)
        self.manifest_allowed: Set[str] = set(manifest_allowed) if manifest_allowed else set()

        # Table 2: Structured Taint Registry (T_taint): Asset ID -> TaintEntry
        # Holds ONLY Credentials and Code Checksums (Never Victim Server URLs)
        self.taint_registry: Dict[str, TaintEntry] = {}

        # Table 3: Unvetted Asset Exposure Counter (C_seen)
        self.asset_seen_counter: Dict[str, int] = {}

    def reset_state(self) -> None:
        """
        Clears state tables for fresh evaluation runs.
        """
        self.taint_registry.clear()
        self.asset_seen_counter.clear()

    def evaluate_event(
        self,
        record: SwarmTracesRawRecord,
        seq_index: int
    ) -> EarlyCheckNotice:

        """
        Evaluates a single event against the 3-pass collaborative anomaly scoring model
        using temporal causal bounds and provenance tracking.
        """
        w_infra = 0.0
        w_check = 0.0
        w_cap = 0.0
        tripped_rules: List[str] = []
        inherited_provenance: Optional[Dict[str, Any]] = None

        text = record.text or ""

        # Extract features reusing DiamondFeatureExtractor
        infra_items = DiamondFeatureExtractor.extract_infrastructure(text)
        cap_items = DiamondFeatureExtractor.extract_capabilities(text) if self.mode == "FULL_DPI" else []

        extracted_creds = [val for stype, val in infra_items if stype == "credential"]
        extracted_checksums = [val for stype, val in infra_items if stype == "checksum"]
        extracted_c2_boards = [val for stype, val in infra_items if stype == "board_or_c2"]

        # Network Header & Infrastructure Policy (Active in both modes)
        for cred in extracted_creds:
            cred_taint = self.taint_registry.get(cred)
            # Temporal Causal Guard: Must be tainted at an EARLIER turn
            if cred_taint is not None and seq_index > cred_taint.tainted_at_seq:
                w_infra += 0.40
                tripped_rules.append("TAINTED_CREDENTIAL_INHERITANCE")
                if not inherited_provenance:
                    inherited_provenance = {
                        "asset_type": "credential",
                        "asset_id": cred,
                        "patient_zero": cred_taint.originator_row_id,
                        "tainted_at_seq": cred_taint.tainted_at_seq,
                        "trigger": cred_taint.trigger_rules
                    }
            elif cred not in self.manifest_allowed:
                w_infra += 0.40
                tripped_rules.append("UNDECLARED_CREDENTIAL_VIOLATION")

            # Stigmergic reuse velocity check
            if self.asset_seen_counter.get(cred, 0) >= 1:
                w_infra += 0.30
                tripped_rules.append("STIGMERGIC_CREDENTIAL_REUSE")

        # C2 Dead-Drop Board Detection 
        # Victim destinations are NEVER checked against taint_registry
        is_c2_marker = False
        for board in extracted_c2_boards:
            if board in self.KNOWN_C2_MARKERS or "artifactory" in board.lower():
                is_c2_marker = True
                break

        if not is_c2_marker:
            for marker in self.KNOWN_C2_MARKERS:
                if marker in text:
                    is_c2_marker = True
                    break

        if is_c2_marker:
            w_infra += 0.20
            tripped_rules.append("TAINTED_DESTINATION_OR_C2_BOARD")

        # Cryptographic Checksum Provenance (Disabled in EGRESS_ONLY)
        if self.mode == "FULL_DPI":
            for csum in extracted_checksums:
                csum_taint = self.taint_registry.get(csum)
                # Temporal Causal Guard
                if csum_taint is not None and seq_index > csum_taint.tainted_at_seq:
                    w_check += 0.30
                    tripped_rules.append("TAINTED_CODE_EXECUTION")
                    if not inherited_provenance:
                        inherited_provenance = {
                            "asset_type": "checksum",
                            "asset_id": csum,
                            "patient_zero": csum_taint.originator_row_id,
                            "tainted_at_seq": csum_taint.tainted_at_seq,
                            "trigger": csum_taint.trigger_rules
                        }
                elif self.asset_seen_counter.get(csum, 0) >= 1:
                    w_check += 0.20
                    tripped_rules.append("SHARED_CODE_VELOCITY")

        # Exploit Capability Signatures (Disabled in EGRESS_ONLY)
        if self.mode == "FULL_DPI":
            for cap_subtype, _ in cap_items:
                if cap_subtype == "rce_injection":
                    w_cap += 0.50
                    tripped_rules.append("RCE_COMMAND_INJECTION")
                elif cap_subtype == "covert_exfil":
                    w_cap += 0.40
                    tripped_rules.append("COVERT_EXFILTRATION")
                elif cap_subtype == "defense_evasion":
                    w_cap += 0.40
                    tripped_rules.append("DEFENSE_EVASION")
                elif cap_subtype == "sync_probe":
                    w_cap += 0.20
                    tripped_rules.append("SYNCHRONOUS_XHR_PROBE")

        # COMPOSITE SCORE & TAINT COMMITMENT
        composite_score = min(1.0, round(w_infra + w_check + w_cap, 4))
        escalate = composite_score >= self.ESCALATION_THRESHOLD
        taint_committed = False

        # Taint Commitment Trigger (Score >= 0.50)
        # Generic destinations are NEVER committed to prevent Taint Explosion.
        if composite_score >= self.TAINT_COMMIT_THRESHOLD:
            taint_committed = True

            # 1. Commit Stolen Credentials
            for cred in extracted_creds:
                if cred not in self.taint_registry:
                    self.taint_registry[cred] = TaintEntry(
                        asset_id=cred,
                        asset_type="credential",
                        tainted_at_seq=seq_index,
                        originator_row_id=record.id,
                        origin_score=composite_score,
                        trigger_rules=list(tripped_rules)
                    )

            # 2. Commit Malicious Code Artifacts (Checksums)
            for csum in extracted_checksums:
                if csum not in self.taint_registry:
                    self.taint_registry[csum] = TaintEntry(
                        asset_id=csum,
                        asset_type="checksum",
                        tainted_at_seq=seq_index,
                        originator_row_id=record.id,
                        origin_score=composite_score,
                        trigger_rules=list(tripped_rules)
                    )

        # Update Exposure Counter (strictly after scoring to prevent turn lookahead)
        for asset in (extracted_creds + extracted_checksums):
            self.asset_seen_counter[asset] = self.asset_seen_counter.get(asset, 0) + 1

        return EarlyCheckNotice(
            row_id=record.id,
            seq_index=seq_index,
            depth=record.depth,
            suspicion_score=composite_score,
            escalate=escalate,
            taint_committed=taint_committed,
            tripped_rules=tripped_rules,
            inherited_taint_provenance=inherited_provenance,
            mode=self.mode
        )

    def evaluate_stream(
        self,
        records: Iterable[SwarmTracesRawRecord]
    ) -> Dict[str, Any]:
        """
        Streams records through the early checker and returns benchmark metrics.
        """
        self.reset_state()

        total_processed = 0
        fast_path_passed = 0
        escalated_count = 0
        escalated_notices: List[EarlyCheckNotice] = []
        first_alert_seq_index: Optional[int] = None
        patient_zero_score = 0.0

        t_start = time.perf_counter()

        for idx, record in enumerate(records):
            seq_index = DiamondFeatureExtractor.extract_seq_index(record.id, fallback_step=idx)
            notice = self.evaluate_event(record, seq_index=seq_index)
            total_processed += 1

            if record.id == "R0000001":
                patient_zero_score = notice.suspicion_score

            if notice.escalate:
                escalated_count += 1
                escalated_notices.append(notice)
                if first_alert_seq_index is None:
                    first_alert_seq_index = seq_index
            else:
                fast_path_passed += 1

        t_elapsed = time.perf_counter() - t_start
        avg_latency_ms = (t_elapsed / total_processed * 1000.0) if total_processed > 0 else 0.0
        reduction_ratio = (fast_path_passed / total_processed * 100.0) if total_processed > 0 else 0.0

        return {
            "mode": self.mode,
            "total_processed": total_processed,
            "fast_path_passed": fast_path_passed,
            "escalated_count": escalated_count,
            "filter_reduction_ratio": round(reduction_ratio, 2),
            "patient_zero_r0000001_score": patient_zero_score,
            "first_alert_seq_index": first_alert_seq_index,
            "total_tainted_assets": len(self.taint_registry),
            "tainted_registry": {k: v.model_dump() for k, v in self.taint_registry.items()},
            "avg_latency_ms": round(avg_latency_ms, 5),
            "total_time_seconds": round(t_elapsed, 3),
            "sample_escalations": [n.model_dump() for n in escalated_notices[:5]]
        }
