from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Generator, List, Optional, Union

from .schema import (
    SwarmTracesRawRecord,
    ExtractedAttackArtifact,
    NormalizedForensicEvent,
)


class SwarmTracesParser:
    """
    Forensic parser that processes raw SwarmTraces records line-by-line,
    extracting execution telemetry, network destinations, and exploit payloads.
    """

    # Regex patterns for high-speed entity extraction directly from JavaScript payloads
    RE_AGENT_ID = re.compile(r'["\']agent_id["\']\s*:\s*["\']([^"\']+)["\']')
    RE_TASK_ID = re.compile(r'["\']task_id["\']\s*:\s*["\']([^"\']+)["\']')
    RE_FILE_PATH = re.compile(r'["\']file_path["\']\s*:\s*["\']([^"\']+)["\']')
    RE_DEST_URL = re.compile(r'https?://[^\s"\'`,]+')
    RE_SERVICE_TAG = re.compile(r'\[SERVICE\s+\d+\s+URL\s+\d+\]')
    RE_SHELL_INJECTION = re.compile(r'[;&|`]\s*(id|uname|cat|ls|whoami|curl|wget|python|sh|bash)\b')

    @classmethod
    def extract_artifact(cls, raw_text: str) -> ExtractedAttackArtifact:
        """
        Parses the raw JavaScript/payload text to extract inner parameters
        and detect command injection attempts.
        """
        if not raw_text:
            return ExtractedAttackArtifact()

        # 1. Extract Agent and Task Identifiers
        agent_match = cls.RE_AGENT_ID.search(raw_text)
        agent_id = agent_match.group(1) if agent_match else None

        task_match = cls.RE_TASK_ID.search(raw_text)
        task_id = task_match.group(1) if task_match else None

        # 2. Extract Destination URL / Service Name
        url_match = cls.RE_DEST_URL.search(raw_text)
        service_match = cls.RE_SERVICE_TAG.search(raw_text)
        
        destination = None
        if url_match and service_match:
            destination = f"{url_match.group(0)}{service_match.group(0)}"
        elif url_match:
            destination = url_match.group(0)
        elif service_match:
            destination = service_match.group(0)

        # 3. Extract File Path and Detect Command Injections
        file_path_match = cls.RE_FILE_PATH.search(raw_text)
        file_path = file_path_match.group(1) if file_path_match else None

        injected_cmds: List[str] = []
        has_injection = False

        if file_path and (";" in file_path or "|" in file_path or "&" in file_path):
            has_injection = True
            # Extract distinct commands separated by semicolons or pipes
            parts = re.split(r'[;&|]', file_path)
            for part in parts:
                clean_cmd = part.strip().rstrip("#").strip()
                if clean_cmd and not clean_cmd.endswith(".c") and not clean_cmd.endswith(".h"):
                    injected_cmds.append(clean_cmd)

        # Catch-all check across raw text if shell injection wasn't in file_path
        if not has_injection and cls.RE_SHELL_INJECTION.search(raw_text):
            has_injection = True

        return ExtractedAttackArtifact(
            agent_id=agent_id,
            task_id=task_id,
            destination_url=destination,
            file_path=file_path,
            injected_commands=injected_cmds,
            has_command_injection=has_injection
        )

    @classmethod
    def normalize_event(
        cls,
        raw_record: SwarmTracesRawRecord,
        step_index: int
    ) -> NormalizedForensicEvent:
        """
        Transforms a validated raw record into a normalized graph-ready event.
        """
        artifact = cls.extract_artifact(raw_record.text)

        # Determine Sender (Agent or upstream record ID)
        sender = artifact.agent_id if artifact.agent_id else raw_record.id

        # Determine Receiver (Target Service, child record, or shared environment)
        if artifact.destination_url:
            receiver = artifact.destination_url
        elif raw_record.source_rows:
            receiver = raw_record.source_rows[0]
        else:
            receiver = "unknown_endpoint"

        # Determine Action Category
        if artifact.has_command_injection:
            action = "command_injection"
        elif "fetch" in raw_record.text or "XMLHttpRequest" in raw_record.text:
            action = "http_egress"
        else:
            action = raw_record.kind

        return NormalizedForensicEvent(
            step=step_index,
            event_id=raw_record.id,
            timestamp=raw_record.time_utc or f"step_{step_index}",
            sender_id=sender,
            receiver_id=receiver,
            action_type=action,
            artifact=artifact,
            raw_content=raw_record.text[:200]  # Store concise preview
        )

    @classmethod
    def stream_file(
        cls,
        file_path: Union[str, Path],
        limit: Optional[int] = None
    ) -> Generator[NormalizedForensicEvent, None, None]:
        """
        Memory-safe generator that yields normalized events line-by-line.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path.resolve()}")

        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for step_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                raw_dict = json.loads(line)
                raw_record = SwarmTracesRawRecord(**raw_dict)
                normalized = cls.normalize_event(raw_record, step_index=step_idx)

                yield normalized

                count += 1
                if limit is not None and count >= limit:
                    break
