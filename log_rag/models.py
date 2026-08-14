"""Canonical event model and safe normalization helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

_SECRET_RE = re.compile(r"(?i)(?:token|password|secret|api[_-]?key)\s*[=:]\s*([^\s,;]+)")
REDACTED_KEYS = {"token", "password", "secret", "api_key", "apikey", "authorization"}


def _utc_timestamp(value: Any) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).isoformat()
    except ValueError:
        return text


def redact(value: Any) -> Any:
    """Recursively redact common credential fields before persistence/indexing."""
    if isinstance(value, dict):
        return {key: "[REDACTED]" if key.lower() in REDACTED_KEYS else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET_RE.sub(lambda match: match.group(0).split(match.group(1))[0] + "[REDACTED]", value)
    return value


@dataclass(frozen=True)
class LogEvent:
    event_id: str
    timestamp: str
    source: str
    message: str
    severity: str
    service: str | None
    host: str | None
    build: str | None
    metadata: dict[str, Any]
    source_hash: str

    @classmethod
    def from_record(cls, record: dict[str, Any], *, source: str, default_build: str | None = None) -> "LogEvent":
        clean = redact(record)
        message = str(clean.get("message") or clean.get("msg") or clean.get("text") or "")
        timestamp = _utc_timestamp(clean.get("timestamp") or clean.get("@timestamp") or clean.get("time"))
        severity = str(clean.get("severity") or clean.get("level") or clean.get("log.level") or "INFO").upper()
        service = clean.get("service") or clean.get("service_name")
        if isinstance(service, dict):
            service = service.get("name")
        host = clean.get("host") or clean.get("hostname")
        if isinstance(host, dict):
            host = host.get("name")
        build = clean.get("build") or clean.get("ib_tag") or default_build
        metadata = {key: value for key, value in clean.items() if key not in {"message", "msg", "text", "timestamp", "@timestamp", "time", "severity", "level", "log.level", "service", "service_name", "host", "hostname", "build", "ib_tag"}}
        canonical = json.dumps({"timestamp": timestamp, "source": source, "message": message, "severity": severity, "service": service, "host": host, "build": build, "metadata": metadata}, sort_keys=True, default=str, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return cls(event_id=digest[:24], timestamp=timestamp, source=source, message=message, severity=severity, service=str(service) if service is not None else None, host=str(host) if host is not None else None, build=str(build) if build is not None else None, metadata=metadata, source_hash=digest)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def evidence_text(self) -> str:
        context = " ".join(f"{key}={value}" for key, value in self.metadata.items() if isinstance(value, (str, int, float, bool)))
        return f"[{self.timestamp}] {self.severity} source={self.source} service={self.service or '-'} host={self.host or '-'} build={self.build or '-'}\n{self.message}\n{context}".strip()
