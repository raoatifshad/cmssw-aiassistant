"""Read-only lookups in a local CMSSW git checkout, so what an LLM says about the code can be checked against the code (user, 2026-11-05:
"don't blindly trust the results, compare it"). Same trust model as cmsrepo and shifter-wiki: a local clone, never fetched while answering.
Every function returns None when the checkout can't be read, so a caller can say "not checked" rather than "not found"."""
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

SOURCE_DIR = os.environ.get("CMSSW_SOURCE_DIR", "/home/rao/CHAT-BOT-AI/cmssw")
_MAX_FILES_FOR_A_SPECIFIC_NAME = 40
_MAX_NAMES = 16


def _git(*args: str, timeout: float = 8) -> str | None:
    """stdout, or None if git failed. `git grep` exits 1 for "no match": that is "" here, not a failure."""
    try:
        done = subprocess.run(["git", "-c", "safe.directory=*", "-C", SOURCE_DIR, *args], capture_output=True, text=True,
                              errors="replace", timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode == 0:
        return done.stdout
    return "" if done.returncode == 1 and args[0] == "grep" and not done.stderr.strip() else None


def source_version() -> dict | None:
    """{"commit", "date"} of the checkout: answers say how old the code they checked against is."""
    out = _git("log", "-1", "--format=%h %cs")
    if not out or " " not in out.strip():
        return None
    commit, date = out.strip().split(" ", 1)
    return {"commit": commit, "date": date}


def files_with(name: str) -> list[str] | None:
    """Files containing this name as a whole word."""
    out = _git("grep", "-l", "-F", "-w", "-e", name, "HEAD")
    if out is None:
        return None
    return [line.split(":", 1)[1] for line in out.splitlines() if ":" in line]


def path_exists(path: str) -> bool | None:
    out = _git("ls-tree", "--name-only", "HEAD", path.rstrip("/"))
    return None if out is None else bool(out.strip())


_PATH_RE = re.compile(r"(?<![\w/.:-])((?:[A-Z][A-Za-z0-9]*/)[A-Za-z0-9_]+(?:/[A-Za-z0-9_.-]+)*)")
_TOOL_RE = re.compile(r"(?<![\w/.-])((?:edm|cms)[A-Z][A-Za-z0-9]+|[a-z][A-Za-z0-9]*[A-Z]\w*\.py)\b")
_CAMEL_RE = re.compile(r"(?<![\w/.-])([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+|[A-Z]{2,}[a-z]+[A-Z]\w*)\b")
_SCOPED_RE = re.compile(r"\b(?:[A-Za-z_]\w*::)+([A-Za-z_]\w+)")
_BACKTICK_RE = re.compile(r"`([A-Za-z_][\w.:/-]{3,})`")
_SNAKE_RE = re.compile(r"(?<![\w/.-])([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+){2,})\b")
# not CMSSW names: an LLM may say them, and the code doesn't have to
_NOT_CODE = {"GitHub", "GitLab", "PullRequest", "RelVal", "RelVals", "OpenMP", "PyTorch", "TensorFlow", "JavaScript", "McM", "MadGraph"}


def code_names(text: str) -> dict[str, list[str]]:
    """What in a text claims to be code: {"paths": [...], "names": [...]}. Paths keep their whole form, a scoped name counts by its last part."""
    text = text or ""
    paths = list(dict.fromkeys(_PATH_RE.findall(text)))
    names: list[str] = []
    for regex in (_TOOL_RE, _SCOPED_RE, _CAMEL_RE, _SNAKE_RE):
        names += regex.findall(text)
    for token in _BACKTICK_RE.findall(text):
        if "/" in token:
            if token[0].isupper() and token not in paths:
                paths.append(token)
        else:
            names.append(token.rsplit("::", 1)[-1])
    in_paths = {part for path in paths for part in path.replace(".", "/").split("/")}
    names = [n for n in dict.fromkeys(names) if n not in _NOT_CODE and n not in in_paths and len(n) >= 5]
    return {"paths": paths, "names": names}


def check_names(text: str, known_text: str = "") -> dict | None:
    """Which names in an LLM's text exist. "missing": in neither the source nor known_text (the failure and the evidence it was given); an
    external package's name is in known_text when it matters, so this is what got made up, or what the code doesn't have. None: can't read the checkout."""
    if source_version() is None:
        return None
    found = code_names(text)
    known = (known_text or "").lower()
    paths = [p for p in found["paths"] if p.lower() not in known][:_MAX_NAMES]
    names = [n for n in found["names"] if n.lower() not in known][:_MAX_NAMES]
    with ThreadPoolExecutor(max_workers=8) as pool:
        path_results = list(pool.map(path_exists, paths))
        name_results = list(pool.map(files_with, names))
    # "Herwig/ThePEG" isn't a path: only a path under a real top-level directory claims to be one in this repository
    missing = [p for p, ok in zip(paths, path_results) if ok is False and path_exists(p.split("/", 1)[0])]
    missing += [n for n, hits in zip(names, name_results) if hits == []]
    in_source = {n: hits[:3] for n, hits in zip(names, name_results) if hits}
    in_source.update({p: [p] for p, ok in zip(paths, path_results) if ok})
    return {"missing": missing, "in_source": in_source, "from_context": len(found["paths"]) + len(found["names"]) - len(paths) - len(names)}


def _rank_files(name: str, files: list[str]) -> list[str]:
    def score(path: str) -> tuple:
        base = path.rsplit("/", 1)[-1]
        return (name.lower() not in base.lower(), "/test/" in path, not base.endswith((".cc", ".h", ".py", ".xml", ".cpp")), len(path))
    return sorted(files, key=score)


def source_excerpts(text: str, names_limit: int = 5, chars: int = 1800) -> list[dict] | None:
    """The code around the specific names in a failure, for the LLM to read: {"id": path, "text": ...}. A name in too many files says nothing."""
    if source_version() is None:
        return None
    found = code_names(text)
    names = sorted(found["names"], key=len, reverse=True)[:_MAX_NAMES]
    with ThreadPoolExecutor(max_workers=8) as pool:
        hits = list(pool.map(files_with, names))
    chosen = [(n, h) for n, h in zip(names, hits) if h and len(h) <= _MAX_FILES_FOR_A_SPECIFIC_NAME]
    # fewest files first: the most specific names
    chosen.sort(key=lambda pair: len(pair[1]))
    out: dict[str, dict] = {}
    for name, files in chosen[:names_limit]:
        for path in _rank_files(name, files)[:2]:
            if path in out:
                continue
            grep = _git("grep", "-n", "-F", "-w", "-C", "10", "-e", name, "HEAD", "--", path)
            if grep:
                body = "\n".join(line.split(":", 1)[1] if line.startswith("HEAD:") else line for line in grep.splitlines())
                out[path] = {"id": path, "name": name, "text": body[:chars]}
            if "/plugins/" in path:
                build = path.split("/plugins/")[0] + "/plugins/BuildFile.xml"
                content = _git("show", f"HEAD:{build}")
                if content and build not in out:
                    out[build] = {"id": build, "name": name, "text": content[:chars]}
    return list(out.values())
