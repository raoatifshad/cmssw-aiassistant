"""Read CMSSDT logAnalysis artifacts from a trusted local archive."""
import pickle
from pathlib import Path


class _AnalyzerUnpickler(pickle.Unpickler):
    _classes: dict[str, type] = {}

    def find_class(self, module: str, name: str):
        if module == "__main__" and name in {"PackageInfo", "ErrorInfo"}:
            return self._classes.setdefault(name, type(name, (), {}))
        raise pickle.UnpicklingError(f"unsupported analyzer record type: {module}.{name}")


def read_build_summary(root: Path, build_tag: str) -> dict | None:
    """Return one platform summary for a build, or None when no artifact exists."""
    matches = sorted(root.glob(f"{build_tag}/*/build-logs/logAnalysis.pkl"))
    if not matches:
        return None
    artifact = matches[0]
    with artifact.open("rb") as stream:
        unpickler = _AnalyzerUnpickler(stream)
        metadata = unpickler.load()
        categories = unpickler.load()
        totals = unpickler.load()
        packages_by_type = unpickler.load()
        analyzed_files = unpickler.load()

    if not (
        isinstance(metadata, list)
        and len(metadata) == 3
        and isinstance(categories, list)
        and isinstance(totals, dict)
        and isinstance(packages_by_type, dict)
        and isinstance(analyzed_files, list)
    ):
        raise ValueError(f"unsupported logAnalysis format: {artifact}")

    return {
        "build": metadata[0],
        "platform": metadata[1],
        "elapsed_seconds": metadata[2],
        "files_analyzed": len(analyzed_files),
        "categories": [
            {
                "type": category,
                "packages": len(packages_by_type.get(category, [])),
                "total": totals.get(category, 0),
            }
            for category in categories
        ],
    }