from pathlib import Path


def resolve_path(path: str, reference_script_location: str):
    """Return composite path from given reference location and relative path."""
    if Path(path).is_absolute():
        return Path(path).resolve()

    if Path(reference_script_location).is_dir():
        return Path(Path(reference_script_location), path).resolve()

    return Path(Path(reference_script_location).parent, path).resolve()
