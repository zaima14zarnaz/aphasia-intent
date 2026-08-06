"""Utilities for reading ELAN (.eaf) annotation files."""

from __future__ import annotations

from pathlib import Path


def _resolve_eaf_paths(input_path: Path) -> list[Path]:
    """Return one or more .eaf files from a file path or directory."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".eaf":
            raise ValueError(f"Expected a .eaf file, got: {input_path}")
        return [input_path]
    if input_path.is_dir():
        eaf_files = sorted(input_path.glob("*.eaf"))
        if not eaf_files:
            raise FileNotFoundError(f"No .eaf files found in {input_path}")
        return eaf_files
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _read_eaf_text(eaf_path: Path) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return eaf_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return eaf_path.read_text(encoding="utf-8", errors="replace")


def _resolve_txt_path(eaf_path: Path, output_path: Path, *, allow_file: bool) -> Path:
    """
    Resolve where to write the .txt output.

    If ``output_path`` ends with ``.txt`` and is not an existing directory, treat it
    as the exact output file path (only valid when ``allow_file`` is True).
    Otherwise treat ``output_path`` as a directory and use ``{eaf_stem}.txt``.
    """
    if (
        allow_file
        and output_path.suffix.lower() == ".txt"
        and not output_path.is_dir()
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"{eaf_path.stem}.txt"


def read_eaf_to_txt(
    input_dir: str | Path,
    output_dir: str | Path,
    eaf_filename: str | None = None,
) -> Path:
    """
    Read an ELAN (.eaf) file and save its contents to a .txt file.

    Args:
        input_dir: Path to a .eaf file, or a directory containing .eaf file(s).
        output_dir: Directory for the .txt file, or an explicit ``.txt`` file path
            when reading a single .eaf file.
        eaf_filename: When ``input_dir`` is a directory, name of the .eaf file to
            read. If omitted, the directory must contain exactly one .eaf file.

    Returns:
        Path to the written .txt file.
    """
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()

    if input_path.is_file():
        eaf_path = input_path
    elif input_path.is_dir():
        if eaf_filename is None:
            eaf_files = _resolve_eaf_paths(input_path)
            if len(eaf_files) != 1:
                names = ", ".join(path.name for path in eaf_files)
                raise ValueError(
                    "input_dir contains multiple .eaf files; pass eaf_filename. "
                    f"Found: {names}"
                )
            eaf_path = eaf_files[0]
        else:
            eaf_path = input_path / eaf_filename
            if not eaf_path.is_file():
                raise FileNotFoundError(f"EAF file not found: {eaf_path}")
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    txt_path = _resolve_txt_path(eaf_path, output_path, allow_file=True)
    txt_path.write_text(_read_eaf_text(eaf_path), encoding="utf-8")
    return txt_path


def read_all_eaf_to_txt(
    input_dir: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """
    Read every .eaf file from a path and save each one as a .txt file.

    ``input_dir`` may be a directory (all ``*.eaf`` files) or a single ``.eaf`` file.
    ``output_dir`` must be a directory unless only one file is being converted.
    """
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    eaf_files = _resolve_eaf_paths(input_path)

    allow_file = len(eaf_files) == 1
    if not allow_file and output_path.suffix.lower() == ".txt":
        raise ValueError(
            "output_dir must be a directory when converting multiple .eaf files"
        )

    written: list[Path] = []
    for eaf_path in eaf_files:
        txt_path = _resolve_txt_path(eaf_path, output_path, allow_file=allow_file)
        txt_path.write_text(_read_eaf_text(eaf_path), encoding="utf-8")
        written.append(txt_path)
    return written


if __name__ == "__main__":
    txt_path = read_eaf_to_txt(
        input_dir="/data/quantization/zaima/aphasiabank/neural2/control/annotations/aphasia/rc100_day1_AB.eaf",
        output_dir="/home/z/zaimazarnaz/research1/Aphasia/src/pre_process/sample_eaf.txt",
    )
    print(f"Wrote {txt_path}")
