"""Utilities for reading CHAT (.cha) transcript files."""

from __future__ import annotations

from pathlib import Path


def _resolve_cha_paths(input_path: Path) -> list[Path]:
    """Return one or more .cha files from a file path or directory."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".cha":
            raise ValueError(f"Expected a .cha file, got: {input_path}")
        return [input_path]
    if input_path.is_dir():
        cha_files = sorted(input_path.glob("*.cha"))
        if not cha_files:
            raise FileNotFoundError(f"No .cha files found in {input_path}")
        return cha_files
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _read_cha_text(cha_path: Path) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return cha_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return cha_path.read_text(encoding="utf-8", errors="replace")


def _resolve_txt_path(cha_path: Path, output_path: Path, *, allow_file: bool) -> Path:
    """
    Resolve where to write the .txt output.

    If ``output_path`` ends with ``.txt`` and is not an existing directory, treat it
    as the exact output file path (only valid when ``allow_file`` is True).
    Otherwise treat ``output_path`` as a directory and use ``{cha_stem}.txt``.
    """
    if (
        allow_file
        and output_path.suffix.lower() == ".txt"
        and not output_path.is_dir()
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"{cha_path.stem}.txt"


def read_cha_to_txt(
    input_dir: str | Path,
    output_dir: str | Path,
    cha_filename: str | None = None,
) -> Path:
    """
    Read a CHAT (.cha) file and save its contents to a .txt file.

    Args:
        input_dir: Path to a .cha file, or a directory containing .cha file(s).
        output_dir: Directory for the .txt file, or an explicit ``.txt`` file path
            when reading a single .cha file.
        cha_filename: When ``input_dir`` is a directory, name of the .cha file to
            read. If omitted, the directory must contain exactly one .cha file.

    Returns:
        Path to the written .txt file.
    """
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()

    if input_path.is_file():
        cha_path = input_path
    elif input_path.is_dir():
        if cha_filename is None:
            cha_files = _resolve_cha_paths(input_path)
            if len(cha_files) != 1:
                names = ", ".join(path.name for path in cha_files)
                raise ValueError(
                    "input_dir contains multiple .cha files; pass cha_filename. "
                    f"Found: {names}"
                )
            cha_path = cha_files[0]
        else:
            cha_path = input_path / cha_filename
            if not cha_path.is_file():
                raise FileNotFoundError(f"CHA file not found: {cha_path}")
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    txt_path = _resolve_txt_path(cha_path, output_path, allow_file=True)
    txt_path.write_text(_read_cha_text(cha_path), encoding="utf-8")
    return txt_path


def read_all_cha_to_txt(
    input_dir: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """
    Read every .cha file from a path and save each one as a .txt file.

    ``input_dir`` may be a directory (all ``*.cha`` files) or a single ``.cha`` file.
    ``output_dir`` must be a directory unless only one file is being converted.
    """
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    cha_files = _resolve_cha_paths(input_path)

    allow_file = len(cha_files) == 1
    if not allow_file and output_path.suffix.lower() == ".txt":
        raise ValueError(
            "output_dir must be a directory when converting multiple .cha files"
        )

    written: list[Path] = []
    for cha_path in cha_files:
        txt_path = _resolve_txt_path(cha_path, output_path, allow_file=allow_file)
        txt_path.write_text(_read_cha_text(cha_path), encoding="utf-8")
        written.append(txt_path)
    return written


if __name__ == "__main__":
    txt_path = read_cha_to_txt(
        input_dir="/data/quantization/zaima/aphasiabank/neural2/control/annotations/aphasia/rc101_day1_AB.elan.cha",
        output_dir="/home/z/zaimazarnaz/research1/Aphasia/src/pre_process/sample_cha.txt",
    )
    print(f"Wrote {txt_path}")
