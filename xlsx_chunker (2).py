#!/usr/bin/env python3
"""
xlsx_chunker.py
---------------
Chunk .xlsx/.xlsm files into token-sized JSONL lines with token-based overlap.
- Supports single file → single output
- Or folder → folder (processes all .xlsx/.xlsm)
- Sliding-window overlap is computed in *rows* to approximate the desired token overlap.

Each JSONL line:
  {
    "id": "sha256:...",
    "text": "... LLM-ready slice ...",
    "meta": { ... rich metadata, including overlap info ... },
    "structured": { "headers": [...], "rows": [[...], ...], "types": [...] },
    "token_count": 438
  }

Usage:
  # Single file → explicit output file
  python xlsx_chunker.py input.xlsx output.jsonl --target-tokens 500 --overlap-tokens 125 --llm-temperature 0.65

  # Folder → Folder (creates one .jsonl per Excel, same base name)
  python xlsx_chunker.py /path/to/input_dir /path/to/output_dir --target-tokens 500 --overlap-tokens 125

Dependencies:
  pip install pandas openpyxl numpy
  # Optional (for accurate token counts):
  pip install tiktoken
"""

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- Token counting -----------------------------
def count_tokens(text: str) -> int:
    """
    Use tiktoken if available; else 4-characters-per-token heuristic.
    """
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, math.ceil(len(text) / 4))


# ----------------------------- Helpers -----------------------------
def normalize_headers(cols) -> List[str]:
    """
    Flatten pandas headers, including MultiIndex. Replace NaNs with empty strings.
    """
    if isinstance(cols, pd.MultiIndex):
        return [" | ".join([str(x) for x in tup if pd.notna(x)]) for tup in cols.tolist()]
    out = []
    for c in cols:
        if pd.isna(c):
            out.append("")
        else:
            out.append(str(c))
    return out


def header_line(headers: List[str]) -> str:
    return "Headers: " + " | ".join(headers)


def render_rows_to_text(headers: List[str], rows: List[List[object]]) -> str:
    """
    Convert table rows to compact, readable text for LLMs.
    """
    lines = [header_line(headers)]
    for r in rows:
        line_vals = []
        for x in r:
            if x is None:
                line_vals.append("")
            elif isinstance(x, float) and np.isnan(x):
                line_vals.append("")
            else:
                line_vals.append(str(x))
        lines.append("- " + " | ".join(line_vals))
    return "\n".join(lines)


@dataclass
class ChunkConfig:
    target_tokens: int = 500
    overlap_tokens: int = 125
    safety_margin: int = 50
    min_rows: int = 20
    max_rows: int = 200
    llm_temperature: float = 0.65


def estimate_tokens_per_row(headers: List[str], sample_rows: List[List[object]]) -> float:
    """
    Estimate tokens per *data row* by subtracting header tokens.
    """
    hdr = header_line(headers)
    hdr_tok = count_tokens(hdr)
    sample_text = hdr + "\n" + "\n".join(
        "- " + " | ".join("" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x) for x in row)
        for row in sample_rows
    )
    total_tok = count_tokens(sample_text)
    row_tok = max(1.0, (total_tok - hdr_tok) / max(1, len(sample_rows)))
    return row_tok


def chunk_table(
    df: pd.DataFrame,
    source_path: str,
    sheet_name: str,
    table_name: Optional[str],
    has_macros: bool,
    modified_time: Optional[str],
    cfg: ChunkConfig,
) -> Iterable[dict]:
    """
    Yield JSONL-ready records for the given DataFrame, chunked by rows to fit token budget, with overlap.
    """
    # Drop fully-empty rows/cols quickly
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return

    headers = normalize_headers(df.columns)
    values = df.values.tolist()
    n = len(values)

    # --- Estimate rows per chunk & overlap rows from token targets ---
    sample_k = min(max(cfg.min_rows, 50), n)
    sample_rows = values[:sample_k]
    per_row_tok = estimate_tokens_per_row(headers, sample_rows)

    # Deduct an approximate header+prefix cost (~ header + 'Sheet: ...' line)
    prefix_text = f"Sheet: {sheet_name}" + (f" | Table: {table_name}" if table_name else "")
    prefix_tok = count_tokens(prefix_text) + count_tokens(header_line(headers)) + 2  # small buffer

    # Compute rows per chunk to meet target tokens
    rows_per_chunk = int(max(cfg.min_rows, min(cfg.max_rows, math.floor((cfg.target_tokens - prefix_tok) / max(1.0, per_row_tok)))))
    if rows_per_chunk < cfg.min_rows:
        rows_per_chunk = cfg.min_rows

    # Compute overlap in rows from desired token overlap
    overlap_rows = int(max(1, round(cfg.overlap_tokens / max(1.0, per_row_tok))))
    # Ensure progress (avoid infinite loop)
    if overlap_rows >= rows_per_chunk:
        overlap_rows = max(1, rows_per_chunk - 1)

    start = 0
    chunk_idx = 0
    while start < n:
        end = min(n, start + rows_per_chunk)

        # Build text block
        header = f"Sheet: {sheet_name}" + (f" | Table: {table_name}" if table_name else "")
        text_block = header + "\n" + render_rows_to_text(headers, values[start:end])
        tok = count_tokens(text_block)

        # If overshoot by a lot, shrink once
        if tok > (cfg.target_tokens + cfg.safety_margin) and (end - start) > cfg.min_rows:
            # recompute rows using token ratio
            new_span = max(cfg.min_rows, int((cfg.target_tokens - prefix_tok) / max(1.0, per_row_tok)))
            end = min(n, start + new_span)
            text_block = header + "\n" + render_rows_to_text(headers, values[start:end])
            tok = count_tokens(text_block)

        meta = {
            "source_path": source_path,
            "sheet": sheet_name,
            "table_name": table_name,
            "row_range": [start + 1, end],  # 1-based inclusive
            "header_rows": 1,
            "col_range": [1, len(headers)],
            "has_macros": has_macros,
            "modified_time": modified_time,
            "target_tokens": cfg.target_tokens,
            "overlap_tokens": cfg.overlap_tokens,
            "approx_rows_per_chunk": rows_per_chunk,
            "approx_overlap_rows": overlap_rows,
            "chunk_index": chunk_idx,
            "llm_temperature": cfg.llm_temperature,
        }
        structured = {
            "headers": headers,
            "rows": values[start:end],
            "types": [str(df.dtypes[c]) for c in df.columns],
        }

        # Stable ID over text + metadata
        h = hashlib.sha256((text_block + json.dumps(meta, sort_keys=True)).encode("utf-8")).hexdigest()
        yield {
            "id": f"sha256:{h}",
            "text": text_block,
            "meta": meta,
            "structured": structured,
            "token_count": tok,
        }

        chunk_idx += 1
        if end >= n:
            break
        # Slide forward with overlap
        start = max(0, end - overlap_rows)


def read_excel_as_sheets(path: str) -> List[Tuple[str, Optional[str], pd.DataFrame]]:
    """
    Return list of (sheet_name, table_name, dataframe).
    Currently treats each non-empty sheet as a single table.
    You can enhance this by splitting on long gaps of empty rows.
    """
    xls = pd.ExcelFile(path, engine="openpyxl")
    out = []
    for sheet in xls.sheet_names:
        # Let pandas infer header from first row; adjust if multi-row header is expected.
        df = xls.parse(sheet, header=0)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        out.append((sheet, None, df))
    return out


def xlsx_to_jsonl(
    xlsx_path: str,
    out_jsonl_path: str,
    target_tokens: int = 500,
    overlap_tokens: int = 125,
    safety_margin: int = 50,
    min_rows: int = 20,
    max_rows: int = 200,
    llm_temperature: float = 0.65,
):
    p = Path(xlsx_path)
    modified_time = datetime.utcfromtimestamp(p.stat().st_mtime).isoformat() + "Z"
    has_macros = p.suffix.lower() == ".xlsm"
    cfg = ChunkConfig(
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        safety_margin=safety_margin,
        min_rows=min_rows,
        max_rows=max_rows,
        llm_temperature=llm_temperature,
    )

    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for sheet, table_name, df in read_excel_as_sheets(xlsx_path):
            for rec in chunk_table(
                df=df,
                source_path=str(p),
                sheet_name=sheet,
                table_name=table_name,
                has_macros=has_macros,
                modified_time=modified_time,
                cfg=cfg,
            ):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def is_excel_file(p: Path) -> bool:
    return p.suffix.lower() in {".xlsx", ".xlsm"}


def process_input_output(
    in_path: Path,
    out_path: Path,
    target_tokens: int,
    overlap_tokens: int,
    safety_margin: int,
    min_rows: int,
    max_rows: int,
    llm_temperature: float,
):
    """
    Route: file→file or folder→folder. Creates output folder when needed.
    """
    if in_path.is_file():
        if not is_excel_file(in_path):
            raise ValueError(f"Input file is not .xlsx/.xlsm: {in_path}")
        # If output is a directory, derive filename; else use as provided
        if out_path.is_dir():
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / (in_path.stem + ".jsonl")
        else:
            out_file = out_path
        print(f"[1/1] Processing {in_path} → {out_file}")
        xlsx_to_jsonl(
            xlsx_path=str(in_path),
            out_jsonl_path=str(out_file),
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            safety_margin=safety_margin,
            min_rows=min_rows,
            max_rows=max_rows,
            llm_temperature=llm_temperature,
        )
    else:
        # Folder → Folder
        out_path.mkdir(parents=True, exist_ok=True)
        excel_files = sorted([p for p in in_path.iterdir() if p.is_file() and is_excel_file(p)])
        if not excel_files:
            print(f"No .xlsx/.xlsm files found in {in_path}")
            return
        total = len(excel_files)
        for i, f in enumerate(excel_files, 1):
            out_file = out_path / (f.stem + ".jsonl")
            print(f"[{i}/{total}] Processing {f} → {out_file}")
            xlsx_to_jsonl(
                xlsx_path=str(f),
                out_jsonl_path=str(out_file),
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
                safety_margin=safety_margin,
                min_rows=min_rows,
                max_rows=max_rows,
                llm_temperature=llm_temperature,
            )


def main():
    parser = argparse.ArgumentParser(description="Chunk .xlsx/.xlsm (file or folder) into token-sized JSONL with overlap.")
    parser.add_argument("input", help="Path to input .xlsx/.xlsm file OR a folder containing such files")
    parser.add_argument("output", help="Path to output .jsonl file OR a folder to place outputs")
    parser.add_argument("--target-tokens", type=int, default=500, help="Target tokens per chunk (default: 500)")
    parser.add_argument("--overlap-tokens", type=int, default=125, help="Desired token overlap between chunks (default: 125)")
    parser.add_argument("--safety-margin", type=int, default=50, help="Allowed overshoot tokens (default: 50)")
    parser.add_argument("--min-rows", type=int, default=20, help="Minimum rows per chunk (default: 20)")
    parser.add_argument("--max-rows", type=int, default=200, help="Maximum rows per chunk (default: 200)")
    parser.add_argument("--llm-temperature", type=float, default=0.65, help="Store the generation temperature in metadata (default: 0.65)")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    process_input_output(
        in_path=in_path,
        out_path=out_path,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        safety_margin=args.safety_margin,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        llm_temperature=args.llm_temperature,
    )


if __name__ == "__main__":
    main()
