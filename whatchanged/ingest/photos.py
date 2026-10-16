"""Query photos (HEIC / JPEG / PNG) -> RGB JPEGs with the EXIF orientation applied, at the work size (long side 960)
and at the SAM 3 tile size (long side 1920)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def open_oriented(path: str | Path) -> tuple[Image.Image, int]:
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    im = Image.open(path)
    orientation = int(im.getexif().get(0x0112, 1))
    return ImageOps.exif_transpose(im).convert("RGB"), orientation


def resized(im: Image.Image, long_side: int, upscale: bool) -> Image.Image:
    s = long_side / max(im.size)
    if s >= 1 and not upscale:
        return im
    return im.resize((int(round(im.size[0] * s)), int(round(im.size[1] * s))), Image.LANCZOS)


def ingest_photo(
    path: str | Path, work_dir: Path, hires_dir: Path, work_long: int, hires_long: int, quality: int
) -> dict:
    path = Path(path)
    im, orientation = open_oriented(path)
    work_dir.mkdir(parents=True, exist_ok=True)
    hires_dir.mkdir(parents=True, exist_ok=True)
    work = resized(im, work_long, upscale=False)
    work.save(work_dir / f"{path.stem}.jpg", quality=quality)
    resized(im, hires_long, upscale=True).save(hires_dir / f"{path.stem}.jpg", quality=quality)
    return dict(
        name=path.stem, src=str(path), exif_orientation=orientation, orig_size=list(im.size), size=list(work.size)
    )
