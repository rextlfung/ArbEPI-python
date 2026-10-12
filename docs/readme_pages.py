"""Fill documentation pages from the repo's own Markdown files.

A page in docs/ may contain marker lines, each replaced at build time by text
read from a repo file, so the site's prose is never a second copy:

    <!-- readme: ge/README.md -->                       the whole file
    <!-- section: README.md :: Getting started -->      one section
    <!-- section: CLAUDE.md :: Data flow :: 2 -->       ... with its heading at level 2
    <!-- section: README.md :: @intro -->               text before the first `## `

A section runs from its heading (matched by prefix, at any level) to the next
heading of the same or higher level, and its headings are shifted so the first
one sits at the given level (default 1, i.e. the page title). `{include}`
would resolve the files' relative links against docs/ instead of the file's
own directory, which is why this is a hook (conf.py's `source-read`).

Links in the pulled text are rewritten: to another pulled file or section ->
its page, to any other path in the repo -> GitHub, to a sibling checkout
(`../ArbEPI`) -> plain text.
"""

import posixpath
import re
from pathlib import Path

GITHUB = "https://github.com/rextlfung/ArbEPI-python/blob/main/"
# Page a link to a file lands on when it names no (known) heading.
DEFAULT_PAGE = {"README.md": "overview"}

_MARKER = re.compile(r"^<!-- (readme|section): (.+?) -->\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_PICTURE = re.compile(r"<picture>.*?</picture>", re.S)
_ALT = re.compile(r'<img alt="([^"]*)"')
_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")


def slug(title: str) -> str:
    """GitHub's heading anchor: lowercase, drop punctuation, spaces -> hyphens."""
    return re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")


def _headings(lines):
    """(index, level, title) of every heading outside a code fence."""
    out, fenced = [], False
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced and (m := _HEADING.match(line)):
            out.append((i, len(m.group(1)), m.group(2)))
    return out


def extract(text: str, heading: str | None, level: int = 1) -> list[str]:
    """Lines of one section (or the whole file / the intro), headings shifted so
    the first is at `level`."""
    lines = text.splitlines()
    heads = _headings(lines)
    if heading is None:  # whole file: the title stays where it is
        return lines
    if heading == "@intro":
        end = next((i for i, lv, _ in heads if lv == 2), len(lines))
        lines, shift = lines[:end], level - 1
    else:
        try:
            start, lv, _ = next(h for h in heads if h[2].startswith(heading))
        except StopIteration:
            raise ValueError(f"no heading starting {heading!r}") from None
        end = next((i for i, l2, _ in heads if i > start and l2 <= lv), len(lines))
        lines, shift = lines[start:end], level - lv
    out = list(lines)
    for i, lv, _ in _headings(out):
        out[i] = "#" * min(6, max(1, lv + shift)) + out[i][lv:]
    return out


def _parse(marker: re.Match):
    kind, rest = marker.groups()
    if kind == "readme":
        return rest.strip(), None, 1
    parts = [p.strip() for p in rest.split("::")]
    return parts[0], parts[1], int(parts[2]) if len(parts) > 2 else 1


class Pages:
    """Scans the wrapper pages once, to know which page each heading lands on."""

    def __init__(self, root: Path, docs: Path):
        self.root = root
        self.page_of: dict[tuple[str, str], str] = {}
        self.default: dict[str, str] = dict(DEFAULT_PAGE)
        for md in sorted(docs.glob("*.md")):
            for line in md.read_text().splitlines():
                m = _MARKER.match(line)
                if not m:
                    continue
                path, heading, level = _parse(m)
                pulled = extract((root / path).read_text(), heading, level)
                for _, _, title in _headings(pulled):
                    self.page_of.setdefault((path, slug(title)), md.stem)
                if heading is None:
                    self.default.setdefault(path, md.stem)

    def _fix_links(self, text: str, path: str, docname: str) -> str:
        here = posixpath.dirname(path)

        def fix(m):
            label, target = m.groups()
            if re.match(r"[a-z]+:", target):
                return m.group(0)
            file, _, frag = target.partition("#")
            full = posixpath.normpath(posixpath.join(here, file)) if file else path
            page = self.page_of.get((full, frag)) or self.default.get(full)
            anchor = f"#{frag}" if frag else ""
            if page is not None:
                if page == docname and anchor:
                    return f"[{label}]({anchor})"
                return f"[{label}]({page}.md{anchor if (full, frag) in self.page_of else ''})"
            if not file:  # a fragment of a file that isn't pulled in
                return m.group(0)
            if full.startswith(".."):  # sibling checkout: nothing to link to
                return label
            return f"[{label}]({GITHUB}{full}{anchor})"

        return _LINK.sub(fix, text)

    def expand(self, source: str, docname: str) -> str:
        out, touched = [], False
        for line in source.splitlines():
            m = _MARKER.match(line)
            if not m:
                out.append(line)
                continue
            touched = True
            path, heading, level = _parse(m)
            text = "\n".join(extract((self.root / path).read_text(), heading, level))
            text = _PICTURE.sub(_picture, text)
            out.append(self._fix_links(text, path, docname))
        return "\n".join(out) + "\n" if touched else source


def _picture(m: re.Match) -> str:
    alt = _ALT.search(m.group(0))
    alt = alt.group(1) if alt else "pipeline"
    return "\n".join(
        f"```{{image}} pipeline/pipeline-{mode}.svg\n:alt: {alt}\n:class: only-{mode}\n```"
        for mode in ("light", "dark")
    )
