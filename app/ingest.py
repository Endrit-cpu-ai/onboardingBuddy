"""Load the markdown docs, redact secrets, split them into chunks and drop prompt-injection sections.

    python -m app.ingest    # print a report
"""
import hashlib
import re
from dataclasses import dataclass, field

import yaml

from app.config import CHUNK_OVERLAP, CHUNK_WORDS, KB_DIR

# anything matching these gets redacted before it reaches the index
SECRET_PATTERNS = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws_key"),
    (re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}"), "slack_token"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "api_key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"), "private_key"),
    (re.compile(r"(\w+://[^:\s/]+:)([^@\s]+)(@)"), "url_password"),           # scheme://user:PASS@host
    (re.compile(r"(?i)\b(password|passwd|pwd|secret|api[ _-]?key|token)(\s*[:=]\s*)(\S+)"), "password"),
]

# sections matching this are skipped entirely
INJECTION = re.compile(
    r"(?i)(ignore|disregard) (all |any )?(the )?(previous|prior|above) instructions"
    r"|system (prompt|instruction)|you are now|note for (ai|llm|assistant)s?"
)

REQUIRED_FIELDS = ("title", "url", "acl")


@dataclass
class Doc:
    path: str
    title: str
    url: str
    acl: list
    hash: str           # sha256 of the raw file, used to skip unchanged docs
    body: str
    secrets: list = field(default_factory=list)


@dataclass
class Chunk:
    id: str             # e.g. "it-setup-vpn.md#2"
    path: str
    title: str
    url: str
    heading: str
    text: str
    acl: list
    doc_hash: str

    @property
    def embed_text(self):
        # prefix with title + heading so the chunk makes sense on its own
        return f"{self.title} > {self.heading}\n{self.text}"


def parse_frontmatter(raw, name):
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.DOTALL)
    if not m:
        raise ValueError(f"{name}: missing the --- metadata block at the top")
    meta = yaml.safe_load(m.group(1)) or {}
    missing = [f for f in REQUIRED_FIELDS if not meta.get(f)]
    if missing:
        raise ValueError(f"{name}: metadata is missing {missing}")
    return meta, m.group(2)


def redact(text):
    found = []
    for pattern, kind in SECRET_PATTERNS:
        def repl(m, kind=kind):
            found.append(kind)
            if kind == "password":
                return f"{m.group(1)}{m.group(2)}[REDACTED]"     # keep "password:", drop the value
            if kind == "url_password":
                return f"{m.group(1)}[REDACTED]{m.group(3)}"
            return "[REDACTED]"
        text = pattern.sub(repl, text)
    return text, found


def load_docs():
    docs = []
    for path in sorted(KB_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        meta, body = parse_frontmatter(raw, path.name)
        body, secrets = redact(body)
        acl = meta["acl"] if isinstance(meta["acl"], list) else [meta["acl"]]
        docs.append(Doc(
            path=path.name, title=meta["title"], url=meta["url"], acl=acl,
            hash=hashlib.sha256(raw.encode()).hexdigest(), body=body, secrets=secrets,
        ))
    return docs


def split_sections(body):
    """Split on ## headings."""
    heading, lines = "Overview", []
    for line in body.splitlines():
        if line.startswith("# "):                # title is in the frontmatter already
            continue
        if line.startswith("## "):
            if "".join(lines).strip():
                yield heading, "\n".join(lines).strip()
            heading, lines = line[3:].strip(), []
        else:
            lines.append(line)
    if "".join(lines).strip():
        yield heading, "\n".join(lines).strip()


def window(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Cut long sections into overlapping windows of `size` words."""
    words = text.split()
    if len(words) <= size:
        return [text]
    pieces, step = [], size - overlap
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return pieces


def chunk_doc(doc):
    chunks, quarantined = [], []
    for heading, text in split_sections(doc.body):
        if INJECTION.search(text):
            quarantined.append(f"{doc.path} > {heading}")
            continue
        for piece in window(text):
            chunks.append(Chunk(
                id=f"{doc.path}#{len(chunks)}", path=doc.path, title=doc.title, url=doc.url,
                heading=heading, text=piece, acl=doc.acl, doc_hash=doc.hash,
            ))
    return chunks, quarantined


def load_and_chunk():
    docs, chunks, quarantined = load_docs(), [], []
    for d in docs:
        c, q = chunk_doc(d)
        chunks += c
        quarantined += q
    return docs, chunks, quarantined


if __name__ == "__main__":
    docs, chunks, quarantined = load_and_chunk()

    print(f"{'document':<30} {'acl':<20} {'chunks':>6}  notes")
    for d in docs:
        n = sum(1 for c in chunks if c.path == d.path)
        notes = f"redacted {len(d.secrets)} secret(s)" if d.secrets else ""
        print(f"{d.path:<30} {','.join(d.acl):<20} {n:>6}  {notes}")

    print(f"\n{len(docs)} docs -> {len(chunks)} chunks")
    for q in quarantined:
        print(f"QUARANTINED (prompt injection): {q}")

    sample = next(c for c in chunks if c.heading == "Printers")
    print(f"\n--- chunk {sample.id}, exactly as it will be embedded ---\n{sample.embed_text}")