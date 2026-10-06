"""Step 2: LOAD + CLEAN + CHUNK.
Read every doc in knowledge_base/, check its metadata, strip secrets, cut it into one chunk per topic,
and drop any section that looks like prompt injection.

    python -m app.ingest      # print a report of docs and chunks
"""
import hashlib
import re
from dataclasses import dataclass, field

import yaml

from app.config import CHUNK_OVERLAP, CHUNK_WORDS, KB_DIR

# ---------- cleaning rules ----------

# Secrets that must never reach the index (and therefore never an answer)
SECRET_PATTERNS = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws_key"),
    (re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}"), "slack_token"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "api_key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"), "private_key"),
    (re.compile(r"(\w+://[^:\s/]+:)([^@\s]+)(@)"), "url_password"),           # postgres://user:PASS@host
    (re.compile(r"(?i)\b(password|passwd|pwd|secret|api[ _-]?key|token)(\s*[:=]\s*)(\S+)"), "password"),
]

# Text that tries to give orders to the AI
INJECTION = re.compile(
    r"(?i)(ignore|disregard) (all |any )?(the )?(previous|prior|above) instructions"
    r"|system (prompt|instruction)|you are now|note for (ai|llm|assistant)s?"
)

REQUIRED_FIELDS = ("title", "url", "acl")


# ---------- data shapes ----------

@dataclass
class Doc:
    path: str
    title: str
    url: str
    acl: list
    hash: str           # fingerprint of the raw file: changes only when the file changes
    body: str
    secrets: list = field(default_factory=list)


@dataclass
class Chunk:
    id: str             # stable id, e.g. "it-setup-vpn.md#2"
    path: str
    title: str
    url: str
    heading: str
    text: str
    acl: list
    doc_hash: str

    @property
    def embed_text(self):
        """What gets embedded: the chunk plus where it came from, so it makes sense on its own."""
        return f"{self.title} > {self.heading}\n{self.text}"


# ---------- load + clean ----------

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
                return f"{m.group(1)}{m.group(2)}[REDACTED]"     # keep the label, hide the value
            if kind == "url_password":
                return f"{m.group(1)}[REDACTED]{m.group(3)}"
            return "[REDACTED]"
        text = pattern.sub(repl, text)
    return text, found


def load_docs():
    docs = []
    for path in sorted(KB_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")   # Windows line endings
        meta, body = parse_frontmatter(raw, path.name)
        body, secrets = redact(body)
        acl = meta["acl"] if isinstance(meta["acl"], list) else [meta["acl"]]
        docs.append(Doc(
            path=path.name, title=meta["title"], url=meta["url"], acl=acl,
            hash=hashlib.sha256(raw.encode()).hexdigest(), body=body, secrets=secrets,
        ))
    return docs


# ---------- chunk ----------

def split_sections(body):
    """One section per '## ' heading: the author already grouped the text by topic."""
    heading, lines = "Overview", []
    for line in body.splitlines():
        if line.startswith("# "):                # H1 = doc title, already in metadata
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
    """Sections longer than `size` words are cut into overlapping windows."""
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
            continue                              # never indexed
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