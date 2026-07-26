"""Deterministic, durable atomic gzip JSONL shards and reference-driven reading."""
from __future__ import annotations
import gzip,hashlib,io,json,os,shutil,time,uuid
from pathlib import Path
from typing import Any,Iterator
from .dataset import DecisionRecord

def _sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()
def _fsync_file(path:Path)->None:
    with path.open("rb") as handle:os.fsync(handle.fileno())
def _fsync_dir(path:Path)->None:
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def _line(record:DecisionRecord)->bytes:return (json.dumps(record.as_dict(),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def _raw_iter(path:Path)->Iterator[DecisionRecord]:
    found=False
    try:
        with gzip.open(path,"rt",encoding="utf-8") as handle:
            for number,line in enumerate(handle,1):
                found=True
                try:yield DecisionRecord.from_dict(json.loads(line))
                except (ValueError,TypeError,KeyError,json.JSONDecodeError) as error:raise ValueError(f"invalid shard record at line {number}: {path}") from error
    except (gzip.BadGzipFile,EOFError) as error:raise ValueError(f"invalid gzip shard: {path}") from error
    if not found:raise ValueError(f"empty shard: {path}")
def iter_records(path:Path|str,*,expected_sha256:str|None=None)->Iterator[DecisionRecord]:
    shard=Path(path)
    if shard.name.endswith(".partial"):raise ValueError("partial shards cannot be read")
    if expected_sha256 is not None and _sha256(shard)!=expected_sha256:raise ValueError(f"shard SHA-256 mismatch: {shard}")
    yield from _raw_iter(shard)
def _exact_nonnegative_int(value:object,label:str)->int:
    if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError(f"invalid dataset reference {label}")
    return value
def _sha(value:object,label:str)->str:
    if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):raise ValueError(f"invalid dataset reference {label}")
    return value
def _validate_record_file(path:Path,expected_hash:str,expected_count:int,split:str)->None:
    if _sha256(path)!=expected_hash:raise ValueError(f"shard SHA-256 mismatch: {path}")
    count=0
    for record in _raw_iter(path):
        if record.split!=split:raise ValueError("record split disagrees with shard")
        count+=1
    if count!=expected_count:raise ValueError("shard count mismatch")
def iter_dataset(root:Path|str)->Iterator[DecisionRecord]:
    base=Path(root);reference_path=base/"dataset_reference.json"
    reference=json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("schema_version")!="dataset_reference_v3":raise ValueError("unsupported dataset reference schema")
    owner_name=reference.get("publication_owner_path")
    if owner_name!="publication_owner.json":raise ValueError("invalid dataset reference owner path")
    declared={"dataset_reference.json",owner_name};observed={"train":0,"validation":0}
    required={"schema_version","counts","shards","unknown_option_field_counts","source_manifest_sha256","source_audit","distribution_audit","split_private_assignments_sha256","split_audit_sha256","protocol_sha256","record_schema_version","action_contract_version","typed_input_schema_version","feature_compiler_version","feature_compiler_sha256","ontology_sha256","publication_owner_path","publication_owner_sha256","content_sha256"}
    counts=reference.get("counts");shard_map=reference.get("shards")
    if set(reference)!=required or not isinstance(counts,dict) or set(counts)!={"train","validation"} or not isinstance(shard_map,dict) or set(shard_map)!={"train","validation"}:raise ValueError("invalid dataset reference schema")
    for split in counts:_exact_nonnegative_int(counts[split],f"counts.{split}")
    for field in ("source_manifest_sha256","split_private_assignments_sha256","split_audit_sha256","protocol_sha256","feature_compiler_sha256","ontology_sha256","publication_owner_sha256","content_sha256"):_sha(reference[field],field)
    for field in ("record_schema_version","action_contract_version","typed_input_schema_version","feature_compiler_version"):
        if not isinstance(reference[field],str) or not reference[field]:raise ValueError("invalid dataset reference types")
    if not isinstance(reference["unknown_option_field_counts"],dict):raise ValueError("invalid dataset reference types")
    entries=[]
    for split in ("train","validation"):
        if not isinstance(shard_map[split],list):raise ValueError("invalid dataset reference shard list")
        for item in shard_map[split]:
            if not isinstance(item,dict) or set(item)!={"path","sha256","count"}:raise ValueError("invalid dataset reference shard declaration schema")
            name=item["path"]
            if not isinstance(name,str):raise ValueError("invalid dataset reference shard path")
            candidate=Path(name);_sha(item["sha256"],"shard sha256");_exact_nonnegative_int(item["count"],"shard count")
            if candidate.name!=name or candidate.is_absolute() or not name.startswith(f"{split}-") or not name.endswith(".jsonl.gz"):raise ValueError("unsafe or mismatched shard path")
            if name in declared:raise ValueError("duplicate shard declaration")
            declared.add(name);entries.append((split,item))
    entries_list=list(base.iterdir())
    actual={path.name for path in entries_list}
    if actual!=declared or any(path.is_symlink() or not path.is_file() for path in entries_list):raise ValueError("dataset contains undeclared or invalid root entries")
    commitment=reference["content_sha256"];copy=dict(reference);del copy["content_sha256"]
    if hashlib.sha256((json.dumps(copy,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()).hexdigest()!=commitment:raise ValueError("dataset reference content commitment mismatch")
    owner_path=base/owner_name
    if _sha256(owner_path)!=reference["publication_owner_sha256"]:raise ValueError("publication owner SHA-256 mismatch")
    owner=json.loads(owner_path.read_text(encoding="utf-8"))
    dataset_basis=dict(reference)
    for field in ("publication_owner_path","publication_owner_sha256","content_sha256"):del dataset_basis[field]
    dataset_commitment=hashlib.sha256((json.dumps(dataset_basis,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()).hexdigest()
    if set(owner)!={"schema_version","dataset_content_sha256"} or owner["schema_version"]!="dataset_publication_owner_v1" or owner["dataset_content_sha256"]!=dataset_commitment:raise ValueError("invalid dataset publication owner")
    for split,item in entries:_validate_record_file(base/item["path"],item["sha256"],item["count"],split);observed[split]+=item["count"]
    if observed!=reference["counts"]:raise ValueError("dataset aggregate count mismatch")
    for split,item in entries:yield from _raw_iter(base/item["path"])


def recover_orphan_staging(
    output_dir: Path | str,
    *,
    diagnostic_path: Path | str,
    reason: str,
) -> dict[str, Any]:
    """Audit and remove one unpublished staging pair left by an interrupted writer."""
    output = Path(output_dir)
    diagnostic = Path(diagnostic_path)
    if output.exists():
        raise FileExistsError(f"published dataset output exists: {output}")
    if not reason.strip():
        raise ValueError("orphan recovery requires a nonempty reason")
    if diagnostic.exists():
        raise FileExistsError(f"orphan diagnostic already exists: {diagnostic}")
    parent = output.parent
    stage_prefix = f".{output.name}.partial-"
    reservation_prefix = f".{output.name}.publish-"
    stages = sorted(path for path in parent.iterdir() if path.name.startswith(stage_prefix))
    reservations = sorted(
        path for path in parent.iterdir() if path.name.startswith(reservation_prefix)
    )
    if len(stages) != 1 or len(reservations) != 1:
        raise ValueError("orphan recovery requires exactly one staging/reservation pair")
    stage, reservation = stages[0], reservations[0]
    stage_token = stage.name.removeprefix(stage_prefix)
    reservation_token = reservation.name.removeprefix(reservation_prefix)
    if not stage_token or stage_token != reservation_token:
        raise ValueError("orphan staging/reservation token mismatch")
    if stage.is_symlink() or reservation.is_symlink():
        raise ValueError("orphan recovery refuses symlinks")
    if not stage.is_dir() or not reservation.is_dir() or any(reservation.iterdir()):
        raise ValueError("invalid orphan staging/reservation layout")

    files: list[dict[str, Any]] = []
    for path in sorted(stage.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid orphan staging entry: {path.name}")
        if not (path.name.endswith(".jsonl.gz") or path.name.endswith(".jsonl.gz.partial")):
            raise ValueError(f"unknown orphan staging entry: {path.name}")
        item: dict[str, Any] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "partial_suffix": path.name.endswith(".partial"),
        }
        try:
            item["record_count"] = sum(1 for _ in _raw_iter(path))
            item["gzip_and_records_valid"] = True
        except (OSError, ValueError) as error:
            item["record_count"] = None
            item["gzip_and_records_valid"] = False
            item["validation_error"] = f"{type(error).__name__}: {error}"
        files.append(item)

    report = {
        "schema_version": "dataset_orphan_recovery_v1",
        "output_path": str(output),
        "staging_name": stage.name,
        "reservation_name": reservation.name,
        "reason": reason.strip(),
        "recovered_at_unix": time.time(),
        "files": files,
    }
    diagnostic.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(diagnostic, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        diagnostic.unlink(missing_ok=True)
        raise
    _fsync_dir(diagnostic.parent)
    shutil.rmtree(stage)
    reservation.rmdir()
    _fsync_dir(parent)
    return report


class AtomicShardWriter:
    def __init__(self,output_dir:Path,*,shard_size:int)->None:
        self.output_dir=output_dir
        if output_dir.exists():raise FileExistsError(f"dataset output already exists: {output_dir}")
        output_dir.parent.mkdir(parents=True,exist_ok=True);prefix=f".{output_dir.name}.partial-"
        if any(p.name.startswith(prefix) for p in output_dir.parent.iterdir()):raise FileExistsError("orphan dataset staging directory exists")
        self.token=uuid.uuid4().hex;self.stage=output_dir.parent/f"{prefix}{self.token}";self.reservation=output_dir.parent/f".{output_dir.name}.publish-{self.token}"
        self.reservation.mkdir();self.stage.mkdir();_fsync_dir(output_dir.parent);self.shard_size=shard_size
        self.counts={"train":0,"validation":0};self.shards={"train":[],"validation":[]};self._handles={};self._paths={};self._closed=False
    def _release_reservation(self)->None:
        if self.reservation.exists():self.reservation.rmdir();_fsync_dir(self.output_dir.parent)
    def _owns_publication(self)->bool:
        marker=self.output_dir/"publication_owner.json"
        try:
            value=json.loads(marker.read_text(encoding="utf-8"))
            expected=getattr(self,"dataset_content_sha256",None)
            return expected is not None and value=={"schema_version":"dataset_publication_owner_v1","dataset_content_sha256":expected}
        except (OSError,json.JSONDecodeError,AttributeError):return False
    def _open(self,split:str)->Any:
        path=self.stage/f"{split}-{self.counts[split]//self.shard_size:05d}.jsonl.gz.partial";raw=path.open("wb");gz=gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0);text=io.TextIOWrapper(gz,encoding="utf-8",newline="")
        self._paths[split]=path;self._handles[split]=(text,raw);return text
    def _finish(self,split:str)->None:
        pair=self._handles.pop(split,None)
        if pair is None:return
        text,raw=pair;text.flush();text.detach().close();raw.flush();os.fsync(raw.fileno());raw.close();partial=self._paths.pop(split);final=partial.with_suffix("");os.replace(partial,final);_fsync_dir(self.stage)
        expected=min(self.shard_size,self.counts[split]-len(self.shards[split])*self.shard_size);count=sum(1 for _ in _raw_iter(final))
        if count!=expected:raise ValueError("shard record count mismatch")
        self.shards[split].append({"path":final.name,"sha256":_sha256(final),"count":count})
    def write(self,split:str,record:DecisionRecord)->None:
        if split not in self.counts:raise ValueError("invalid split")
        if self.counts[split]%self.shard_size==0:self._finish(split);handle=self._open(split)
        else:handle=self._handles[split][0]
        handle.write(_line(record).decode());self.counts[split]+=1
    def finalize(self,unknown:dict[str,int],metadata:dict[str,Any])->dict[str,Any]:
        for split in self.counts:self._finish(split)
        if not all(self.counts.values()):raise ValueError("both train and validation must be nonempty")
        basis={"schema_version":"dataset_reference_v3","counts":dict(self.counts),"shards":self.shards,"unknown_option_field_counts":dict(sorted(unknown.items())),**metadata}
        self.dataset_content_sha256=hashlib.sha256((json.dumps(basis,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()).hexdigest()
        owner_bytes=(json.dumps({"schema_version":"dataset_publication_owner_v1","dataset_content_sha256":self.dataset_content_sha256},sort_keys=True,separators=(",",":"))+"\n").encode()
        reference={**basis,"publication_owner_path":"publication_owner.json","publication_owner_sha256":hashlib.sha256(owner_bytes).hexdigest()}
        reference["content_sha256"]=hashlib.sha256((json.dumps(reference,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()).hexdigest()
        partial=self.stage/"dataset_reference.json.partial";partial.write_text(json.dumps(reference,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n",encoding="utf-8");_fsync_file(partial);os.replace(partial,self.stage/"dataset_reference.json");_fsync_dir(self.stage)
        owner=self.stage/"publication_owner.json";owner.write_bytes(owner_bytes);_fsync_file(owner);_fsync_dir(self.stage)
        renamed=False
        try:
            os.replace(self.stage,self.output_dir);renamed=True;_fsync_dir(self.output_dir.parent)
            self._release_reservation()
        except BaseException:
            if renamed and self._owns_publication():shutil.rmtree(self.output_dir);_fsync_dir(self.output_dir.parent)
            try:self._release_reservation()
            except BaseException:pass
            raise
        self._closed=True;return reference
    def abort(self)->None:
        if self._closed:return
        for text,raw in self._handles.values():
            try:text.close()
            except Exception:pass
            try:raw.close()
            except Exception:pass
        self._handles.clear();shutil.rmtree(self.stage,ignore_errors=True)
        if self._owns_publication():shutil.rmtree(self.output_dir,ignore_errors=True)
        self._release_reservation();_fsync_dir(self.output_dir.parent);self._closed=True
__all__=["AtomicShardWriter","iter_dataset","iter_records","recover_orphan_staging"]
