"""Order-independent overlap graph and auditable SAM mask post-processing."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np

@dataclass(frozen=True)
class Segment:
    segment_id: int
    mask: np.ndarray
    source_segment_ids: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        object.__setattr__(self, "mask", np.asarray(self.mask, dtype=bool))
        if self.mask.ndim != 2: raise ValueError("segment mask must be 2-D")
        if not self.source_segment_ids: object.__setattr__(self, "source_segment_ids", (self.segment_id,))
    @property
    def merge_count(self): return len(self.source_segment_ids)

def mask_iou(a, b):
    a=np.asarray(a,bool); b=np.asarray(b,bool)
    if a.shape != b.shape: raise ValueError("mask shapes differ")
    union=np.count_nonzero(a|b)
    return 1.0 if union == 0 else np.count_nonzero(a&b)/union

def containment(a,b):
    a=np.asarray(a,bool); b=np.asarray(b,bool); denominator=min(a.sum(),b.sum())
    return 1.0 if denominator == 0 and not (a|b).any() else (0.0 if denominator == 0 else np.count_nonzero(a&b)/denominator)

def postprocess_masks(segments, *, enabled=True, strategy="merge", iou_threshold=.8,
                      containment_enabled=False, containment_threshold=.95):
    items=list(segments)
    if not enabled or strategy == "none": return items
    parent=list(range(len(items)))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    links=[]
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            score=mask_iou(items[i].mask,items[j].mask); contained=containment(items[i].mask,items[j].mask)
            if score >= iou_threshold or (containment_enabled and contained >= containment_threshold): union(i,j); links.append((items[i].segment_id,items[j].segment_id,score,contained))
    groups={}
    for i in range(len(items)): groups.setdefault(find(i),[]).append(items[i])
    output=[]
    for new_id, group in enumerate(groups.values(),1):
        sources=tuple(sorted(x for s in group for x in s.source_segment_ids))
        chosen=max(group,key=lambda s:s.mask.sum())
        mask=np.logical_or.reduce([s.mask for s in group]) if strategy=="merge" else chosen.mask.copy()
        source_metadata = [record for segment in group
                           for record in segment.metadata.get("source_segments", [])]
        metadata={"merged_segment_id":new_id,"source_segment_ids":sources,
                  "merge_count":len(sources), "source_segments":source_metadata,
                  "overlap_links":[{"source_segment_id":x[0],
                                    "target_segment_id":x[1], "iou":x[2],
                                    "containment":x[3]} for x in links
                                   if x[0] in sources and x[1] in sources]}
        output.append(Segment(new_id,mask,sources,metadata))
    return output
