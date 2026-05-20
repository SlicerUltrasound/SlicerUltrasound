"""Deterministic DICOM UID remapping for the AnonymizeUltrasound de-id pipeline.

Anonymized UIDs live in the OID arc ``2.25.`` (ITU-T X.667 / ISO-IEC 9834-8),
which permits UUID-derived UIDs without organizational registration. The
deterministic SHA-256 input means the same source UID always produces the
same anonymized UID — satisfying DICOM PS3.15 E.1.1 action ``U``'s "internally
consistent within a set of Instances" requirement without a session cache.
"""

import hashlib


def remap_uid(orig_uid: str) -> str:
    digest = hashlib.sha256(orig_uid.encode("ascii")).digest()[:15]
    return f"2.25.{int.from_bytes(digest, 'big')}"
