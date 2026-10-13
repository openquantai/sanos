"""Guarded WRDS extraction and raw-source contracts."""

from sanos.wrds.client import (
    WrdsAccessDisabledError,
    WrdsAuthenticationError,
    WrdsConfigurationError,
    WrdsCredentials,
    WrdsError,
    WrdsQueryError,
)
from sanos.wrds.download import (
    RawArtifactRequest,
    RawCacheMissError,
    RawDownloadError,
    RawDownloadResult,
    WrdsNetworkAuthorizationError,
    download_raw,
    normalize_raw_frame,
    raw_artifact_request,
    raw_artifact_validation_record,
    raw_content_producer_version,
)
from sanos.wrds.lease import WrdsConnectionLease, WrdsLeaseUnavailableError
from sanos.wrds.plan import (
    ALL_RAW_TABLES,
    RawDownloadConfig,
    RawPartition,
    RawTable,
    plan_raw_partitions,
)

__all__ = [
    "ALL_RAW_TABLES",
    "RawArtifactRequest",
    "RawCacheMissError",
    "RawDownloadConfig",
    "RawDownloadError",
    "RawDownloadResult",
    "RawPartition",
    "RawTable",
    "WrdsAccessDisabledError",
    "WrdsAuthenticationError",
    "WrdsConfigurationError",
    "WrdsConnectionLease",
    "WrdsCredentials",
    "WrdsError",
    "WrdsLeaseUnavailableError",
    "WrdsNetworkAuthorizationError",
    "WrdsQueryError",
    "download_raw",
    "normalize_raw_frame",
    "plan_raw_partitions",
    "raw_artifact_request",
    "raw_artifact_validation_record",
    "raw_content_producer_version",
]
