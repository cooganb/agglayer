use agglayer_types::{Height, NetworkId};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("RocksDB error: {0}")]
    RocksDB(#[from] rocksdb::Error),

    #[error("Trying to access an unknown ColumnFamily")]
    ColumnFamilyNotFound,

    #[error(r#"Serialization error: {0}
        This is a critical bug that needs to be reported on `https://github.com/agglayer/agglayer/issues`"#)]
    Serialization(#[from] bincode::Error),

    #[error(r#"An unexpected error occurred: {0}
        This is a critical bug that needs to be reported on `https://github.com/agglayer/agglayer/issues`"#)]
    Unexpected(String),

    #[error("No certificate found")]
    NoCertificate,

    #[error("No proof found")]
    NoProof,

    #[error("The store is already in packing mode")]
    AlreadyInPackingMode,

    #[error(transparent)]
    CertificateCandidateError(#[from] CertificateCandidateError),

    #[error("Unprocessed action: {0}")]
    UnprocessedAction(String),
}

#[derive(Debug, thiserror::Error)]
pub enum CertificateCandidateError {
    #[error("Invalid certificate candidate for network {0} at height {1} for current epoch")]
    Invalid(NetworkId, Height),

    #[error(
        "Invalid certificate candidate for network {0}: {1} wasn't expected, current height {2}"
    )]
    UnexpectedHeight(NetworkId, Height, Height),
}
