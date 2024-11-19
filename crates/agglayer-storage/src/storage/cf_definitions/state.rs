use rocksdb::ColumnFamilyDescriptor;

pub const CFS: [&str; 6] = [
    crate::columns::CERTIFICATE_HEADER_CF,
    crate::columns::CERTIFICATE_PER_NETWORK_CF,
    crate::columns::LATEST_SETTLED_CERTIFICATE_PER_NETWORK_CF,
    crate::columns::METADATA_CF,
    //  crate::columns::LOCAL_EXIT_TREE_PER_NETWORK_CF,
    crate::columns::BALANCE_TREE_PER_NETWORK_CF,
    crate::columns::NULLIFIER_TREE_PER_NETWORK_CF,
];

/// Definitions for the column families in the state storage.
pub fn state_db_cf_definitions() -> Vec<ColumnFamilyDescriptor> {
    let mut vec = super::default_db_cf_definitions(&CFS);

    let mut cfg = rocksdb::Options::default();

    cfg.set_compression_type(rocksdb::DBCompressionType::Lz4);
    cfg.create_if_missing(true);
    cfg.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(2 * 32));

    vec.push(ColumnFamilyDescriptor::new(
        crate::columns::LOCAL_EXIT_TREE_PER_NETWORK_CF,
        cfg.clone(),
    ));

    vec
}
