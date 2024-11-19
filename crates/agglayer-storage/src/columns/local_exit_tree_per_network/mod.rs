use agglayer_types::Hash;
use serde::{Deserialize, Serialize};

use super::{Codec, ColumnSchema, LOCAL_EXIT_TREE_PER_NETWORK_CF};

/// Column family for the local exit tree per network.
///
/// ## Column definition
///
/// | key                             | value  |
/// | --                              | --     |
/// | (`NetworkId`, `Layer`, `Index`) | `Hash` |
pub struct LocalExitTreePerNetworkColumn;

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key {
    pub(crate) network_id: u32,
    pub(crate) layer: u32,
    pub(crate) index: u32,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Prefix {
    pub(crate) network_id: u32,
    pub(crate) layer: u32,
}

pub type Value = Hash;

impl Codec for Key {}

impl ColumnSchema for LocalExitTreePerNetworkColumn {
    type Key = Key;
    type Value = Value;

    const COLUMN_FAMILY_NAME: &'static str = LOCAL_EXIT_TREE_PER_NETWORK_CF;
}
