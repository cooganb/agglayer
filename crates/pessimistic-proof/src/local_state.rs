use std::collections::{btree_map::Entry, BTreeMap};

use alloy_primitives::{ruint::UintTryFrom, U256, U512};
use serde::{Deserialize, Serialize};
use sha2::{Digest as Sha256Digest, Sha256};
use sp1_zkvm::lib::verify::verify_sp1_proof;

use crate::{
    bridge_exit::{LeafType, L1_ETH, L1_NETWORK_ID},
    imported_bridge_exit::{commit_imported_bridge_exits, Error},
    keccak::{Digest, Hash},
    local_balance_tree::LocalBalanceTree,
    local_exit_tree::{hasher::Keccak256Hasher, LocalExitTree},
    multi_batch_header::{signature_commitment, MultiBatchHeader},
    nullifier_tree::{NullifierKey, NullifierTree},
    ProofError,
};

/// State representation of one network without the leaves, taken as input by
/// the prover.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct LocalNetworkState {
    /// Commitment to the [`BridgeExit`].
    pub exit_tree: LocalExitTree<Keccak256Hasher>,
    /// Commitment to the balance for each token.
    pub balance_tree: LocalBalanceTree<Keccak256Hasher>,
    /// Commitment to the Nullifier tree for the local network, tracks claimed
    /// assets on foreign networks
    pub nullifier_tree: NullifierTree<Keccak256Hasher>,
}

/// The roots of one [`LocalNetworkState`].
#[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StateCommitment {
    pub exit_root: Digest,
    pub balance_root: Digest,
    pub nullifier_root: Digest,
}

impl StateCommitment {
    pub fn display_to_hex(&self) -> String {
        format!(
            "exit_root: {}, balance_root: {}, nullifier_root: {}",
            Hash(self.exit_root),
            Hash(self.balance_root),
            Hash(self.nullifier_root)
        )
    }
}

impl LocalNetworkState {
    /// Returns the roots.
    pub fn roots(&self) -> StateCommitment {
        StateCommitment {
            exit_root: self.exit_tree.get_root(),
            balance_root: self.balance_tree.root,
            nullifier_root: self.nullifier_tree.root,
        }
    }

    /// Apply the [`MultiBatchHeader`] on the current [`LocalNetworkState`].
    /// Checks that the transition reaches the target [`StateCommitment`].
    /// The state isn't modified on error.
    pub fn apply_batch_header(
        &mut self,
        multi_batch_header: &MultiBatchHeader<Keccak256Hasher>,
    ) -> Result<StateCommitment, ProofError> {
        let mut clone = self.clone();
        let roots = clone.apply_batch_header_helper(multi_batch_header)?;
        *self = clone;

        Ok(roots)
    }

    /// Apply the [`MultiBatchHeader`] on the current [`LocalNetworkState`].
    /// Returns the resulting [`StateCommitment`] upon success.
    /// The state can be modified on error.
    fn apply_batch_header_helper(
        &mut self,
        multi_batch_header: &MultiBatchHeader<Keccak256Hasher>,
    ) -> Result<StateCommitment, ProofError> {
        // Check the initial state
        let computed_root = self.exit_tree.get_root();
        if computed_root != multi_batch_header.prev_local_exit_root {
            return Err(ProofError::InvalidPreviousLocalExitRoot {
                computed: Hash(computed_root),
                declared: Hash(multi_batch_header.prev_local_exit_root),
            });
        }
        if self.balance_tree.root != multi_batch_header.prev_balance_root {
            return Err(ProofError::InvalidPreviousBalanceRoot {
                computed: Hash(self.balance_tree.root),
                declared: Hash(multi_batch_header.prev_balance_root),
            });
        }

        if self.nullifier_tree.root != multi_batch_header.prev_nullifier_root {
            return Err(ProofError::InvalidPreviousNullifierRoot {
                computed: Hash(self.nullifier_tree.root),
                declared: Hash(multi_batch_header.prev_nullifier_root),
            });
        }

        // TODO: benchmark if BTreeMap is the best choice in terms of SP1 cycles
        let mut new_balances = BTreeMap::new();
        for (k, v) in &multi_batch_header.balances_proofs {
            if new_balances.insert(*k, U512::from(v.0)).is_some() {
                return Err(ProofError::DuplicateTokenBalanceProof(*k));
            }
        }

        // Check batch_header.imported_exits_root
        let imported_exits_root = commit_imported_bridge_exits(
            multi_batch_header
                .imported_bridge_exits
                .iter()
                .map(|(exit, _)| exit),
        );

        if let Some(batch_imported_exits_root) = multi_batch_header.imported_exits_root {
            if imported_exits_root != batch_imported_exits_root {
                return Err(ProofError::InvalidImportedExitsRoot {
                    declared: Hash(batch_imported_exits_root),
                    computed: Hash(imported_exits_root),
                });
            }
        } else if !multi_batch_header.imported_bridge_exits.is_empty() {
            return Err(ProofError::MismatchImportedExitsRoot);
        }

        // Apply the imported bridge exits
        for (imported_bridge_exit, nullifier_path) in &multi_batch_header.imported_bridge_exits {
            if imported_bridge_exit.global_index.network_id() == multi_batch_header.origin_network {
                // We don't allow a chain to exit to itself
                return Err(ProofError::CannotExitToSameNetwork);
            }
            // Check that the destination network of the bridge exit matches the current
            // network
            if imported_bridge_exit.bridge_exit.dest_network != multi_batch_header.origin_network {
                return Err(ProofError::InvalidImportedBridgeExit {
                    source: Error::InvalidExitNetwork,
                    global_index: imported_bridge_exit.global_index,
                });
            }

            // Check the inclusion proof
            imported_bridge_exit
                .verify_path(multi_batch_header.l1_info_root)
                .map_err(|source| ProofError::InvalidImportedBridgeExit {
                    source,
                    global_index: imported_bridge_exit.global_index,
                })?;

            // Check the nullifier non-inclusion path and update the nullifier tree
            let nullifier_key: NullifierKey = imported_bridge_exit.global_index.into();
            self.nullifier_tree
                .verify_and_update(nullifier_key, nullifier_path)?;

            // The amount corresponds to L1 ETH if the leaf is a message
            let token_info = match imported_bridge_exit.bridge_exit.leaf_type {
                LeafType::Message => L1_ETH,
                _ => imported_bridge_exit.bridge_exit.token_info,
            };

            if multi_batch_header.origin_network == token_info.origin_network {
                // When the token is native to the chain, we don't care about the local balance
                continue;
            }

            // Update the token balance.
            let amount = imported_bridge_exit.bridge_exit.amount;
            let entry = new_balances.entry(token_info);
            match entry {
                Entry::Vacant(_) => return Err(ProofError::MissingTokenBalanceProof(token_info)),
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() = entry
                        .get()
                        .checked_add(U512::from(amount))
                        .ok_or(ProofError::BalanceOverflowInBridgeExit)?;
                }
            }
        }

        // Apply the bridge exits
        for bridge_exit in &multi_batch_header.bridge_exits {
            if bridge_exit.dest_network == multi_batch_header.origin_network {
                // We don't allow a chain to exit to itself
                return Err(ProofError::CannotExitToSameNetwork);
            }
            self.exit_tree.add_leaf(bridge_exit.hash())?;

            // For message exits, the origin network in token info should be the origin
            // network of the batch header.
            if bridge_exit.is_message()
                && bridge_exit.token_info.origin_network != multi_batch_header.origin_network
            {
                return Err(ProofError::InvalidMessageOriginNetwork);
            }

            // For ETH transfers, we need to check that the origin network is the L1 network
            if bridge_exit.token_info.origin_token_address.is_zero()
                && bridge_exit.token_info.origin_network != L1_NETWORK_ID
            {
                return Err(ProofError::InvalidL1TokenInfo(bridge_exit.token_info));
            }

            // The amount corresponds to L1 ETH if the leaf is a message
            let token_info = match bridge_exit.leaf_type {
                LeafType::Message => L1_ETH,
                _ => bridge_exit.token_info,
            };

            if multi_batch_header.origin_network == token_info.origin_network {
                // When the token is native to the chain, we don't care about the local balance
                continue;
            }

            // Update the token balance.
            let amount = bridge_exit.amount;
            let entry = new_balances.entry(token_info);
            match entry {
                Entry::Vacant(_) => return Err(ProofError::MissingTokenBalanceProof(token_info)),
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() = entry
                        .get()
                        .checked_sub(U512::from(amount))
                        .ok_or(ProofError::BalanceUnderflowInBridgeExit)?;
                }
            }
        }

        // Verify that the original balances were correct and update the local balance
        // tree with the new balances. TODO: implement batch `verify_and_update`
        // for the LBT
        for (token, (old_balance, balance_path)) in &multi_batch_header.balances_proofs {
            let new_balance = new_balances[token];
            let new_balance = U256::uint_try_from(new_balance)
                .map_err(|_| ProofError::BalanceOverflowInBridgeExit)?;
            self.balance_tree
                .verify_and_update(*token, balance_path, *old_balance, new_balance)?;
        }

        // Verify that the signature is valid
        // TODO: change this to SHA2 ?
        let combined_hash = signature_commitment(
            self.exit_tree.get_root(),
            multi_batch_header
                .imported_bridge_exits
                .iter()
                .map(|(exit, _)| exit),
        );

        // TODO: figure out what else needs to be a pv in the consensus proof.
        let consensus_public_values = [
            combined_hash.as_slice(),
            multi_batch_header.consensus_config.as_slice(),
        ]
        .concat();

        let vkey = multi_batch_header.vkey;
        let public_values_digest = Sha256::digest(&consensus_public_values);
        #[cfg(target_os = "zkvm")] // TODO: add a native verify otherwise.
        verify_sp1_proof(&vkey, &public_values_digest.into());

        // TODO: add native verification for the consensus proof if
        // `not(target_os="zkvm")`.

        Ok(self.roots())
    }
}
