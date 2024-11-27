use std::{collections::BTreeMap, path::PathBuf, time::Instant};

use agglayer_types::{Certificate, LocalNetworkStateData, U256};
use clap::Parser;
use pessimistic_proof::{
    bridge_exit::{NetworkId, TokenInfo},
    generate_pessimistic_proof,
    global_index::GlobalIndex,
    LocalNetworkState, PessimisticProofOutput,
};
use pessimistic_proof_test_suite::{
    runner::Runner,
    sample_data::{self as data},
};
use reth_primitives::Address;
use serde::{Deserialize, Serialize};
use sp1_sdk::HashableKey;
use tracing::{info, warn};
use uuid::Uuid;

/// The arguments for the pp generator.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct PPGenArgs {
    /// The number of bridge exits.
    #[clap(long, default_value = "10")]
    n_exits: usize,

    /// The number of imported bridge exits.
    #[clap(long, default_value = "0")]
    n_imported_exits: usize,

    /// The optional output directory to write the proofs in JSON. If not set,
    /// the proof is simply logged.
    #[clap(long)]
    proof_dir: Option<PathBuf>,

    /// The optional path to the custom sample data.
    #[clap(long)]
    sample_path: Option<PathBuf>,
}

fn get_events(n: usize, path: Option<PathBuf>) -> Vec<(TokenInfo, U256)> {
    if let Some(p) = path {
        data::sample_bridge_exits(p)
            .cycle()
            .take(n)
            .map(|e| (e.token_info, e.amount))
            .collect::<Vec<_>>()
    } else {
        data::sample_bridge_exits_01()
            .cycle()
            .take(n)
            .map(|e| (e.token_info, e.amount))
            .collect::<Vec<_>>()
    }
}

pub fn main() {
    sp1_sdk::utils::setup_logger();

    let args = PPGenArgs::parse();

    let mut state = data::sample_state_00();
    let old_state = state.state_b.clone();

    let mut certificates: Vec<Certificate> = [
        "n15-cert_h0.json",
        "n15-cert_h1.json",
        "n15-cert_h2-v2.json",
        "n15-cert_h3-v2.json",
    ]
    .iter()
    .map(|p| data::load_certificate(p))
    .collect();

    let mut lns = LocalNetworkStateData::default();
    let mut nullifier: BTreeMap<GlobalIndex, usize> = BTreeMap::new();

    // Error during certification process of
    // 0x059fae19d364f09e954a7252ac23e7320b6e1f0b97a79298c2eb04015f3a7cb1: native
    // execution failed: InvalidNewBalanceRoot { declared:
    // a0d47f6ba5be0c44914a657868206c9847536230102abcb59a76812f1926dadf, computed:
    // 2c82837a204270bb078669e019992cd06a77393d843878df023d9a7fb5327829 }
    certificates[3].prev_local_exit_root = certificates[2].new_local_exit_root;
    for (idx, certificate) in certificates.iter().enumerate() {
        info!(
            "Certificate ({idx}|{}) | {}, nib:{} b:{}",
            certificate.height,
            certificate.hash(),
            certificate.imported_bridge_exits.len(),
            certificate.bridge_exits.len(),
        );

        let signer = certificate.signer().unwrap();
        let l1_info_root = certificate.l1_info_root().unwrap().unwrap_or_default();

        info!(
            "Certificate ({idx}|{}) | signer: {}",
            certificate.height, signer
        );
        for ib in &certificate.imported_bridge_exits {
            // let u256global_index: U256 = ib.global_index.into();
            // info!(
            //     "cert {idx} claim ib: ({}) {:?}",
            //     u256global_index, ib.global_index,
            // );
            if !nullifier.contains_key(&ib.global_index) {
                nullifier.insert(ib.global_index, idx);
            } else {
                info!(
                    "Certificate {} tries to claim {:?} but already claimed by certificate {}",
                    idx,
                    ib.global_index,
                    nullifier.get(&ib.global_index).unwrap()
                );
            }
        }
        let multi_batch_header = lns
            .make_multi_batch_header(&certificate, signer, l1_info_root)
            .unwrap();

        info!("Certificate {idx}: successful witness generation");

        let initial_state = LocalNetworkState::from(lns.clone());

        generate_pessimistic_proof(initial_state, &multi_batch_header).unwrap();
        info!("Certificate {idx}: successful native execution");

        lns.apply_certificate(&certificate, signer, l1_info_root)
            .unwrap();
        info!("Certificate {idx}: successful state transition, waiting for the next");
    }

    let bridge_exits = get_events(args.n_exits, args.sample_path.clone());
    let imported_bridge_exits = get_events(args.n_imported_exits, args.sample_path);

    let (certificate, signer) = state.apply_events(&imported_bridge_exits, &bridge_exits);

    info!(
        "Certificate {}: [{}]",
        certificate.hash(),
        serde_json::to_string(&certificate).unwrap()
    );

    let l1_info_root = certificate.l1_info_root().unwrap().unwrap_or_default();
    let multi_batch_header = old_state
        .make_multi_batch_header(&certificate, signer, l1_info_root)
        .unwrap();

    info!(
        "Generating the proof for {} bridge exit(s) and {} imported bridge exit(s)",
        bridge_exits.len(),
        imported_bridge_exits.len()
    );

    let start = Instant::now();
    let (proof, vk, new_roots) = Runner::new()
        .generate_plonk_proof(&old_state.into(), &multi_batch_header)
        .expect("proving failed");
    let duration = start.elapsed();
    info!(
        "Successfully generated the plonk proof with a latency of {:?}",
        duration
    );

    let vkey = vk.bytes32().to_string();
    info!("vkey: {}", vkey);

    let fixture = PessimisticProofFixture {
        certificate,
        pp_inputs: new_roots.into(),
        signer,
        vkey: vkey.clone(),
        public_values: format!("0x{}", hex::encode(proof.public_values.as_slice())),
        proof: format!("0x{}", hex::encode(proof.bytes())),
    };

    if let Some(proof_dir) = args.proof_dir {
        // Save the plonk proof to a json file.
        let proof_path = proof_dir.join(format!(
            "{}-exits-v{}-{}.json",
            args.n_exits,
            &vkey[..8],
            Uuid::new_v4()
        ));
        if let Err(e) = std::fs::create_dir_all(&proof_dir) {
            warn!("Failed to create directory: {e}");
        }
        info!("Writing the proof to {:?}", proof_path);
        std::fs::write(proof_path, serde_json::to_string_pretty(&fixture).unwrap())
            .expect("failed to write fixture");
    } else {
        info!("Proof: {:?}", fixture);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct VerifierInputs {
    /// The previous local exit root.
    pub prev_local_exit_root: String,
    /// The previous pessimistic root.
    pub prev_pessimistic_root: String,
    /// The l1 info root against which we prove the inclusion of the
    /// imported bridge exits.
    pub l1_info_root: String,
    /// The origin network of the pessimistic proof.
    pub origin_network: NetworkId,
    /// The consensus hash.
    pub consensus_hash: String,
    /// The new local exit root.
    pub new_local_exit_root: String,
    /// The new pessimistic root which commits to the balance and nullifier
    /// tree.
    pub new_pessimistic_root: String,
}

impl From<PessimisticProofOutput> for VerifierInputs {
    fn from(v: PessimisticProofOutput) -> Self {
        Self {
            prev_local_exit_root: format!("0x{}", hex::encode(v.prev_local_exit_root)),
            prev_pessimistic_root: format!("0x{}", hex::encode(v.prev_pessimistic_root)),
            l1_info_root: format!("0x{}", hex::encode(v.l1_info_root)),
            origin_network: v.origin_network,
            consensus_hash: format!("0x{}", hex::encode(v.consensus_hash)),
            new_local_exit_root: format!("0x{}", hex::encode(v.new_local_exit_root)),
            new_pessimistic_root: format!("0x{}", hex::encode(v.new_pessimistic_root)),
        }
    }
}

/// A fixture that can be used to test the verification of SP1 zkVM proofs
/// inside Solidity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct PessimisticProofFixture {
    certificate: Certificate,
    pp_inputs: VerifierInputs,
    signer: Address,
    vkey: String,
    public_values: String,
    proof: String,
}
