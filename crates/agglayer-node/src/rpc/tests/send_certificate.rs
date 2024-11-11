use std::{net::IpAddr, sync::Arc};

use agglayer_certificate_orchestrator::CertificateSubmitter;
use agglayer_config::Config;
use agglayer_types::{Certificate, CertificateId};
use ethers::providers;
use jsonrpsee::{core::client::ClientT, http_client::HttpClientBuilder, rpc_params};

use super::next_available_addr;
use crate::{
    kernel::Kernel,
    rpc::{tests::DummyStore, AgglayerImpl},
};

#[test_log::test(tokio::test)]
async fn send_certificate_method_can_be_called() {
    let _ = tracing_subscriber::FmtSubscriber::builder()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let mut config = Config::new_for_test();
    let addr = next_available_addr();
    if let IpAddr::V4(ip) = addr.ip() {
        config.rpc.host = ip;
    }
    config.rpc.port = addr.port();

    let config = Arc::new(config);

    let (provider, _mock) = providers::Provider::mocked();
    let (certificate_sender, mut certificate_receiver) = tokio::sync::mpsc::channel(1);

    let kernel = Kernel::new(Arc::new(provider), config.clone());

    let certificate_submitter = CertificateSubmitter::new(certificate_sender);

    let _server_handle = AgglayerImpl::new(
        kernel,
        Arc::new(DummyStore {}),
        Arc::new(DummyStore {}),
        Arc::new(DummyStore {}),
        certificate_submitter,
        config.clone(),
    )
    .start()
    .await
    .unwrap();

    let url = format!("http://{}/", config.rpc_addr());
    let client = HttpClientBuilder::default().build(url).unwrap();

    futures::future::join(
        async {
            let _: CertificateId = client
                .request(
                    "interop_sendCertificate",
                    rpc_params![Certificate::new_for_test(1.into(), 0)],
                )
                .await
                .unwrap();
        },
        async {
            let (_cert, resp) = certificate_receiver.recv().await.unwrap();
            let _ = resp.send(Ok(()));
        },
    )
    .await;
}

#[test_log::test(tokio::test)]
async fn send_certificate_method_can_be_called_and_fail() {
    let mut config = Config::new_for_test();
    let addr = next_available_addr();
    if let IpAddr::V4(ip) = addr.ip() {
        config.rpc.host = ip;
    }
    config.rpc.port = addr.port();

    let config = Arc::new(config);

    let (provider, _mock) = providers::Provider::mocked();
    let (certificate_sender, certificate_receiver) = tokio::sync::mpsc::channel(1);

    let kernel = Kernel::new(Arc::new(provider), config.clone());

    let certificate_submitter = CertificateSubmitter::new(certificate_sender);

    let _server_handle = AgglayerImpl::new(
        kernel,
        Arc::new(DummyStore {}),
        Arc::new(DummyStore {}),
        Arc::new(DummyStore {}),
        certificate_submitter,
        config.clone(),
    )
    .start()
    .await
    .unwrap();

    let url = format!("http://{}/", config.rpc_addr());
    let client = HttpClientBuilder::default().build(url).unwrap();

    drop(certificate_receiver);

    let res: Result<(), _> = client
        .request(
            "interop_sendCertificate",
            rpc_params![Certificate::default()],
        )
        .await;

    assert!(res.is_err());
}
