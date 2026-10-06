//! PRL0 Kryptex miner - 0% dev fee PearlHash pool miner.
//!
//! This binary is intended to be dropped into the MIT/Apache-2.0
//! `puneet-mehta/pearl-hashrate-miner` source tree as src/bin/kryptex_miner.rs.
//! It reuses that project's SM86 CUDA/Triton kernels and proof builder, while
//! replacing solo-node RPC submission with Pearl Stratum pool submission.
//!
//! Environment:
//!   PRL_POOL       default prl-eu.kryptex.network:7048
//!   PRL_WALLET     required (PRL wallet or Kryptex login accepted by pool)
//!   PRL_WORKER     default WORKER_NAME or "hive"
//!   PRL_PASS       default x
//!   PRL_SHAPE      small | big_m | huge_m (default small; safest for RTX 3060 Ti)
//!   PEARL_FATBIN   default ./pearl_gemm.fatbin
//!   PEARL_DEVICES  comma separated CUDA ordinals, default all
//!   MAX_ITERS      0 = unlimited
//!
//! There is intentionally no developer-wallet path and no time-sliced fee.

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use pearl_hashrate_miner::driver::{device_count, CapturedGraph, CudaCtx, DevBuf, Module, Stream};
use pearl_hashrate_miner::gateway::{build_mining_config_triton_norotl, MiningConfig};
use pearl_hashrate_miner::miner_bufs::HOST_SIGNAL_HEADER_SIZE;
use pearl_hashrate_miner::error::cu_check;
use cudarc::driver::sys as cu;
use pearl_hashrate_miner::proof::{
    build_plain_proof, extract_indices, signal_header::ParsedSignalHeader,
};
use pearl_hashrate_miner::{MinerBufs, MinerBufsConfig, MinerError};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const AGENT: &str = "prl0-kryptex/0.1.6";
const RECONNECT_SECS: u64 = 2;
const LOG_SECS: u64 = 5;
const SUBMIT_ID_BASE: u64 = 1000;

fn miner_err(method: &str, msg: impl Into<String>) -> MinerError {
    MinerError::Rpc {
        method: method.to_string(),
        msg: msg.into(),
    }
}

struct PoolJob {
    header_bytes: [u8; 76],
    height: u64,
    job_id: String,
    target_be: [u8; 32],
    adjusted_target_le: [u8; 32],
    key: [u8; 32],
    mining_config: MiningConfig,
    cert_version: Option<u64>,
    cache_tag: Vec<u8>,
}

struct ReadyJob {
    pool_job: Arc<PoolJob>,
    b_bytes: Vec<u8>,
    m: usize,
    n: usize,
}

struct HitWork {
    job: Arc<ReadyJob>,
    a_bytes: Vec<u8>,
    a_rows: Vec<usize>,
    b_cols: Vec<usize>,
    src_gpu: i32,
}

struct SharedPool {
    latest: Mutex<Option<Arc<PoolJob>>>,
    writer: Mutex<Option<TcpStream>>,
    accepted: AtomicU64,
    rejected: AtomicU64,
    submit_seq: AtomicU64,
    authorized: AtomicBool,
}

impl SharedPool {
    fn new() -> Self {
        Self {
            latest: Mutex::new(None),
            writer: Mutex::new(None),
            accepted: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            submit_seq: AtomicU64::new(SUBMIT_ID_BASE),
            authorized: AtomicBool::new(false),
        }
    }

    fn submit(&self, job_id: &str, plain_proof_b64: &str) -> Result<u64, String> {
        let id = self.submit_seq.fetch_add(1, Ordering::Relaxed);
        let msg = json!({
            "id": id,
            "method": "mining.submit",
            "params": {
                "job_id": job_id,
                "plain_proof": plain_proof_b64
            }
        });
        let mut line = serde_json::to_vec(&msg).map_err(|e| e.to_string())?;
        line.push(b'\n');
        let mut guard = self.writer.lock().map_err(|_| "writer mutex poisoned".to_string())?;
        let stream = guard
            .as_mut()
            .ok_or_else(|| "pool not connected".to_string())?;
        stream.write_all(&line).map_err(|e| e.to_string())?;
        stream.flush().map_err(|e| e.to_string())?;
        Ok(id)
    }
}

fn mining_config() -> Result<MiningConfig, MinerError> {
    // Current fast Triton path used by the upstream miner: k=2048, rank=128,
    // rows=[0,1], cols=0..127.  Shape (m/n) is independent from this proof pattern.
    build_mining_config_triton_norotl(2048, 128)
}

fn parse_hex_32_be(s: &str) -> Result<[u8; 32], MinerError> {
    let raw = hex::decode(s).map_err(|e| miner_err("stratum.target", format!("bad hex: {e}")))?;
    if raw.len() != 32 {
        return Err(miner_err(
            "stratum.target",
            format!("expected 32-byte target, got {} bytes", raw.len()),
        ));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&raw);
    Ok(out)
}

fn parse_header_76(s: &str) -> Result<[u8; 76], MinerError> {
    let raw = hex::decode(s).map_err(|e| miner_err("stratum.header", format!("bad hex: {e}")))?;
    if raw.len() != 76 {
        return Err(miner_err(
            "stratum.header",
            format!("expected 76-byte header, got {} bytes", raw.len()),
        ));
    }
    let mut out = [0u8; 76];
    out.copy_from_slice(&raw);
    Ok(out)
}

/// 256-bit big-endian integer multiplied by u128. Errors on overflow.
fn mul_u256_u128(a_be: &[u8; 32], b: u128) -> Result<[u8; 32], MinerError> {
    let mut a_le = *a_be;
    a_le.reverse();
    let mut out = [0u128; 33];
    for i in 0..32 {
        let prod = (a_le[i] as u128) * b;
        out[i] += prod & 0xff;
        let mut carry = prod >> 8;
        let mut j = i + 1;
        while carry > 0 && j < out.len() {
            let s = out[j] + (carry & 0xff);
            out[j] = s & 0xff;
            carry = (carry >> 8) + (s >> 8);
            j += 1;
        }
    }
    let mut carry = 0u128;
    let mut le = [0u8; 32];
    for i in 0..32 {
        let s = out[i] + carry;
        le[i] = (s & 0xff) as u8;
        carry = s >> 8;
    }
    if out[32] + carry > 0 {
        return Err(miner_err("stratum.target", "adjusted target overflow"));
    }
    let mut be = le;
    be.reverse();
    Ok(be)
}

fn pool_job_from_notify(params: &Value) -> Result<PoolJob, MinerError> {
    let header_hex = params
        .get("header")
        .and_then(Value::as_str)
        .ok_or_else(|| miner_err("stratum.notify", "missing header"))?;
    let job_id = params
        .get("job_id")
        .and_then(Value::as_str)
        .ok_or_else(|| miner_err("stratum.notify", "missing job_id"))?
        .to_string();
    let target_hex = params
        .get("target")
        .and_then(Value::as_str)
        .ok_or_else(|| miner_err("stratum.notify", "missing target"))?;
    let height = params.get("height").and_then(Value::as_u64).unwrap_or(0);
    let cert_version = params.get("cert_version").and_then(Value::as_u64);
    // Current Pearl mainnet pools use certificate/noise-seed version 3.
    // Do not silently mine a cryptographically different legacy job.
    if let Some(v) = cert_version {
        if v != 3 {
            return Err(miner_err(
                "stratum.notify",
                format!("unsupported cert_version={v}; PRL0 0.1.6 expects v3"),
            ));
        }
    }

    let header_bytes = parse_header_76(header_hex)?;
    let target_be = parse_hex_32_be(target_hex)?;
    let mining_config = mining_config()?;

    // Upstream kernel compares against a penalized/adjusted target.  This is
    // the same transform used by MiningJob::build in pearl-hashrate-miner.
    let h = mining_config.rows_pattern.to_list().len() as u128;
    let w = mining_config.cols_pattern.to_list().len() as u128;
    let k = mining_config.dot_product_length() as u128;
    let adjustment = h
        .checked_mul(w)
        .and_then(|x| x.checked_mul(k))
        .ok_or_else(|| miner_err("stratum.target", "difficulty adjustment overflow"))?;
    let adjusted_be = mul_u256_u128(&target_be, adjustment)?;
    let mut adjusted_target_le = adjusted_be;
    adjusted_target_le.reverse();

    // Pearl per-job key = BLAKE3(header || serialized MiningConfiguration).
    let mut keying = Vec::with_capacity(128);
    keying.extend_from_slice(&header_bytes);
    keying.extend_from_slice(&mining_config.to_bytes());
    let key = *blake3::hash(&keying).as_bytes();

    // ensure_for_job only uses this byte vector as its change-detection key.
    // Include target + job id so vardiff changes refresh pow_target even if
    // a pool reuses the same 76-byte block header.
    let mut cache_tag = Vec::with_capacity(76 + 32 + job_id.len());
    cache_tag.extend_from_slice(&header_bytes);
    cache_tag.extend_from_slice(&target_be);
    cache_tag.extend_from_slice(job_id.as_bytes());

    Ok(PoolJob {
        header_bytes,
        height,
        job_id,
        target_be,
        adjusted_target_le,
        key,
        mining_config,
        cert_version,
        cache_tag,
    })
}

fn authorize_line(wallet: &str, worker: &str, pass: &str) -> Vec<u8> {
    let msg = json!({
        "id": 1,
        "method": "mining.authorize",
        "params": {
            "wallet": wallet,
            "worker": worker,
            "pass": pass,
            "agent": AGENT
        }
    });
    let mut out = serde_json::to_vec(&msg).expect("authorize json");
    out.push(b'\n');
    out
}

fn pool_thread(pool: Arc<SharedPool>, addr: String, wallet: String, worker: String, pass: String) {
    loop {
        println!("[pool] connecting to {addr} ...");
        match TcpStream::connect(&addr) {
            Ok(mut stream) => {
                let _ = stream.set_nodelay(true);
                let writer = match stream.try_clone() {
                    Ok(w) => w,
                    Err(e) => {
                        eprintln!("[pool] clone failed: {e}");
                        thread::sleep(Duration::from_secs(RECONNECT_SECS));
                        continue;
                    }
                };
                {
                    let mut guard = pool.writer.lock().unwrap();
                    *guard = Some(writer);
                }
                pool.authorized.store(false, Ordering::Relaxed);
                let auth = authorize_line(&wallet, &worker, &pass);
                if let Err(e) = stream.write_all(&auth).and_then(|_| stream.flush()) {
                    eprintln!("[pool] authorize write failed: {e}");
                    *pool.writer.lock().unwrap() = None;
                    thread::sleep(Duration::from_secs(RECONNECT_SECS));
                    continue;
                }
                println!("[pool] authorize sent as {wallet}/{worker}");

                let mut reader = BufReader::new(stream);
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line) {
                        Ok(0) => {
                            eprintln!("[pool] connection closed");
                            break;
                        }
                        Ok(_) => {
                            let trimmed = line.trim();
                            if trimmed.is_empty() {
                                continue;
                            }
                            let v: Value = match serde_json::from_str(trimmed) {
                                Ok(v) => v,
                                Err(e) => {
                                    eprintln!("[pool] bad JSON: {e}: {trimmed}");
                                    continue;
                                }
                            };

                            if v.get("method").and_then(Value::as_str) == Some("mining.notify") {
                                if let Some(params) = v.get("params") {
                                    match pool_job_from_notify(params) {
                                        Ok(job) => {
                                            println!(
                                                "[pool] job={} height={} target={} cert={:?}",
                                                job.job_id,
                                                job.height,
                                                hex::encode(&job.target_be[..8]),
                                                job.cert_version
                                            );
                                            *pool.latest.lock().unwrap() = Some(Arc::new(job));
                                        }
                                        Err(e) => eprintln!("[pool] notify rejected: {e}"),
                                    }
                                }
                                continue;
                            }

                            let id = v.get("id").and_then(Value::as_u64);
                            if id == Some(1) {
                                let ok = v.get("result").and_then(Value::as_bool).unwrap_or(false)
                                    && v.get("error").map(|x| x.is_null()).unwrap_or(true);
                                pool.authorized.store(ok, Ordering::Relaxed);
                                if ok {
                                    println!("[pool] authorized");
                                } else {
                                    eprintln!("[pool] authorization rejected: {v}");
                                }
                                continue;
                            }

                            if let Some(id) = id {
                                if id >= SUBMIT_ID_BASE {
                                    let ok = v.get("result").and_then(Value::as_bool).unwrap_or(false)
                                        && v.get("error").map(|x| x.is_null()).unwrap_or(true);
                                    if ok {
                                        let a = pool.accepted.fetch_add(1, Ordering::Relaxed) + 1;
                                        println!("[share] ACCEPT id={id} accepted={a}");
                                    } else {
                                        let r = pool.rejected.fetch_add(1, Ordering::Relaxed) + 1;
                                        eprintln!("[share] REJECT id={id} rejected={r} reply={v}");
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[pool] read failed: {e}");
                            break;
                        }
                    }
                }
                pool.authorized.store(false, Ordering::Relaxed);
                *pool.writer.lock().unwrap() = None;
            }
            Err(e) => eprintln!("[pool] connect failed: {e}"),
        }
        thread::sleep(Duration::from_secs(RECONNECT_SECS));
    }
}

fn hit_poller(rx: mpsc::Receiver<HitWork>, pool: Arc<SharedPool>) {
    while let Ok(work) = rx.recv() {
        let pjob = &work.job.pool_job;
        let started = Instant::now();
        let plain = match build_plain_proof(
            work.job.m,
            work.job.n,
            pjob.mining_config.common_dim as usize,
            pjob.mining_config.rank as usize,
            &work.a_bytes,
            &work.job.b_bytes,
            work.a_rows,
            work.b_cols,
            pjob.key,
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[proof] gpu{} build failed: {e}", work.src_gpu);
                continue;
            }
        };
        let wire = match bincode::serialize(&plain) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[proof] gpu{} serialize failed: {e}", work.src_gpu);
                continue;
            }
        };
        let encoded = BASE64.encode(wire);
        match pool.submit(&pjob.job_id, &encoded) {
            Ok(id) => println!(
                "[share] gpu{} submit id={} job={} proof={}B build={:.3}s",
                work.src_gpu,
                id,
                pjob.job_id,
                encoded.len(),
                started.elapsed().as_secs_f64()
            ),
            Err(e) => eprintln!("[share] gpu{} submit failed: {e}", work.src_gpu),
        }
    }
}

fn pick_config() -> MinerBufsConfig {
    match std::env::var("PRL_SHAPE")
        .unwrap_or_else(|_| "small".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "huge" | "huge_m" | "32768" => {
            println!("[miner] shape huge_m: 32768x32768x2048");
            MinerBufsConfig::shape_huge_m()
        }
        "big" | "big_m" | "16384" => {
            println!("[miner] shape big_m: 16384x32768x2048");
            MinerBufsConfig::shape_big_m()
        }
        _ => {
            println!("[miner] shape small: 8192x32768x2048 (RTX 30 starting profile)");
            MinerBufsConfig::shape_default()
        }
    }
}

fn pick_devices() -> Result<Vec<i32>, MinerError> {
    let avail = device_count()?;
    if avail <= 0 {
        return Err(miner_err("device_count", "no CUDA GPUs visible"));
    }
    let devs = match std::env::var("PEARL_DEVICES") {
        Ok(s) => s
            .split(',')
            .filter_map(|x| x.trim().parse::<i32>().ok())
            .collect::<Vec<_>>(),
        Err(_) => (0..avail).collect::<Vec<_>>(),
    };
    if devs.is_empty() {
        return Err(miner_err("PEARL_DEVICES", "empty device list"));
    }
    for d in &devs {
        if *d < 0 || *d >= avail {
            return Err(miner_err(
                "PEARL_DEVICES",
                format!("device {d} out of range 0..{}", avail - 1),
            ));
        }
    }
    Ok(devs)
}

fn resolve_worker_salt() -> String {
    for var in ["PEARL_WORKER_ID", "WORKER_NAME", "PRL_WORKER"] {
        if let Ok(v) = std::env::var(var) {
            let v = v.trim();
            if !v.is_empty() {
                return v.to_string();
            }
        }
    }
    let mut b = [0u8; 16];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        if f.read_exact(&mut b).is_ok() {
            return hex::encode(b);
        }
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("pid{}-{now}", std::process::id())
}

struct WorkerCtx {
    device_ord: i32,
    fatbin: Arc<Vec<u8>>,
    cfg: MinerBufsConfig,
    pool: Arc<SharedPool>,
    hit_tx: mpsc::Sender<HitWork>,
    gpu_ths_bits: Arc<AtomicU64>,
    max_iters: u64,
    worker_salt: Arc<String>,
}

fn worker(w: WorkerCtx) -> Result<(), MinerError> {
    let ctx = CudaCtx::new(w.device_ord)?;
    let tag = format!("[gpu{}]", w.device_ord);
    println!("{tag} device: {}", ctx.device_name()?);
    let module = Module::load_fatbin(&w.fatbin)?;
    let mut bufs = MinerBufs::new(&module, w.cfg)?;
    println!("{tag} m={} n={} k={} r={}", bufs.m, bufs.n, bufs.k, bufs.r);

    // NVIDIA 580 on this HiveOS host crashes when a kernel writes directly to
    // CU_MEMHOSTALLOC_DEVICEMAP pages. Keep the upstream pinned allocations
    // alive, but redirect every signal-header device pointer to ordinary VRAM.
    // After each batch we copy the tiny 1 KiB headers back to pageable RAM.
    let signal_dev_pool = (0..bufs.ring_size)
        .map(|_| DevBuf::alloc(HOST_SIGNAL_HEADER_SIZE))
        .collect::<Result<Vec<_>, MinerError>>()?;
    for slot in 0..bufs.ring_size {
        bufs.host_signal_header_pool[slot].device_ptr = signal_dev_pool[slot].ptr;
    }
    println!("{tag} host signal: device RAM + D2H (DEVICEMAP disabled)");

    let stream = Stream::new()?;
    // CUDA graph capture segfaults on the tested HiveOS/NVIDIA 580 + RTX 3060 Ti
    // stack. Default to the eager kernel path; graphs remain opt-in for later tests.
    let use_graphs = matches!(
        std::env::var("PRL_GRAPHS")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    );
    println!(
        "{tag} execution mode: {}",
        if use_graphs { "CUDA graphs" } else { "safe eager CUDA (1 iter + sync)" }
    );
    let mut graphs: Option<Vec<CapturedGraph>> = None;
    let mut current: Option<Arc<ReadyJob>> = None;
    let mut current_id = String::new();
    let mut iter_idx = 0u64;
    let mut last_log = Instant::now();
    let mut iters_at_log = 0u64;
    let mut hits_total = 0u64;
    let macs_per_iter_t = (bufs.m as f64 * bufs.n as f64 * bufs.k as f64) / 1e12;

    loop {
        let newest = w.pool.latest.lock().unwrap().as_ref().map(Arc::clone);
        if let Some(job) = newest {
            let identity = format!("{}:{}", job.job_id, hex::encode(&job.target_be[..8]));
            if identity != current_id {
                current_id = identity;
                let gpu_bytes = (w.device_ord as u32).to_le_bytes();
                let seed_material = [
                    &job.header_bytes[..],
                    b"prl0-kryptex-seed-v1",
                    w.worker_salt.as_bytes(),
                    &gpu_bytes[..],
                    job.job_id.as_bytes(),
                ]
                .concat();
                let seed = *blake3::hash(&seed_material).as_bytes();
                // Avoid the 64 MiB DEVICEMAP pinned B snapshot on HiveOS/NVIDIA 580.
                // Prepare the same per-job CUDA state, then copy B into normal host RAM.
                println!("{tag} job init: upload key/target/seed");
                bufs.key_tensor.copy_from(&job.key)?;
                bufs.pow_target_tensor.copy_from(&job.adjusted_target_le)?;
                bufs.seed_tensor.copy_from(&seed)?;
                println!("{tag} job init: generate B");
                unsafe {
                    bufs.random_int8.launch(
                        (bufs.n * bufs.k) as i32,
                        bufs.seed_tensor.ptr,
                        0,
                        bufs.b.ptr,
                        stream.handle,
                    )?;
                    bufs.tensor_hash.launch(
                        bufs.b.ptr,
                        bufs.n * bufs.k,
                        bufs.key_tensor.ptr,
                        bufs.b_tensor_hash.ptr,
                        stream.handle,
                    )?;
                }
                stream.synchronize()?;
                println!("{tag} job init: copy B to host RAM");
                let mut b_bytes = vec![0u8; bufs.n * bufs.k];
                bufs.b.copy_to(&mut b_bytes)?;
                println!("{tag} job init complete");
                if use_graphs && graphs.is_none() {
                    println!("{tag} capturing CUDA graphs");
                    graphs = Some(unsafe { bufs.capture_all_slots(&stream)? });
                    println!("{tag} captured CUDA graphs");
                }
                println!("{tag} loaded job={} height={}", job.job_id, job.height);
                current = Some(Arc::new(ReadyJob {
                    pool_job: job,
                    b_bytes,
                    m: bufs.m,
                    n: bufs.n,
                }));
            }
        }

        let ready = match &current {
            Some(j) => Arc::clone(j),
            None => {
                thread::sleep(Duration::from_millis(100));
                continue;
            }
        };
        let batch_start = iter_idx;
        // In eager compatibility mode process one slot at a time. This avoids
        // both the upstream mapped-host zero and an 8-iteration async burst,
        // which is unstable on the tested HiveOS/NVIDIA 580 stack.
        if use_graphs {
            for signal in &signal_dev_pool {
                signal.zero()?;
            }
        } else {
            signal_dev_pool[bufs.slot(iter_idx)].zero()?;
        }

        if !use_graphs && iter_idx == 0 {
            // Fully instrument the first iteration and synchronize after every
            // kernel so a driver crash can be localized to one exact stage.
            let slot = bufs.slot(iter_idx);
            println!("{tag} diag: signal VRAM clear OK");

            println!("{tag} diag: random A launch");
            unsafe { bufs.random_fill_a(iter_idx, slot, stream.handle)?; }
            stream.synchronize()?;
            println!("{tag} diag: random A OK");

            println!("{tag} diag: tensor hash A");
            unsafe {
                bufs.tensor_hash.launch(
                    bufs.a_pool[slot].ptr,
                    bufs.m * bufs.k,
                    bufs.key_tensor.ptr,
                    bufs.a_tensor_hash_pool[slot].ptr,
                    stream.handle,
                )?;
            }
            stream.synchronize()?;
            println!("{tag} diag: tensor hash A OK");

            println!("{tag} diag: commitment hash");
            unsafe {
                bufs.commitment_hash.launch(
                    bufs.a_tensor_hash_pool[slot].ptr,
                    bufs.b_tensor_hash.ptr,
                    bufs.key_tensor.ptr,
                    bufs.commit_a_pool[slot].ptr,
                    bufs.commit_b_pool[slot].ptr,
                    stream.handle,
                )?;
            }
            stream.synchronize()?;
            println!("{tag} diag: commitment hash OK");

            println!("{tag} diag: noise dense int8 A");
            unsafe {
                bufs.noise_gen.launch_dense_int8(
                    bufs.m as i32,
                    bufs.commit_a_pool[slot].ptr,
                    bufs.seed_label_a.ptr,
                    bufs.eal.ptr,
                    stream.handle,
                )?;
            }
            stream.synchronize()?;
            println!("{tag} diag: noise dense int8 A OK");

            println!("{tag} diag: noise dense int8 B");
            unsafe {
                bufs.noise_gen.launch_dense_int8(
                    bufs.n as i32,
                    bufs.commit_b_pool[slot].ptr,
                    bufs.seed_label_b.ptr,
                    bufs.ebr.ptr,
                    stream.handle,
                )?;
            }
            stream.synchronize()?;
            println!("{tag} diag: noise dense int8 B OK");

            println!("{tag} diag: noise dense fp16 A");
            unsafe {
                bufs.noise_gen.launch_dense_fp16(
                    bufs.m as i32,
                    bufs.commit_a_pool[slot].ptr,
                    bufs.seed_label_a.ptr,
                    1,
                    bufs.eal_fp16.ptr,
                    stream.handle,
                )?;
            }
            stream.synchronize()?;
            println!("{tag} diag: noise dense fp16 A OK");

            println!("{tag} diag: noise dense fp16 B");
            unsafe {
                bufs.noise_gen.launch_dense_fp16(
                    bufs.n as i32,
                    bufs.commit_b_pool[slot].ptr,
                    bufs.seed_label_b.ptr,
                    1,
                    bufs.ebr_fp16.ptr,
                    stream.handle,
                )?;
            }
            stream.synchronize()?;
            println!("{tag} diag: noise dense fp16 B OK");

            bufs.ear_r_major.zero()?;
            bufs.ebl_r_major.zero()?;
            println!("{tag} diag: sparse buffers zero OK");

            println!("{tag} diag: noise sparse A");
            unsafe {
                bufs.noise_gen.launch_sparse(
                    bufs.k as i32,
                    bufs.commit_a_pool[slot].ptr,
                    bufs.seed_label_a.ptr,
                    bufs.ear_r_major.ptr,
                    stream.handle,
                )?;
            }
            stream.synchronize()?;
            println!("{tag} diag: noise sparse A OK");

            println!("{tag} diag: noise sparse B");
            unsafe {
                bufs.noise_gen.launch_sparse(
                    bufs.k as i32,
                    bufs.commit_b_pool[slot].ptr,
                    bufs.seed_label_b.ptr,
                    bufs.ebl_r_major.ptr,
                    stream.handle,
                )?;
            }
            stream.synchronize()?;
            println!("{tag} diag: noise sparse B OK");

            println!("{tag} diag: transpose A");
            unsafe {
                bufs.noise_gen.launch_transpose(
                    bufs.k as i32,
                    bufs.r as i32,
                    bufs.ear_r_major.ptr,
                    bufs.ear_k_major.ptr,
                    stream.handle,
                )?;
            }
            stream.synchronize()?;
            println!("{tag} diag: transpose A OK");

            println!("{tag} diag: transpose B");
            unsafe {
                bufs.noise_gen.launch_transpose(
                    bufs.k as i32,
                    bufs.r as i32,
                    bufs.ebl_r_major.ptr,
                    bufs.ebl_k_major.ptr,
                    stream.handle,
                )?;
            }
            stream.synchronize()?;
            println!("{tag} diag: transpose B OK");

            if let Some(triton) = bufs.triton.as_ref() {
                println!("{tag} diag: triton noising A");
                unsafe {
                    triton.noising.launch(
                        bufs.m as i32,
                        bufs.k as i32,
                        bufs.r as i32,
                        bufs.a_pool[slot].ptr,
                        bufs.eal.ptr,
                        bufs.ear_r_major.ptr,
                        bufs.ap_ea.ptr,
                        stream.handle,
                    )?;
                }
                stream.synchronize()?;
                println!("{tag} diag: triton noising A OK");

                println!("{tag} diag: triton noising B");
                unsafe {
                    triton.noising.launch(
                        bufs.n as i32,
                        bufs.k as i32,
                        bufs.r as i32,
                        bufs.b.ptr,
                        bufs.ebr.ptr,
                        bufs.ebl_r_major.ptr,
                        bufs.bp_eb.ptr,
                        stream.handle,
                    )?;
                }
                stream.synchronize()?;
                println!("{tag} diag: triton noising B OK");

                triton.transcripts.zero()?;
                println!("{tag} diag: transcripts zero OK");

                println!("{tag} diag: triton search");
                unsafe {
                    triton.search.launch(
                        bufs.m as i32,
                        bufs.n as i32,
                        bufs.k as i32,
                        bufs.ap_ea.ptr,
                        bufs.bp_eb.ptr,
                        triton.transcripts.ptr,
                        stream.handle,
                    )?;
                }
                stream.synchronize()?;
                println!("{tag} diag: triton search OK");

                unsafe {
                    cu_check(
                        cu::cuMemsetD32Async(
                            bufs.pow_workspace_scan.ptr,
                            0xFFFFFFFFu32,
                            1,
                            stream.handle,
                        ),
                        "diag cuMemsetD32Async(scan)",
                    )?;
                }
                stream.synchronize()?;
                println!("{tag} diag: scan sentinel OK");

                let total_candidates =
                    triton.num_triton_tile_m * triton.num_triton_tile_n * 64;

                println!("{tag} diag: postpass blake3 compare");
                unsafe {
                    triton.postpass.launch_blake3_compare(
                        triton.transcripts.ptr,
                        bufs.commit_a_pool[slot].ptr,
                        bufs.pow_target_tensor.ptr,
                        bufs.pow_workspace_hash.ptr,
                        bufs.pow_workspace_hit.ptr,
                        total_candidates,
                        stream.handle,
                    )?;
                }
                stream.synchronize()?;
                println!("{tag} diag: postpass blake3 compare OK");

                println!("{tag} diag: postpass scan");
                unsafe {
                    triton.postpass.launch_scan(
                        bufs.pow_workspace_hit.ptr,
                        total_candidates,
                        bufs.pow_workspace_scan.ptr,
                        stream.handle,
                    )?;
                }
                stream.synchronize()?;
                println!("{tag} diag: postpass scan OK");

                println!("{tag} diag: postpass emit");
                unsafe {
                    triton.postpass.launch_emit(
                        bufs.pow_workspace_scan.ptr,
                        bufs.pow_target_tensor.ptr,
                        bufs.host_signal_header_pool[slot].device_ptr,
                        triton.num_triton_tile_m,
                        triton.num_triton_tile_n,
                        64,
                        bufs.m as i32,
                        bufs.n as i32,
                        bufs.k as i32,
                        128,
                        128,
                        64,
                        stream.handle,
                    )?;
                }
                stream.synchronize()?;
                println!("{tag} diag: postpass emit OK");
            } else {
                println!("{tag} diag: non-Triton path");
                unsafe { bufs.mine_one_post_random(slot, stream.handle)?; }
                stream.synchronize()?;
                println!("{tag} diag: non-Triton pipeline OK");
            }

            println!("{tag} diag: first mining iteration complete");
            iter_idx += 1;
        } else if use_graphs {
            let graphs_ref = graphs.as_mut().expect("graphs captured after job load");
            for _ in 0..bufs.ring_size as u64 {
                unsafe {
                    bufs.mine_one_with_graphs(iter_idx, graphs_ref, &stream)?;
                }
                iter_idx += 1;
            }
            stream.synchronize()?;
        } else {
            // Safe eager path: never call MinerBufs::mine_one(), because that
            // touches the upstream mapped pinned host-signal buffer. Launch a
            // single iteration, synchronize it, then read back its 1 KiB signal.
            let slot = bufs.slot(iter_idx);
            unsafe {
                bufs.random_fill_a(iter_idx, slot, stream.handle)?;
                bufs.mine_one_post_random(slot, stream.handle)?;
            }
            stream.synchronize()?;
            iter_idx += 1;
        }

        let mut hits = Vec::<(Vec<u8>, Vec<u8>)>::new();
        for i in batch_start..iter_idx {
            let slot = bufs.slot(i);
            let mut hdr = vec![0u8; HOST_SIGNAL_HEADER_SIZE];
            signal_dev_pool[slot].copy_to(&mut hdr)?;
            let status = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
            if status == 1 {
                hits_total += 1;
                // Hits are rare, so use a synchronous pageable-RAM copy instead
                // of the large DEVICEMAP pinned A snapshot.
                let mut a_bytes = vec![0u8; bufs.m * bufs.k];
                bufs.a_pool[slot].copy_to(&mut a_bytes)?;
                hits.push((hdr, a_bytes));
            }
        }
        for (hdr, a_bytes) in hits {
            let parsed = ParsedSignalHeader::parse(&hdr)?;
            let (a_rows, b_cols) = extract_indices(&parsed);
            let _ = w.hit_tx.send(HitWork {
                job: Arc::clone(&ready),
                a_bytes,
                a_rows,
                b_cols,
                src_gpu: w.device_ord,
            });
        }

        if w.max_iters > 0 && iter_idx >= w.max_iters {
            break;
        }
        if last_log.elapsed() >= Duration::from_secs(LOG_SECS) {
            let dt = last_log.elapsed().as_secs_f64();
            let d = iter_idx - iters_at_log;
            let ths = (d as f64 / dt) * macs_per_iter_t;
            w.gpu_ths_bits.store(ths.to_bits(), Ordering::Relaxed);
            println!("{tag} {:.2} TH/s iters={} hits={}", ths, iter_idx, hits_total);
            last_log = Instant::now();
            iters_at_log = iter_idx;
        }
    }
    Ok(())
}

fn run() -> Result<(), MinerError> {
    let pool_addr = std::env::var("PRL_POOL")
        .unwrap_or_else(|_| "prl-eu.kryptex.network:7048".to_string())
        .trim_start_matches("stratum+tcp://")
        .trim_start_matches("stratum://")
        .trim_start_matches("tcp://")
        .to_string();
    let wallet = std::env::var("PRL_WALLET")
        .map_err(|_| miner_err("config", "PRL_WALLET is required"))?;
    let worker_name = std::env::var("PRL_WORKER")
        .or_else(|_| std::env::var("WORKER_NAME"))
        .unwrap_or_else(|_| "hive".to_string());
    let pass = std::env::var("PRL_PASS").unwrap_or_else(|_| "x".to_string());
    let fatbin_path = std::env::var("PEARL_FATBIN")
        .unwrap_or_else(|_| "./pearl_gemm.fatbin".to_string());
    let max_iters = std::env::var("MAX_ITERS")
        .ok()
        .and_then(|x| x.parse::<u64>().ok())
        .unwrap_or(0);

    let devs = pick_devices()?;
    let cfg = pick_config();
    let fatbin = Arc::new(std::fs::read(&fatbin_path)?);
    println!("PRL0 Kryptex 0.1.6 | DEV FEE: 0.00%");
    println!("[miner] pool={pool_addr} worker={worker_name} GPUs={devs:?}");
    println!("[miner] fatbin={} ({} bytes)", fatbin_path, fatbin.len());

    let pool = Arc::new(SharedPool::new());
    {
        let p = Arc::clone(&pool);
        let a = pool_addr.clone();
        let wa = wallet.clone();
        let wo = worker_name.clone();
        let pa = pass.clone();
        thread::Builder::new()
            .name("stratum".into())
            .spawn(move || pool_thread(p, a, wa, wo, pa))
            .expect("spawn stratum");
    }

    let (hit_tx, hit_rx) = mpsc::channel::<HitWork>();
    {
        let p = Arc::clone(&pool);
        thread::Builder::new()
            .name("proof-submit".into())
            .spawn(move || hit_poller(hit_rx, p))
            .expect("spawn proof-submit");
    }

    let worker_salt = Arc::new(resolve_worker_salt());
    let mut handles = Vec::new();
    let mut rates = Vec::new();
    for &dev in &devs {
        let rate = Arc::new(AtomicU64::new(0f64.to_bits()));
        rates.push(Arc::clone(&rate));
        let wc = WorkerCtx {
            device_ord: dev,
            fatbin: Arc::clone(&fatbin),
            cfg,
            pool: Arc::clone(&pool),
            hit_tx: hit_tx.clone(),
            gpu_ths_bits: rate,
            max_iters,
            worker_salt: Arc::clone(&worker_salt),
        };
        handles.push((dev, thread::spawn(move || worker(wc))));
    }
    drop(hit_tx);

    let started = Instant::now();
    {
        let rates = rates.clone();
        let p = Arc::clone(&pool);
        thread::Builder::new()
            .name("hive-stats".into())
            .spawn(move || loop {
                thread::sleep(Duration::from_secs(LOG_SECS));
                let vals = rates
                    .iter()
                    .map(|a| f64::from_bits(a.load(Ordering::Relaxed)))
                    .collect::<Vec<_>>();
                let total: f64 = vals.iter().sum();
                let csv = vals
                    .iter()
                    .map(|v| format!("{v:.3}"))
                    .collect::<Vec<_>>()
                    .join(",");
                println!(
                    "[hive] gpu_ths={} total_ths={:.3} accepted={} rejected={} uptime={}",
                    csv,
                    total,
                    p.accepted.load(Ordering::Relaxed),
                    p.rejected.load(Ordering::Relaxed),
                    started.elapsed().as_secs()
                );
            })
            .expect("spawn stats");
    }

    let mut first_err = None;
    for (dev, h) in handles {
        match h.join() {
            Ok(Ok(())) => println!("[miner] gpu{dev} finished"),
            Ok(Err(e)) => {
                eprintln!("[miner] gpu{dev} error: {e}");
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
            Err(_) => eprintln!("[miner] gpu{dev} panicked"),
        }
    }
    if let Some(e) = first_err {
        Err(e)
    } else {
        Ok(())
    }
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("FATAL: {e}");
            std::process::ExitCode::from(1)
        }
    }
}
