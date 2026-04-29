// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Minimal HTTP/1.1 server with `Range` support and per-request counters.
//!
//! Bench-local: this module is only built into `tensogram-benchmarks` (a
//! `publish = false` crate), and serves `.tgm` fixtures off a `BTreeMap`
//! held in memory.  No `hyper`, no `tokio` — a `std::net::TcpListener`
//! thread-per-connection loop keeps the dep surface minimal.
//!
//! The per-request counters discriminate `GET` vs `HEAD` and whether
//! the request carried a `Range:` header, so the bench harness can
//! quote `total_requests`, `range_get_requests`, `head_requests`, and
//! `response_body_bytes` against the flip criterion.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct RequestRecord {
    pub method: String,
    pub path: String,
    pub had_range_header: bool,
    pub status: u16,
    pub response_bytes: usize,
}

#[derive(Debug, Default, Clone)]
pub struct Counters {
    pub records: Vec<RequestRecord>,
}

impl Counters {
    pub fn total_requests(&self) -> usize {
        self.records.len()
    }
    pub fn range_get_requests(&self) -> usize {
        self.records
            .iter()
            .filter(|r| r.method == "GET" && r.had_range_header)
            .count()
    }
    pub fn head_requests(&self) -> usize {
        self.records.iter().filter(|r| r.method == "HEAD").count()
    }
    pub fn response_body_bytes(&self) -> usize {
        self.records.iter().map(|r| r.response_bytes).sum()
    }
}

pub struct MockServer {
    base_url: String,
    counters: Arc<Mutex<Counters>>,
    shutdown: Arc<AtomicBool>,
}

impl MockServer {
    pub fn start(fixtures: BTreeMap<String, Arc<Vec<u8>>>) -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let port = listener.local_addr()?.port();
        let base_url = format!("http://127.0.0.1:{port}");
        let counters = Arc::new(Mutex::new(Counters::default()));
        let shutdown = Arc::new(AtomicBool::new(false));

        let counters_thread = counters.clone();
        let shutdown_thread = shutdown.clone();
        let fixtures_thread = Arc::new(fixtures);

        thread::spawn(move || {
            while !shutdown_thread.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        // The listener is non-blocking so the accept loop can
                        // honour `shutdown_thread`.  On macOS (unlike Linux),
                        // accepted streams inherit the listener's non-blocking
                        // flag, which makes `handle_one`'s blocking `read`
                        // return `WouldBlock` immediately and the worker drop
                        // the connection before sending any response.  Force
                        // the worker stream back to blocking so reads wait for
                        // the request bytes the way the handler expects.
                        // Surfacing the error is critical: silently swallowing
                        // it would let the same flake reappear if a future
                        // platform regresses, with no log to diagnose from.
                        if let Err(e) = stream.set_nonblocking(false) {
                            eprintln!(
                                "mock_http: set_nonblocking(false) failed on accepted stream: {e}; \
                                dropping connection"
                            );
                            continue;
                        }
                        let fixtures = fixtures_thread.clone();
                        let counters = counters_thread.clone();
                        thread::spawn(move || {
                            handle_one(stream, &fixtures, &counters);
                        });
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(_) => break,
                }
            }
        });

        Ok(MockServer {
            base_url,
            counters,
            shutdown,
        })
    }

    pub fn url_for(&self, fixture_name: &str) -> String {
        format!("{}/{}", self.base_url, fixture_name)
    }

    pub fn reset(&self) {
        self.counters
            .lock()
            .expect("MockServer counters mutex poisoned")
            .records
            .clear();
    }

    pub fn snapshot(&self) -> Counters {
        self.counters
            .lock()
            .expect("MockServer counters mutex poisoned")
            .clone()
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

fn handle_one(
    mut stream: TcpStream,
    fixtures: &BTreeMap<String, Arc<Vec<u8>>>,
    counters: &Arc<Mutex<Counters>>,
) {
    let mut buf = vec![0u8; 8192];
    let n = match stream.read(&mut buf) {
        Ok(0) => return,
        Ok(n) => n,
        Err(_) => return,
    };
    let request = String::from_utf8_lossy(&buf[..n]).to_string();
    let mut lines = request.lines();
    let request_line = match lines.next() {
        Some(l) => l,
        None => return,
    };
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let raw_path = parts.next().unwrap_or("").to_string();
    let range_header = lines
        .find(|l| l.to_ascii_lowercase().starts_with("range:"))
        .and_then(|l| l.split_once(':').map(|(_, v)| v.trim().to_string()));

    let path = raw_path.trim_start_matches('/').to_string();
    let data = fixtures.get(&path);

    // Build the full response into a single buffer, then commit the
    // counter record BEFORE writing.  The client returns as soon as
    // its `read()` sees the bytes; if we recorded after `write_all`,
    // a synchronous test that issues two GETs back-to-back could
    // observe the snapshot in between the second GET's response
    // arriving and our worker thread acquiring the counters mutex —
    // that race surfaced as a flaky `total_requests = 1` instead of
    // `2` in `counters_segregate_methods_and_ranges`.
    let (status, response_bytes, response_buf) = match (method.as_str(), data) {
        (_, None) => (404, 0, status_only(404)),
        ("HEAD", Some(data)) => (200, 0, head_response(data.len())),
        ("GET", Some(data)) => match range_header
            .as_deref()
            .and_then(|h| parse_range(h, data.len()))
        {
            Some((start, end)) => {
                let chunk = &data[start..end];
                (
                    206,
                    chunk.len(),
                    range_response(start, end, data.len(), chunk),
                )
            }
            None if range_header.is_some() => (416, 0, range_not_satisfiable(data.len())),
            None => (200, data.len(), full_get_response(data)),
        },
        _ => (405, 0, status_only(405)),
    };

    counters
        .lock()
        .expect("MockServer counters mutex poisoned")
        .records
        .push(RequestRecord {
            method,
            path: raw_path,
            had_range_header: range_header.is_some(),
            status,
            response_bytes,
        });

    let _ = stream.write_all(&response_buf);
}

fn status_only(status: u16) -> Vec<u8> {
    let phrase = match status {
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "OK",
    };
    format!("HTTP/1.1 {status} {phrase}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        .into_bytes()
}

fn head_response(total: usize) -> Vec<u8> {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {total}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n"
    )
    .into_bytes()
}

fn range_response(start: usize, end: usize, total: usize, chunk: &[u8]) -> Vec<u8> {
    let header = format!(
        "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {}-{}/{}\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
        start,
        end - 1,
        total,
        chunk.len(),
    );
    let mut buf = header.into_bytes();
    buf.extend_from_slice(chunk);
    buf
}

fn range_not_satisfiable(total: usize) -> Vec<u8> {
    format!(
        "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{total}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )
    .into_bytes()
}

fn full_get_response(data: &[u8]) -> Vec<u8> {
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
        data.len()
    );
    let mut buf = header.into_bytes();
    buf.extend_from_slice(data);
    buf
}

fn parse_range(header: &str, total: usize) -> Option<(usize, usize)> {
    let spec = header.strip_prefix("bytes=")?;
    if total == 0 {
        return None;
    }
    if let Some(suffix) = spec.strip_prefix('-') {
        let n: usize = suffix.parse().ok()?;
        if n == 0 {
            return None;
        }
        return Some((total.saturating_sub(n), total));
    }
    let (start_s, end_s) = spec.split_once('-')?;
    let start: usize = start_s.parse().ok()?;
    if start >= total {
        return None;
    }
    if end_s.is_empty() {
        return Some((start, total));
    }
    let end: usize = end_s.parse().ok()?;
    if end < start {
        return None;
    }
    Some((start, end.min(total - 1) + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str, data: Vec<u8>) -> BTreeMap<String, Arc<Vec<u8>>> {
        let mut map = BTreeMap::new();
        map.insert(name.to_string(), Arc::new(data));
        map
    }

    #[test]
    fn counters_segregate_methods_and_ranges() {
        let server = MockServer::start(fixture("a.bin", vec![1, 2, 3, 4, 5])).expect("start");
        let url = server.url_for("a.bin");

        let agent = ureq_like_get(&url, None);
        assert_eq!(agent.status, 200);
        assert_eq!(agent.body.len(), 5);

        let agent = ureq_like_get(&url, Some("bytes=0-2"));
        assert_eq!(agent.status, 206);
        assert_eq!(agent.body.len(), 3);

        let snap = server.snapshot();
        assert_eq!(snap.total_requests(), 2);
        assert_eq!(snap.range_get_requests(), 1);
        assert_eq!(snap.head_requests(), 0);
        assert_eq!(snap.response_body_bytes(), 5 + 3);
    }

    #[test]
    fn parse_range_accepts_suffix_and_explicit() {
        assert_eq!(parse_range("bytes=0-9", 100), Some((0, 10)));
        assert_eq!(parse_range("bytes=10-", 100), Some((10, 100)));
        assert_eq!(parse_range("bytes=-5", 100), Some((95, 100)));
        assert_eq!(parse_range("bytes=200-300", 100), None);
        assert_eq!(parse_range("malformed", 100), None);
        assert_eq!(parse_range("bytes=0-9", 0), None);
    }

    struct Reply {
        status: u16,
        body: Vec<u8>,
    }

    fn ureq_like_get(url: &str, range: Option<&str>) -> Reply {
        use std::io::{BufRead, BufReader};
        let url = url.strip_prefix("http://").unwrap();
        let (host_port, path) = url.split_once('/').unwrap();
        let mut sock = TcpStream::connect(host_port).expect("connect");
        let mut req = format!("GET /{path} HTTP/1.1\r\nHost: {host_port}\r\nConnection: close\r\n");
        if let Some(r) = range {
            req.push_str(&format!("Range: {r}\r\n"));
        }
        req.push_str("\r\n");
        sock.write_all(req.as_bytes()).expect("write");
        let mut reader = BufReader::new(sock);
        let mut status_line = String::new();
        reader.read_line(&mut status_line).expect("status");
        let status: u16 = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .expect("status code");
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).expect("header");
            if line == "\r\n" || line.is_empty() {
                break;
            }
            if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            reader.read_exact(&mut body).expect("body");
        }
        Reply { status, body }
    }
}
