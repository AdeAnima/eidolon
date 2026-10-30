use std::collections::{HashMap, VecDeque};
use std::io::{Cursor, Read};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tiny_http::{Header, Method, Response, Server, StatusCode};

#[derive(Clone)]
pub struct Resource {
    pub bytes: Vec<u8>,
    pub mime: String,
    pub etag: Option<String>,
    pub modified: Option<String>,
    pub status: u16,
    pub location: Option<String>,
    pub ignore_range: bool,
    pub unknown_length: bool,
    pub reject_head: bool,
    pub cache_control: Option<String>,
    pub age: Option<u64>,
    pub failures: VecDeque<(u16, String)>,
    pub failure_method: Method,
}

impl Resource {
    pub fn new(bytes: Vec<u8>, mime: &str) -> Self {
        Self {
            bytes,
            mime: mime.to_owned(),
            etag: Some("\"fixture-v1\"".to_owned()),
            modified: Some("Wed, 21 Oct 2015 07:28:00 GMT".to_owned()),
            status: 200,
            location: None,
            ignore_range: false,
            unknown_length: false,
            reject_head: false,
            cache_control: Some("no-cache".to_owned()),
            age: None,
            failures: VecDeque::new(),
            failure_method: Method::Head,
        }
    }
}

pub struct TestServer {
    pub base: String,
    pub routes: Arc<Mutex<HashMap<String, Resource>>>,
    pub bytes: Arc<AtomicU64>,
    pub ranges: Arc<AtomicU64>,
    pub revalidations: Arc<AtomicU64>,
    pub requests: Arc<Mutex<Vec<(String, String)>>>,
    stop: Arc<AtomicBool>,
    server: Arc<Server>,
    worker: Option<JoinHandle<()>>,
}

impl TestServer {
    pub fn new() -> Self {
        let server = Arc::new(Server::http("127.0.0.1:0").expect("bind test HTTP server"));
        let base = format!("http://{}", server.server_addr());
        let routes = Arc::new(Mutex::new(HashMap::<String, Resource>::new()));
        let bytes = Arc::new(AtomicU64::new(0));
        let ranges = Arc::new(AtomicU64::new(0));
        let revalidations = Arc::new(AtomicU64::new(0));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let worker = {
            let (server, routes, bytes, ranges, revalidations, requests, stop) = (
                Arc::clone(&server),
                Arc::clone(&routes),
                Arc::clone(&bytes),
                Arc::clone(&ranges),
                Arc::clone(&revalidations),
                Arc::clone(&requests),
                Arc::clone(&stop),
            );
            thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let Ok(Some(request)) = server.recv_timeout(Duration::from_millis(50)) else {
                        continue;
                    };
                    let method = request.method().as_str().to_owned();
                    requests
                        .lock()
                        .expect("request log")
                        .push((method, request.url().to_owned()));
                    respond(request, &routes, &bytes, &ranges, &revalidations);
                }
            })
        };
        Self {
            base,
            routes,
            bytes,
            ranges,
            revalidations,
            requests,
            stop,
            server,
            worker: Some(worker),
        }
    }

    pub fn add(&self, path: &str, resource: Resource) {
        self.routes
            .lock()
            .expect("routes")
            .insert(path.to_owned(), resource);
    }

    pub fn served(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.server.unblock();
        if let Some(worker) = self.worker.take() {
            worker.join().expect("HTTP fixture worker");
        }
    }
}

fn respond(
    request: tiny_http::Request,
    routes: &Mutex<HashMap<String, Resource>>,
    bytes: &Arc<AtomicU64>,
    ranges: &AtomicU64,
    revalidations: &AtomicU64,
) {
    let header = |name: &'static str| {
        request
            .headers()
            .iter()
            .find(|header| header.field.equiv(name))
            .map(|header| header.value.as_str())
    };
    assert!(header("User-Agent").is_some_and(|agent| agent.starts_with("eidolon/")));
    assert!(header("Cookie").is_none());
    assert!(header("Authorization").is_none());
    let resource = {
        let mut routes = routes.lock().expect("routes");
        let Some(resource) = routes.get_mut(request.url()) else {
            let _ = request.respond(Response::empty(StatusCode(404)));
            return;
        };
        if request.method() == &resource.failure_method
            && let Some((status, delay)) = resource.failures.pop_front()
        {
            drop(routes);
            let response =
                Response::empty(StatusCode(status)).with_header(make_header("Retry-After", &delay));
            let _ = request.respond(response);
            return;
        }
        resource.clone()
    };
    if resource.reject_head && request.method() == &Method::Head {
        let _ = request.respond(Response::empty(StatusCode(405)));
        return;
    }
    let conditional = resource
        .etag
        .as_deref()
        .is_some_and(|etag| header("If-None-Match") == Some(etag))
        || (header("If-None-Match").is_none()
            && resource
                .modified
                .as_deref()
                .is_some_and(|date| header("If-Modified-Since") == Some(date)));
    if conditional {
        revalidations.fetch_add(1, Ordering::Relaxed);
        let _ = request.respond(Response::empty(StatusCode(304)));
        return;
    }
    let mut headers = vec![
        make_header("Content-Type", &resource.mime),
        make_header("Accept-Ranges", "bytes"),
    ];
    if let Some(value) = &resource.etag {
        headers.push(make_header("ETag", value));
    }
    if let Some(value) = &resource.modified {
        headers.push(make_header("Last-Modified", value));
    }
    if let Some(value) = &resource.cache_control {
        headers.push(make_header("Cache-Control", value));
    }
    if let Some(age) = resource.age {
        headers.push(make_header("Age", &age.to_string()));
    }
    if let Some(value) = &resource.location {
        headers.push(make_header("Location", value));
    }
    let mut status = resource.status;
    let length = resource.bytes.len();
    let mut body = resource.bytes;
    if request.method() == &Method::Get
        && !resource.ignore_range
        && let Some(range) = header("Range")
    {
        if let Some((start, end)) = parse_range(range, length) {
            ranges.fetch_add(1, Ordering::Relaxed);
            status = 206;
            headers.push(make_header(
                "Content-Range",
                &format!("bytes {start}-{end}/{length}"),
            ));
            body = body[start..=end].to_vec();
        } else {
            let _ = request.respond(Response::empty(StatusCode(416)));
            return;
        }
    }
    let body_length = (!resource.unknown_length).then_some(body.len());
    let reader = Counted {
        inner: Cursor::new(body),
        bytes: Arc::clone(bytes),
    };
    // tiny_http otherwise switches known >32 KiB bodies (including HEAD) to
    // chunked transfer, hiding the range origin's actual representation size.
    let response = Response::new(StatusCode(status), headers, reader, body_length, None)
        .with_chunked_threshold(usize::MAX);
    let _ = request.respond(response);
}

fn make_header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("valid fixture header")
}

fn parse_range(value: &str, size: usize) -> Option<(usize, usize)> {
    let (start, end) = value.strip_prefix("bytes=")?.split_once('-')?;
    if start.is_empty() {
        let length = end.parse::<usize>().ok()?;
        (length > 0 && size > 0).then_some((size.saturating_sub(length), size - 1))
    } else {
        let start = start.parse::<usize>().ok()?;
        let end = if end.is_empty() {
            size.checked_sub(1)?
        } else {
            end.parse::<usize>().ok()?.min(size.checked_sub(1)?)
        };
        (start <= end).then_some((start, end))
    }
}

struct Counted {
    inner: Cursor<Vec<u8>>,
    bytes: Arc<AtomicU64>,
}

impl Read for Counted {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = self.inner.read(buffer)?;
        self.bytes.fetch_add(count as u64, Ordering::Relaxed);
        Ok(count)
    }
}
