//! Public resolver tests against a real, byte-counted HTTP origin.

mod support;

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::Ordering;
use std::time::Duration;

use eidolon_fetch::{FetchOptions, Fetcher, VideoMode};
use image::{DynamicImage, ImageFormat, RgbImage};
use support::{Resource, TestServer};

fn image_bytes() -> Vec<u8> {
    let image = RgbImage::from_fn(48, 32, |x, y| {
        image::Rgb([(x * 5) as u8, (y * 7) as u8, (x * y) as u8])
    });
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(image)
        .write_to(&mut bytes, ImageFormat::Png)
        .expect("encode PNG fixture");
    bytes.into_inner()
}

fn fetcher(cache: &Path) -> Fetcher {
    Fetcher::with_options(FetchOptions {
        cache_dir: cache.to_owned(),
        timeout: Duration::from_secs(10),
        ..FetchOptions::default()
    })
    .expect("fetch client")
}

#[test]
fn fractional_refinement_limit_rejects_invalid_values_before_creating_cache() {
    let directory = tempfile::tempdir().expect("fraction validation");
    for (index, fraction) in [0.0, -0.5, 1.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
        .into_iter()
        .enumerate()
    {
        let cache = directory.path().join(index.to_string());
        let result = Fetcher::with_options(FetchOptions {
            cache_dir: cache.clone(),
            max_fetch_fraction: fraction,
            ..FetchOptions::default()
        });
        assert!(result.is_err(), "invalid fraction {fraction} was accepted");
        assert!(!cache.exists(), "invalid options must not create a cache");
    }
}

#[test]
fn image_download_is_complete_and_cached_response_revalidates_without_body_bytes() {
    let server = TestServer::new();
    let image = image_bytes();
    server.add("/image.png", Resource::new(image.clone(), "image/png"));
    let cache = tempfile::tempdir().expect("cache");
    let fetcher = fetcher(cache.path());
    assert!(server.requests.lock().expect("requests").is_empty());
    let result = fetcher
        .resolve(&format!("{}/image.png", server.base))
        .expect("fetch image");
    assert_eq!(result.artifacts.len(), 1);
    let remote = &result.artifacts[0];
    assert_eq!(
        std::fs::read(&remote.local_path).expect("cached image"),
        image
    );
    assert_eq!(
        image::open(&remote.local_path)
            .expect("decode cached image")
            .width(),
        48
    );
    assert!(remote.complete_bytes);
    assert_eq!(remote.mode, eidolon_fetch::FetchMode::Full);
    assert_eq!(remote.mode_reason, "complete_resource");
    assert_eq!(
        serde_json::to_value(remote).expect("provenance JSON")["mode"],
        "full"
    );
    assert_eq!(remote.role, "direct");
    assert_eq!(result.bytes_fetched, image.len() as u64);
    assert_eq!(server.served(), result.bytes_fetched);
    assert_eq!(result.total_size, Some(image.len() as u64));
    let cached = fetcher
        .resolve(&format!("{}/image.png", server.base))
        .expect("revalidate image");
    assert_eq!(cached.bytes_fetched, 0);
    assert_eq!(cached.artifacts[0].local_path, remote.local_path);
    assert_eq!(cached.artifacts[0].mode, remote.mode);
    assert_eq!(cached.artifacts[0].mode_reason, remote.mode_reason);
    assert_eq!(server.revalidations.load(Ordering::Relaxed), 1);
    assert_eq!(fetcher.bytes_fetched(), image.len() as u64);
    eprintln!(
        "MEASURE image: fetched={} served={} total={} ratio=1.000000",
        result.bytes_fetched,
        server.served(),
        image.len()
    );
}

#[test]
fn freshness_window_is_opt_in_and_304_extends_it() {
    for policy in [None, Some("public, max-age=600"), Some("no-cache")] {
        for fresh_enabled in [false, true] {
            let server = TestServer::new();
            let mut resource = Resource::new(image_bytes(), "image/png");
            resource.cache_control = policy.map(str::to_owned);
            server.add("/cached.png", resource);
            let cache = tempfile::tempdir().expect("cache");
            let client = || {
                Fetcher::with_options(FetchOptions {
                    cache_dir: cache.path().to_owned(),
                    fresh: fresh_enabled,
                    ..FetchOptions::default()
                })
                .expect("fetch client")
            };
            let url = format!("{}/cached.png", server.base);
            let first = client().resolve(&url).expect("initial source");
            let metadata: serde_json::Value = serde_json::from_slice(
                &std::fs::read(first.artifacts[0].local_path.with_extension("json"))
                    .expect("metadata"),
            )
            .expect("cached metadata");
            assert_eq!(metadata["version"], 4);
            let expected_lifetime = match policy {
                None => 30,
                Some("public, max-age=600") => 600,
                _ => 0,
            };
            assert_eq!(metadata["headers"]["fresh_for"], expected_lifetime);
            assert!(
                metadata["headers"]["fresh_until"]
                    .as_u64()
                    .is_some_and(|until| until > 0)
            );
            let requests = server.requests.lock().expect("requests").len();
            let second = client().resolve(&url).expect("cached source");
            assert_eq!(second.bytes_fetched, 0);
            assert_eq!(
                second.artifacts[0].local_path,
                first.artifacts[0].local_path
            );
            let fresh = fresh_enabled && policy != Some("no-cache");
            let expected_heads = usize::from(!fresh);
            assert_eq!(
                server.requests.lock().expect("requests")[requests..],
                vec![("HEAD".to_owned(), "/cached.png".to_owned()); expected_heads]
            );
            assert_eq!(
                server.revalidations.load(Ordering::Relaxed),
                expected_heads as u64
            );
            if fresh {
                let metadata_path = first.artifacts[0].local_path.with_extension("json");
                let mut metadata: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&metadata_path).expect("metadata"))
                        .expect("cached metadata");
                metadata["headers"]["fresh_until"] = 0.into();
                std::fs::write(&metadata_path, serde_json::to_vec(&metadata).expect("JSON"))
                    .expect("expire metadata");
                client().resolve(&url).expect("304 revalidation");
                assert_eq!(server.revalidations.load(Ordering::Relaxed), 1);
                client().resolve(&url).expect("extended freshness");
                assert_eq!(server.revalidations.load(Ordering::Relaxed), 1);
                // Default policy still revalidates a cache populated with an enabled window.
                fetcher(cache.path())
                    .resolve(&url)
                    .expect("default revalidation");
                assert_eq!(server.revalidations.load(Ordering::Relaxed), 2);
            }
        }
    }
}

#[test]
fn shared_cache_age_and_no_cache_directives_force_revalidation() {
    for (age, control) in [
        (Some(601), "public, max-age=600"),
        (None, "no-cache"),
        (None, "no-store"),
        (None, "max-age=0"),
    ] {
        let server = TestServer::new();
        let mut resource = Resource::new(image_bytes(), "image/png");
        resource.cache_control = Some(control.to_owned());
        resource.age = age;
        server.add("/cached.png", resource);
        let cache = tempfile::tempdir().expect("cache");
        let client = Fetcher::with_options(FetchOptions {
            cache_dir: cache.path().to_owned(),
            fresh: true,
            ..FetchOptions::default()
        })
        .expect("freshness-enabled client");
        let url = format!("{}/cached.png", server.base);
        client.resolve(&url).expect("initial source");
        let requests = server.requests.lock().expect("requests").len();
        let cached = client.resolve(&url).expect("expired cache revalidation");
        assert_eq!(cached.bytes_fetched, 0);
        assert_eq!(
            server.requests.lock().expect("requests")[requests..],
            [("HEAD".to_owned(), "/cached.png".to_owned())],
            "policy {control}, Age {age:?} must not bypass validation",
        );
        assert_eq!(server.revalidations.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn old_cache_schema_is_invalidated() {
    let server = TestServer::new();
    server.add("/cached.png", Resource::new(image_bytes(), "image/png"));
    let cache = tempfile::tempdir().expect("cache");
    let url = format!("{}/cached.png", server.base);
    let first = fetcher(cache.path()).resolve(&url).expect("initial source");
    let metadata_path = first.artifacts[0].local_path.with_extension("json");
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&metadata_path).expect("metadata"))
            .expect("cached metadata");
    metadata["version"] = 3.into();
    metadata["headers"]["fresh_for"] = 600.into();
    metadata["headers"]["fresh_until"] = u64::MAX.into();
    std::fs::write(
        &metadata_path,
        serde_json::to_vec(&metadata).expect("metadata JSON"),
    )
    .expect("old cache metadata");
    let second = Fetcher::with_options(FetchOptions {
        cache_dir: cache.path().to_owned(),
        fresh: true,
        ..FetchOptions::default()
    })
    .expect("freshness-enabled client")
    .resolve(&url)
    .expect("new acquisition");
    assert_eq!(second.bytes_fetched, image_bytes().len() as u64);
    assert_eq!(server.revalidations.load(Ordering::Relaxed), 0);
}

#[test]
fn rate_limited_heads_and_gets_retry_at_most_twice() {
    for status in [429, 503] {
        for method in [tiny_http::Method::Head, tiny_http::Method::Get] {
            let server = TestServer::new();
            let mut resource = Resource::new(image_bytes(), "image/png");
            resource.failure_method = method;
            resource.failures = [(status, "0".to_owned()), (status, "0".to_owned())].into();
            server.add("/limited.png", resource);
            let cache = tempfile::tempdir().expect("cache");
            let result = fetcher(cache.path())
                .resolve(&format!("{}/limited.png", server.base))
                .expect("third request succeeds");
            assert_eq!(result.bytes_fetched, image_bytes().len() as u64);
            assert_eq!(server.requests.lock().expect("requests").len(), 4);
        }
    }
    let server = TestServer::new();
    let mut resource = Resource::new(image_bytes(), "image/png");
    resource.failures = std::iter::repeat_n((429, "0".to_owned()), 8).collect();
    server.add("/exhausted.png", resource);
    let cache = tempfile::tempdir().expect("cache");
    assert!(
        fetcher(cache.path())
            .resolve(&format!("{}/exhausted.png", server.base))
            .is_err()
    );
    assert_eq!(server.requests.lock().expect("requests").len(), 3);
}

#[test]
fn retry_after_waits_before_retrying() {
    let server = TestServer::new();
    let mut resource = Resource::new(image_bytes(), "image/png");
    resource.failures.push_back((503, "1".to_owned()));
    server.add("/delayed.png", resource);
    let cache = tempfile::tempdir().expect("cache");
    let start = std::time::Instant::now();
    fetcher(cache.path())
        .resolve(&format!("{}/delayed.png", server.base))
        .expect("retried source");
    assert!(start.elapsed() >= Duration::from_secs(1));
    assert_eq!(server.requests.lock().expect("requests").len(), 3);
}

#[test]
fn concurrent_fetchers_limit_each_host_to_two_inflight_requests() {
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let server = Arc::new(tiny_http::Server::http("127.0.0.1:0").expect("concurrent origin"));
    let url = format!("http://{}/image.png", server.server_addr());
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicUsize::new(0));
    let active = Arc::new(AtomicUsize::new(0));
    let origin = {
        let (server, stop, peak, active) = (
            Arc::clone(&server),
            Arc::clone(&stop),
            Arc::clone(&peak),
            Arc::clone(&active),
        );
        thread::spawn(move || {
            let mut handlers = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                let Ok(Some(request)) = server.recv_timeout(Duration::from_millis(20)) else {
                    continue;
                };
                let (active, peak) = (Arc::clone(&active), Arc::clone(&peak));
                handlers.push(thread::spawn(move || {
                    let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(count, Ordering::SeqCst);
                    thread::sleep(Duration::from_millis(80));
                    active.fetch_sub(1, Ordering::SeqCst);
                    let response = tiny_http::Response::from_data(image_bytes()).with_header(
                        tiny_http::Header::from_bytes("Content-Type", "image/png").expect("header"),
                    );
                    request.respond(response).expect("response");
                }));
            }
            for handler in handlers {
                handler.join().expect("origin handler");
            }
        })
    };
    let cache = tempfile::tempdir().expect("cache");
    let barrier = Arc::new(Barrier::new(7));
    let results = thread::scope(|scope| {
        let workers: Vec<_> = (0..6)
            .map(|index| {
                let fetcher = fetcher(&cache.path().join(index.to_string()));
                let (url, barrier) = (&url, Arc::clone(&barrier));
                scope.spawn(move || {
                    barrier.wait();
                    fetcher.resolve(url)
                })
            })
            .collect();
        barrier.wait();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("fetch worker"))
            .collect::<Vec<_>>()
    });
    stop.store(true, Ordering::Relaxed);
    server.unblock();
    origin.join().expect("origin");
    assert!(
        results.iter().all(Result::is_ok),
        "all concurrent inputs resolve"
    );
    assert_eq!(
        peak.load(Ordering::SeqCst),
        2,
        "origin requests must stay bounded, without serializing them"
    );
}

#[test]
fn changed_etag_and_last_modified_only_caches_are_revalidated() {
    let server = TestServer::new();
    let mut first = Resource::new(b"First article text".to_vec(), "text/plain");
    first.etag = None;
    server.add("/text", first.clone());
    let cache = tempfile::tempdir().expect("cache");
    let fetcher = fetcher(cache.path());
    let url = format!("{}/text", server.base);
    fetcher.resolve(&url).expect("initial text");
    assert_eq!(
        fetcher
            .resolve(&url)
            .expect("last-modified revalidation")
            .bytes_fetched,
        0
    );
    first.modified = Some("Thu, 22 Oct 2015 07:28:00 GMT".to_owned());
    first.bytes = b"A new article revision".to_vec();
    server.add("/text", first);
    let updated = fetcher.resolve(&url).expect("changed source");
    assert_eq!(
        std::fs::read_to_string(&updated.artifacts[0].local_path).expect("updated text"),
        "A new article revision"
    );
    let mut resource = Resource::new(b"An ETag revision".to_vec(), "text/plain");
    resource.etag = Some("\"second\"".to_owned());
    server.add("/text", resource);
    assert!(fetcher.resolve(&url).expect("changed ETag").bytes_fetched > 0);
}

#[test]
fn html_discovers_images_and_article_while_failed_video_urls_warn() {
    let server = TestServer::new();
    let image = image_bytes();
    for name in ["cover.png", "card.png", "preview.png", "poster.png"] {
        server.add(
            &format!("/{name}"),
            Resource::new(image.clone(), "image/png"),
        );
    }
    let html =
        include_str!("fixtures/article.html").replace("https://images.example.test", &server.base);
    let page_size = html.len() as u64;
    server.add(
        "/article",
        Resource::new(html.into_bytes(), "text/html; charset=utf-8"),
    );
    let cache = tempfile::tempdir().expect("cache");
    let result = fetcher(cache.path())
        .resolve(&format!("{}/article", server.base))
        .expect("resolve article");
    assert_eq!(result.artifacts.len(), 5);
    assert_eq!(result.warnings.len(), 4);
    let text = result
        .artifacts
        .iter()
        .find(|artifact| artifact.role == "article text")
        .expect("article artifact");
    assert!(!text.complete_bytes);
    assert_eq!(text.mode, eidolon_fetch::FetchMode::Derived);
    assert_eq!(text.mode_reason, "article_text_extraction");
    let text = std::fs::read_to_string(&text.local_path).expect("article text");
    assert!(text.contains("A patient observer"));
    assert!(!text.contains("Subscribe"));
    assert_eq!(result.bytes_fetched, page_size + 4 * image.len() as u64);
    assert_eq!(server.served(), result.bytes_fetched);
    assert_eq!(result.total_size, None);
    assert!(
        result
            .artifacts
            .iter()
            .all(|artifact| artifact.page_url.as_deref()
                == Some(format!("{}/article", server.base).as_str()))
    );
    eprintln!(
        "MEASURE html+4-images: fetched={} served={} total=unknown ratio=unknown",
        result.bytes_fetched,
        server.served()
    );
}

#[test]
fn failed_page_media_retains_known_sizes_and_counts_consumed_bytes() {
    for (unknown_length, reject_head, consumed) in
        [(false, false, 0), (false, true, 4), (true, false, 5)]
    {
        let server = TestServer::new();
        let html = "<html><head><meta property=\"og:image\" content=\"/large.png\"></head>\
                    <body><article><p>A useful article remains available.</p></article></body></html>";
        server.add(
            "/article",
            Resource::new(html.as_bytes().to_vec(), "text/html"),
        );
        let mut image = Resource::new(vec![0; 10], "image/png");
        image.unknown_length = unknown_length;
        image.reject_head = reject_head;
        server.add("/large.png", image);
        let cache = tempfile::tempdir().expect("cache");
        let fetcher = Fetcher::with_options(FetchOptions {
            cache_dir: cache.path().to_owned(),
            max_image_bytes: 4,
            ..FetchOptions::default()
        })
        .expect("fetch client");
        let result = fetcher
            .resolve(&format!("{}/article", server.base))
            .expect("article survives failed image");
        assert_eq!(result.artifacts.len(), 1);
        assert_eq!(result.warnings.len(), 1);
        assert_eq!(
            result.total_size,
            (!unknown_length).then_some(html.len() as u64 + 10)
        );
        assert_eq!(result.bytes_fetched, html.len() as u64 + consumed);
        assert_eq!(fetcher.bytes_fetched(), result.bytes_fetched);
    }
}

#[test]
fn redirects_retain_actual_url_and_reject_unsafe_targets() {
    let server = TestServer::new();
    server.add("/image.png", Resource::new(image_bytes(), "image/png"));
    let mut redirect = Resource::new(Vec::new(), "text/plain");
    redirect.status = 302;
    redirect.location = Some(format!("{}/image.png", server.base));
    server.add("/redirect", redirect.clone());
    let cache = tempfile::tempdir().expect("cache");
    let fetcher = fetcher(cache.path());
    let result = fetcher
        .resolve(&format!("{}/redirect", server.base))
        .expect("redirect");
    assert_eq!(
        result.artifacts[0].url,
        format!("{}/image.png", server.base)
    );
    redirect.location = Some("file:///etc/passwd".to_owned());
    server.add("/unsafe", redirect);
    assert!(fetcher.resolve(&format!("{}/unsafe", server.base)).is_err());
    for url in [
        "file:///etc/passwd",
        "ftp://example.test/image.png",
        "https://user:pass@example.test/image",
    ] {
        assert!(fetcher.resolve(url).is_err());
    }
}

#[test]
fn missing_oversized_and_chunked_oversized_inputs_fail_without_poisoning_other_urls() {
    let server = TestServer::new();
    server.add("/large.png", Resource::new(image_bytes(), "image/png"));
    let mut chunked = Resource::new(vec![42; 2048], "image/png");
    chunked.unknown_length = true;
    server.add("/chunked.png", chunked);
    server.add(
        "/small.txt",
        Resource::new(b"Still works".to_vec(), "text/plain"),
    );
    let cache = tempfile::tempdir().expect("cache");
    let fetcher = Fetcher::with_options(FetchOptions {
        cache_dir: cache.path().to_owned(),
        max_image_bytes: 32,
        ..FetchOptions::default()
    })
    .expect("client");
    assert!(
        fetcher
            .resolve(&format!("{}/missing", server.base))
            .expect_err("404")
            .to_string()
            .contains("404")
    );
    assert!(
        fetcher
            .resolve(&format!("{}/large.png", server.base))
            .expect_err("size cap")
            .to_string()
            .contains("size cap")
    );
    assert_eq!(fetcher.bytes_fetched(), 0);
    assert!(
        fetcher
            .resolve(&format!("{}/chunked.png", server.base))
            .is_err()
    );
    assert!(fetcher.bytes_fetched() > 0);
    assert_eq!(
        fetcher
            .resolve(&format!("{}/small.txt", server.base))
            .expect("other URL continues")
            .artifacts
            .len(),
        1
    );
}

#[test]
fn head_rejection_uses_a_bounded_probe() {
    let server = TestServer::new();
    let mut resource = Resource::new(image_bytes(), "image/png");
    resource.reject_head = true;
    server.add("/image.png", resource);
    let cache = tempfile::tempdir().expect("cache");
    let result = fetcher(cache.path())
        .resolve(&format!("{}/image.png", server.base))
        .expect("HEAD fallback");
    assert_eq!(result.artifacts.len(), 1);
    assert!(result.bytes_fetched > result.total_size.expect("source size"));
    assert_eq!(result.bytes_fetched, server.served());
}

fn ffmpeg(arguments: &[&str], output: &Path) {
    let result = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-threads",
            "1",
        ])
        .args(arguments)
        .arg(output)
        .output()
        .expect("installed ffmpeg");
    assert!(
        result.status.success(),
        "ffmpeg fixture: {}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn decode(path: &Path, keyframes: bool) -> Vec<u8> {
    let mut command = Command::new("ffmpeg");
    command.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-threads",
        "1",
    ]);
    if keyframes {
        command.args(["-skip_frame", "nokey"]);
    }
    command.arg("-i").arg(path).args([
        "-an",
        "-fps_mode",
        "passthrough",
        "-vf",
        "scale=96:96:flags=area",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]);
    let output = command.output().expect("decode video");
    assert!(
        output.status.success(),
        "decode: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn assert_source_timeline(proxy: &Path) {
    let metadata: serde_json::Value = serde_json::from_slice(
        &std::fs::read(PathBuf::from(format!("{}.remote.json", proxy.display())))
            .expect("source timeline sidecar"),
    )
    .expect("metadata JSON");
    assert_eq!(metadata["duration_seconds"], 64.0);
    assert_eq!(metadata["source_fps"], 24.0);
    let points: Vec<_> = metadata["sample_pts_seconds"]
        .as_array()
        .expect("source PTS")
        .iter()
        .map(|value| value.as_f64().expect("PTS number"))
        .collect();
    assert_eq!(points.len(), 16);
    for (index, point) in points.iter().enumerate() {
        assert!((point - index as f64 * 4.0).abs() < 0.000_001);
    }
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
        ])
        .arg(proxy)
        .output()
        .expect("probe proxy PTS");
    assert!(probe.status.success());
    let probe: serde_json::Value = serde_json::from_slice(&probe.stdout).expect("probe JSON");
    let actual: Vec<_> = probe["frames"]
        .as_array()
        .expect("decoded frames")
        .iter()
        .map(|frame| {
            frame["best_effort_timestamp_time"]
                .as_str()
                .expect("PTS string")
                .parse::<f64>()
                .expect("PTS value")
        })
        .collect();
    assert_eq!(actual, points);
}

fn assert_unknown_video_budget(server: &TestServer, source: &[u8], cache: &Path) {
    server.add(
        "/opaque-video",
        Resource::new(source.to_vec(), "application/octet-stream"),
    );
    let get_count = || {
        server
            .requests
            .lock()
            .expect("request log")
            .iter()
            .filter(|(method, _)| method == "GET")
            .count() as u64
    };
    let gets_before = get_count();
    let ranges_before = server.ranges.load(Ordering::Relaxed);
    let cap = 64 * 1024;
    let fetcher = Fetcher::with_options(FetchOptions {
        cache_dir: cache.to_owned(),
        max_video_bytes: cap,
        ..FetchOptions::default()
    })
    .expect("bounded video client");
    assert!(
        fetcher
            .resolve(&format!("{}/opaque-video", server.base))
            .is_err()
    );
    assert!(
        fetcher.bytes_fetched() <= cap,
        "sniff + ranges spent {} bytes, cap is {cap}",
        fetcher.bytes_fetched()
    );
    assert_eq!(
        get_count() - gets_before,
        server.ranges.load(Ordering::Relaxed) - ranges_before,
        "known oversize fallback must not start a full GET"
    );
}

fn frame_pts(path: &Path) -> Vec<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-threads",
            "1",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .expect("probe decoded frame timestamps");
    assert!(
        output.status.success(),
        "probe failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let data: serde_json::Value = serde_json::from_slice(&output.stdout).expect("frame JSON");
    data["frames"]
        .as_array()
        .expect("frame list")
        .iter()
        .map(|frame| {
            frame["best_effort_timestamp_time"]
                .as_str()
                .expect("frame timestamp")
                .parse()
                .expect("numeric frame timestamp")
        })
        .collect()
}

fn assert_native_refinement(
    fetcher: &Fetcher,
    proxy: &Path,
    original: &Path,
    server: &TestServer,
    url: &str,
) {
    use eidolon_core::{FileEntry, FileId, FrameRefiner, Kind};

    let file = FileEntry {
        id: FileId(0),
        path: std::fs::canonicalize(proxy).expect("canonical proxy"),
        size: std::fs::metadata(proxy).expect("proxy metadata").len(),
        mtime_ns: 0,
        kind: Kind::Video,
    };
    assert!(fetcher.has_sparse_videos());
    let before = server.served();
    let spent = fetcher.bytes_fetched();
    let window = fetcher
        .refine(&file, 5.0, 0.5)
        .expect("refine source")
        .expect("native window");
    let source_times = frame_pts(original);
    assert!(
        (window.source_offset_s - source_times[96]).abs() < 0.000_001,
        "window clock must use the actual first source presentation timestamp"
    );
    let native_times = frame_pts(&window.path);
    assert!(
        native_times.len() < 96,
        "native window stops before the GOP end"
    );
    assert!(
        native_times
            .iter()
            .any(|time| (time + window.source_offset_s - 5.5).abs() < 0.000_001)
    );
    let pixels = decode(original, false);
    let stride = 96 * 96 * 3;
    let native = decode(&window.path, false);
    assert_eq!(native.len(), native_times.len() * stride);
    for (frame, time) in native.chunks_exact(stride).zip(&native_times) {
        let source = source_times
            .iter()
            .position(|source| (time + window.source_offset_s - source).abs() < 0.000_001)
            .expect("native PTS maps to an original frame");
        assert_eq!(
            frame,
            &pixels[source * stride..(source + 1) * stride],
            "prefix preserves original pixels"
        );
    }
    let delta = server.served() - before;
    assert!(delta > 0, "refinement fetches missing native packets");
    assert_eq!(
        fetcher
            .refinement_bytes(&file.path)
            .expect("artifact transfer"),
        delta
    );
    assert_eq!(fetcher.bytes_fetched(), spent + delta);
    let repeated = fetcher
        .refine(&file, 5.25, 0.25)
        .expect("repeat refinement")
        .expect("cached window");
    assert_eq!(repeated.path, window.path);
    assert_eq!(
        server.served(),
        before + delta,
        "same prefix is not fetched twice"
    );
    if let Ok(relative) = file.path.strip_prefix("/private") {
        let alias = Path::new("/").join(relative);
        assert_eq!(
            fetcher.refinement_bytes(&alias).expect("alias lookup"),
            delta
        );
    }
    eprintln!(
        "MEASURE video-refinement: initial={spent} additional={delta} total={} source_offset={}",
        fetcher.bytes_fetched(),
        window.source_offset_s
    );
    let mut changed = Resource::new(Vec::new(), "video/mp4");
    changed.etag = Some("\"changed-source\"".to_owned());
    server.add(url::Url::parse(url).expect("source URL").path(), changed);
    let error = fetcher
        .refine(&file, 13.0, 0.5)
        .expect_err("changed source must not refine old samples");
    assert!(error.to_string().contains("source changed"));
    assert_eq!(
        fetcher.bytes_fetched(),
        spent + delta,
        "changed-source HEAD fetches no media"
    );
    assert_eq!(
        fetcher
            .refinement_bytes(&file.path)
            .expect("failed transfer count"),
        delta
    );
}

fn assert_fractional_refinement_budget(
    server: &TestServer,
    url: &str,
    cache: &Path,
    source_size: u64,
) {
    use eidolon_core::{FileEntry, FileId, FrameRefiner, Kind};

    let client = fetcher(cache);
    let before = server.served();
    let result = client.resolve(url).expect("initial sparse proxy");
    let artifact = &result.artifacts[0];
    assert!(!artifact.budget_exhausted);
    let video = FileEntry {
        id: FileId(0),
        path: artifact.local_path.clone(),
        size: std::fs::metadata(&artifact.local_path)
            .expect("proxy metadata")
            .len(),
        mtime_ns: 0,
        kind: Kind::Video,
    };
    assert!(
        client
            .refine(&video, 32.0, 32.0)
            .expect("fractional exhaustion is not a fetch failure")
            .is_none(),
        "a whole native video must not bypass the default fractional limit"
    );
    let transferred = server.served() - before;
    assert!(transferred > result.bytes_fetched);
    assert!(
        transferred <= source_size / 4,
        "quarter budget overshot: {transferred}/{source_size}"
    );
    assert_eq!(client.bytes_fetched(), transferred);
    assert!(client.budget_exhausted(&video.path).expect("budget status"));
    let ranges = server.ranges.load(Ordering::SeqCst);
    assert!(
        client
            .refine(&video, 32.0, 32.0)
            .expect("repeated oversized refinement")
            .is_none()
    );
    assert_eq!(server.ranges.load(Ordering::SeqCst), ranges);
    assert_eq!(server.served() - before, transferred);
    let cached = client.resolve(url).expect("cached provenance");
    assert!(cached.artifacts[0].budget_exhausted);
    assert_eq!(
        serde_json::to_value(&cached.artifacts[0]).expect("provenance JSON")["budget_exhausted"],
        true
    );

    let unlimited = Fetcher::with_options(FetchOptions {
        cache_dir: cache.to_owned(),
        max_fetch_fraction: 1.0,
        ..FetchOptions::default()
    })
    .expect("explicit unlimited fraction");
    unlimited
        .resolve(url)
        .expect("reuse retained source ranges");
    assert!(
        unlimited
            .refine(&video, 32.0, 32.0)
            .expect("unlimited native refinement")
            .is_some()
    );
    assert!(
        !unlimited
            .budget_exhausted(&video.path)
            .expect("unlimited budget status")
    );
}

#[test]
fn long_gop_refinement_fetches_only_the_prefix_through_the_proposed_time() {
    use eidolon_core::{FileEntry, FileId, FrameRefiner, Kind};

    let directory = tempfile::tempdir().expect("long GOP fixture");
    for (name, flags) in [
        ("regular", "+faststart"),
        ("fragmented", "+frag_keyframe+empty_moov+default_base_moof"),
    ] {
        let original = directory.path().join(format!("{name}.mp4"));
        ffmpeg(
            &[
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x180:rate=8:duration=32",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "18",
                "-bf",
                "0",
                "-g",
                "256",
                "-keyint_min",
                "256",
                "-sc_threshold",
                "0",
                "-movflags",
                flags,
            ],
            &original,
        );
        let source = std::fs::read(&original).expect("video fixture");
        let server = TestServer::new();
        server.add("/video.mp4", Resource::new(source.clone(), "video/mp4"));
        let client = fetcher(&directory.path().join(format!("{name}-cache")));
        let result = client
            .resolve(&format!("{}/video.mp4", server.base))
            .expect("sparse video");
        let artifact = &result.artifacts[0];
        let video = FileEntry {
            id: FileId(0),
            path: artifact.local_path.clone(),
            size: std::fs::metadata(&artifact.local_path)
                .expect("proxy metadata")
                .len(),
            mtime_ns: 0,
            kind: Kind::Video,
        };
        let window = client
            .refine(&video, 1.0, 0.0)
            .expect("prefix refinement")
            .expect("prefix fits quarter budget");
        let times = frame_pts(&window.path);
        assert_eq!(times.len(), 9, "only keyframe through proposed timestamp");
        assert_eq!(times.last().copied(), Some(1.0));
        let pixels = decode(&original, false);
        assert_eq!(decode(&window.path, false), pixels[..9 * 96 * 96 * 3]);
        assert!(client.bytes_fetched() * 4 <= source.len() as u64);
        let spent = client.bytes_fetched();
        client
            .refine(&video, 1.0, 0.0)
            .expect("repeat")
            .expect("cached prefix");
        assert_eq!(
            client.bytes_fetched(),
            spent,
            "proof can stop without fetching the GOP remainder"
        );
    }
}

#[test]
fn fresh_sparse_refinement_skips_conditional_head_within_window() {
    use eidolon_core::{FileEntry, FileId, FrameRefiner, Kind};

    let directory = tempfile::tempdir().expect("fresh refinement fixture");
    let original = directory.path().join("original.mp4");
    budget_recovery_video(&original);
    let server = TestServer::new();
    let mut resource = Resource::new(
        std::fs::read(&original).expect("video fixture"),
        "video/mp4",
    );
    resource.cache_control = Some("public, max-age=600".to_owned());
    server.add("/video.mp4", resource);
    let client = Fetcher::with_options(FetchOptions {
        cache_dir: directory.path().join("cache"),
        fresh: true,
        max_fetch_fraction: 1.0,
        ..FetchOptions::default()
    })
    .expect("freshness-enabled client");
    let result = client
        .resolve(&format!("{}/video.mp4", server.base))
        .expect("sparse video");
    assert_eq!(result.artifacts[0].mode, eidolon_fetch::FetchMode::Sparse);
    let path = &result.artifacts[0].local_path;
    let video = FileEntry {
        id: FileId(0),
        path: path.clone(),
        size: std::fs::metadata(path).expect("proxy metadata").len(),
        mtime_ns: 0,
        kind: Kind::Video,
    };
    let before = server.requests.lock().expect("requests").len();
    for center in [1.0, 5.0] {
        let window = client
            .refine(&video, center, 0.0)
            .expect("native refinement")
            .expect("native window");
        assert_eq!(
            frame_pts(&window.path)
                .last()
                .map(|time| time + window.source_offset_s),
            Some(center),
        );
    }
    let requests = server.requests.lock().expect("requests");
    assert_eq!(
        requests[..before]
            .iter()
            .filter(|(method, _)| method == "HEAD")
            .count(),
        1
    );
    assert!(
        requests[before..]
            .iter()
            .all(|(method, _)| method != "HEAD"),
        "fresh refinement must not revalidate: {:?}",
        &requests[before..],
    );
}

fn budget_recovery_video(path: &Path) {
    // A noisy first GOP needs a large uncached read; later GOPs need much less.
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=8:duration=16,noise=alls=100:allf=t:all_seed=713841:enable='lt(t,4)'",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-bf",
            "0",
            "-g",
            "32",
            "-keyint_min",
            "32",
            "-sc_threshold",
            "0",
            "-movflags",
            "+faststart",
        ],
        path,
    );
}

#[test]
fn refused_large_prefix_allows_fitting_and_cached_prefixes_with_truthful_provenance() {
    use eidolon_core::{FileEntry, FileId, FrameRefiner, Kind};

    let directory = tempfile::tempdir().expect("budget recovery fixture");
    let original = directory.path().join("original.mp4");
    budget_recovery_video(&original);
    let source = std::fs::read(&original).expect("video fixture");
    let server = TestServer::new();
    server.add("/video.mp4", Resource::new(source.clone(), "video/mp4"));
    let url = format!("{}/video.mp4", server.base);
    let cache = directory.path().join("cache");
    let warm = fetcher(&cache)
        .resolve(&url)
        .expect("warm sparse acquisition");
    let artifact = &warm.artifacts[0];
    let video = FileEntry {
        id: FileId(0),
        path: artifact.local_path.clone(),
        size: std::fs::metadata(&artifact.local_path)
            .expect("proxy size")
            .len(),
        mtime_ns: 0,
        kind: Kind::Video,
    };
    let client = Fetcher::with_options(FetchOptions {
        cache_dir: cache,
        max_fetch_fraction: 33_280.0 / source.len() as f64,
        ..FetchOptions::default()
    })
    .expect("bounded recovery client");
    assert_eq!(client.resolve(&url).expect("cached proxy").bytes_fetched, 0);
    let before = server.served();
    assert!(
        client
            .refine(&video, 1.0, 0.0)
            .expect("large prefix refusal")
            .is_none()
    );
    assert!(
        client
            .budget_exhausted(&video.path)
            .expect("refusal provenance")
    );
    assert_eq!(
        server.served(),
        before,
        "oversized first read is refused before HTTP"
    );

    let small = client
        .refine(&video, 5.0, 0.0)
        .expect("fitting prefix")
        .expect("a refused request must not block a smaller prefix");
    let transferred = server.served() - before;
    assert!(
        transferred > 0,
        "the fitting prefix must exercise a new HTTP read"
    );
    assert!(transferred <= 33_280, "cumulative cap remains enforced");
    assert_eq!(client.bytes_fetched(), transferred);
    assert_eq!(
        client.refinement_bytes(&video.path).expect("accounting"),
        transferred
    );
    assert_eq!(frame_pts(&small.path).last().copied(), Some(1.0));
    assert!((small.source_offset_s - 4.0).abs() < 1e-6);

    let ranges = server.ranges.load(Ordering::SeqCst);
    let cached = client
        .refine(&video, 5.0, 0.0)
        .expect("cached lookup")
        .expect("cached prefix");
    assert_eq!(cached.path, small.path);
    assert_eq!(server.served() - before, transferred);
    assert_eq!(server.ranges.load(Ordering::SeqCst), ranges);
    assert!(
        client
            .budget_exhausted(&video.path)
            .expect("sticky refusal provenance")
    );
    assert!(
        client
            .resolve(&url)
            .expect("refreshed provenance")
            .artifacts[0]
            .budget_exhausted
    );
}

#[test]
fn cached_windows_and_cached_range_prefixes_remain_available_with_zero_allowance() {
    use eidolon_core::{FileEntry, FileId, FrameRefiner, Kind};

    let directory = tempfile::tempdir().expect("zero allowance fixture");
    let original = directory.path().join("original.mp4");
    budget_recovery_video(&original);
    let source = std::fs::read(&original).expect("video fixture");
    let server = TestServer::new();
    server.add("/video.mp4", Resource::new(source.clone(), "video/mp4"));
    let url = format!("{}/video.mp4", server.base);
    let cache = directory.path().join("cache");
    let warm = fetcher(&cache);
    let result = warm.resolve(&url).expect("warm sparse proxy");
    let artifact = &result.artifacts[0];
    let video = FileEntry {
        id: FileId(0),
        path: artifact.local_path.clone(),
        size: std::fs::metadata(&artifact.local_path)
            .expect("proxy size")
            .len(),
        mtime_ns: 0,
        kind: Kind::Video,
    };
    let cached = warm
        .refine(&video, 5.0, 0.0)
        .expect("warm window")
        .expect("native prefix");
    let client = Fetcher::with_options(FetchOptions {
        cache_dir: cache,
        max_fetch_fraction: 0.5 / source.len() as f64,
        ..FetchOptions::default()
    })
    .expect("zero-byte allowance client");
    assert_eq!(
        client.resolve(&url).expect("cached source").bytes_fetched,
        0
    );
    let before = server.served();

    let hit = client
        .refine(&video, 5.0, 0.0)
        .expect("cached lookup with no allowance")
        .expect("cached windows need no new transfer allowance");
    assert_eq!(hit.path, cached.path);
    assert!(
        !client
            .budget_exhausted(&video.path)
            .expect("no request refused yet")
    );
    assert!(
        client
            .refine(&video, 1.0, 0.0)
            .expect("uncached request")
            .is_none()
    );
    assert!(
        client
            .budget_exhausted(&video.path)
            .expect("refusal provenance")
    );

    let smaller = client
        .refine(&video, 4.5, 0.0)
        .expect("cached source ranges")
        .expect("a new shorter window can use already retained bytes");
    assert_ne!(smaller.path, cached.path);
    assert_eq!(frame_pts(&smaller.path).last().copied(), Some(0.5));
    assert!(
        client
            .refine(&video, 5.0, 0.0)
            .expect("cached after refusal")
            .is_some()
    );
    assert_eq!(
        server.served(),
        before,
        "zero allowance never fetches new body bytes"
    );
    assert_eq!(client.bytes_fetched(), 0);
    assert_eq!(client.refinement_bytes(&video.path).expect("accounting"), 0);
    assert!(
        client
            .budget_exhausted(&video.path)
            .expect("refusal remains recorded")
    );
}

#[test]
fn long_video_proxy_uses_less_than_quarter_bytes_and_preserves_keyframe_pixels_and_pts() {
    let directory = tempfile::tempdir().expect("video fixture");
    let original = directory.path().join("original.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=24:duration=64",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-g",
            "96",
            "-keyint_min",
            "96",
            "-sc_threshold",
            "0",
            "-movflags",
            "+faststart",
        ],
        &original,
    );
    let source = std::fs::read(&original).expect("fixture video");
    let server = TestServer::new();
    server.add("/original.mp4", Resource::new(source.clone(), "video/mp4"));
    let cache = directory.path().join("sparse-cache");
    let fetcher = fetcher(&cache);
    let url = format!("{}/original.mp4", server.base);
    let sparse = fetcher.resolve(&url).expect("sparse proxy");
    assert!(sparse.warnings.is_empty(), "{:?}", sparse.warnings);
    assert_eq!(sparse.artifacts.len(), 1);
    assert!(!sparse.artifacts[0].complete_bytes);
    assert_eq!(sparse.artifacts[0].mode, eidolon_fetch::FetchMode::Sparse);
    assert_eq!(sparse.artifacts[0].mode_reason, "indexed_keyframes");
    assert_eq!(sparse.bytes_fetched, server.served());
    assert!(
        sparse.bytes_fetched * 4 <= source.len() as u64,
        "sparse fetch {}/{}",
        sparse.bytes_fetched,
        source.len()
    );
    assert!(server.ranges.load(Ordering::Relaxed) > 0);
    let proxy = &sparse.artifacts[0].local_path;
    assert_eq!(
        decode(proxy, true),
        decode(&original, true),
        "retained keyframes must decode to original RGB pixels"
    );
    assert_source_timeline(proxy);
    assert_eq!(
        fetcher
            .resolve(&url)
            .expect("proxy cache revalidation")
            .bytes_fetched,
        0
    );
    let before = server.served();
    let full = Fetcher::with_options(FetchOptions {
        cache_dir: directory.path().join("cadence-cache"),
        video_mode: VideoMode::FullCadence,
        ..FetchOptions::default()
    })
    .expect("full-cadence client")
    .resolve(&url)
    .expect("full-cadence proxy");
    assert!(full.warnings.is_empty(), "{:?}", full.warnings);
    assert_eq!(full.artifacts[0].mode, eidolon_fetch::FetchMode::Full);
    assert_eq!(full.artifacts[0].mode_reason, "requested_full_cadence");
    assert!(
        !full.artifacts[0].complete_bytes,
        "remuxed proxy is not original bytes"
    );
    assert_eq!(full.bytes_fetched, server.served() - before);
    assert_eq!(
        decode(&full.artifacts[0].local_path, false),
        decode(&original, false)
    );
    eprintln!(
        "MEASURE video-keyframes: fetched={} served={} total={} ratio={:.6} independent_observations=16 nominal_two_fps_budget=128 retained_pixel_accuracy=1.0",
        sparse.bytes_fetched,
        before,
        source.len(),
        sparse.bytes_fetched as f64 / source.len() as f64
    );
    eprintln!(
        "MEASURE video-full-cadence: fetched={} served={} total={} ratio={:.6} source_frames_available=1536 nominal_two_fps_budget=128 decoded_pixel_accuracy=1.0",
        full.bytes_fetched,
        server.served() - before,
        source.len(),
        full.bytes_fetched as f64 / source.len() as f64
    );
    assert_unknown_video_budget(&server, &source, &directory.path().join("bounded-cache"));
    assert_refinement_budget(&server, &url, &directory.path().join("refinement-cap"));
    let fractional_cache = directory.path().join("fractional-cap");
    assert_fractional_refinement_budget(&server, &url, &fractional_cache, source.len() as u64);
    assert_native_refinement(&fetcher, proxy, &original, &server, &url);
}

#[test]
fn fragmented_video_proxy_keeps_sync_pixels_clock_and_native_refinement() {
    let directory = tempfile::tempdir().expect("fragmented fixture");
    let original = directory.path().join("fragmented.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=24:duration=64",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-g",
            "96",
            "-keyint_min",
            "96",
            "-sc_threshold",
            "0",
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof+dash",
        ],
        &original,
    );
    let source = std::fs::read(&original).expect("fragmented bytes");
    let server = TestServer::new();
    server.add(
        "/fragmented.mp4",
        Resource::new(source.clone(), "video/mp4"),
    );
    let url = format!("{}/fragmented.mp4", server.base);
    let fetcher = fetcher(&directory.path().join("fragment-cache"));
    let result = fetcher.resolve(&url).expect("fragmented proxy");
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(result.bytes_fetched, server.served());
    assert!(
        result.bytes_fetched * 4 <= source.len() as u64,
        "fragmented sparse ratio"
    );
    let proxy = &result.artifacts[0].local_path;
    assert!(!result.artifacts[0].complete_bytes);
    assert!(
        decode(proxy, true) == decode(&original, true),
        "fragmented keyframe pixels"
    );
    let source_pts = frame_pts(&original);
    let proxy_pts = frame_pts(proxy);
    let metadata: serde_json::Value = serde_json::from_slice(
        &std::fs::read(PathBuf::from(format!("{}.remote.json", proxy.display())))
            .expect("sparse clock"),
    )
    .expect("clock JSON");
    assert_eq!(proxy_pts.len(), 16);
    for ((actual, expected), retained) in proxy_pts.iter().zip(source_pts.iter().step_by(96)).zip(
        metadata["sample_pts_seconds"]
            .as_array()
            .expect("retained points"),
    ) {
        assert!(
            (actual - expected).abs() < 0.000_001,
            "fragmented presentation clock"
        );
        assert!((retained.as_f64().expect("retained PTS") - expected).abs() < 0.000_001);
    }
    eprintln!(
        "MEASURE video-fragmented-keyframes: fetched={} served={} total={} ratio={:.6} independent_observations=16 retained_pixel_accuracy=1.0",
        result.bytes_fetched,
        server.served(),
        source.len(),
        result.bytes_fetched as f64 / source.len() as f64
    );
    assert_full_fragmented(
        &server,
        &url,
        &original,
        source.len() as u64,
        &directory.path().join("full-fragment-cache"),
    );
    assert_refinement_budget(&server, &url, &directory.path().join("refinement-cap"));
    assert_native_refinement(&fetcher, proxy, &original, &server, &url);
}

#[test]
fn unindexed_webm_and_range_unsupported_videos_fall_back_with_an_explicit_warning() {
    let server = TestServer::new();
    let fixture = tempfile::tempdir().expect("fallback fixture");
    let webm = fixture.path().join("sample.webm");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=8:duration=2",
            "-an",
            "-c:v",
            "libvpx-vp9",
            "-deadline",
            "realtime",
            "-live",
            "1",
        ],
        &webm,
    );
    let source = std::fs::read(&webm).expect("real WebM bytes");
    server.add("/video.webm", Resource::new(source.clone(), "video/webm"));
    let cache = tempfile::tempdir().expect("cache");
    let fetcher = fetcher(cache.path());
    let result = fetcher
        .resolve(&format!("{}/video.webm", server.base))
        .expect("bounded fallback");
    assert!(result.bytes_fetched >= source.len() as u64);
    assert_eq!(result.bytes_fetched, server.served());
    assert_eq!(
        std::fs::read(&result.artifacts[0].local_path).expect("fallback bytes"),
        source
    );
    assert_eq!(
        decode(&result.artifacts[0].local_path, false),
        decode(&webm, false)
    );
    assert!(result.artifacts[0].complete_bytes);
    assert_eq!(result.artifacts[0].mode, eidolon_fetch::FetchMode::Full);
    assert!(
        result.artifacts[0]
            .mode_reason
            .contains("range proxy unavailable")
    );
    let cached = fetcher
        .resolve(&format!("{}/video.webm", server.base))
        .expect("cached fallback");
    assert_eq!(cached.bytes_fetched, 0);
    assert_eq!(cached.artifacts[0].mode, result.artifacts[0].mode);
    assert_eq!(
        cached.artifacts[0].mode_reason,
        result.artifacts[0].mode_reason
    );
    assert!(result.warnings[0].contains("downloading video in full"));
    eprintln!(
        "MEASURE video-webm-fallback: fetched={} served={} total={} ratio={:.6}",
        result.bytes_fetched,
        server.served(),
        source.len(),
        result.bytes_fetched as f64 / source.len() as f64
    );
    let mp4 = fixture.path().join("sample.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=8:duration=2",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
        ],
        &mp4,
    );
    let mut unsupported = Resource::new(std::fs::read(&mp4).expect("real MP4 bytes"), "video/mp4");
    unsupported.ignore_range = true;
    server.add("/unsupported.mp4", unsupported);
    let fallback_url = format!("{}/unsupported.mp4", server.base);
    let fallback = fetcher.resolve(&fallback_url).expect("range fallback");
    assert!(fallback.warnings[0].contains("range proxy unavailable"));
    assert_eq!(fallback.artifacts[0].mode, eidolon_fetch::FetchMode::Full);
    assert!(
        fallback.artifacts[0]
            .mode_reason
            .contains("range proxy unavailable")
    );
    let cached = fetcher
        .resolve(&fallback_url)
        .expect("cached range fallback");
    assert_eq!(cached.bytes_fetched, 0);
    assert_eq!(
        cached.artifacts[0].mode_reason,
        fallback.artifacts[0].mode_reason
    );
}

#[test]
fn hevc_falls_back_to_original_when_decoder_config_cannot_be_preserved() {
    let directory = tempfile::tempdir().expect("HEVC fixture");
    let original = directory.path().join("hevc.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=4:duration=1",
            "-an",
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-x265-params",
            "pools=none:frame-threads=1:log-level=error",
        ],
        &original,
    );
    let source = std::fs::read(&original).expect("HEVC bytes");
    let server = TestServer::new();
    server.add("/hevc.mp4", Resource::new(source.clone(), "video/mp4"));
    let result = fetcher(&directory.path().join("cache"))
        .resolve(&format!("{}/hevc.mp4", server.base))
        .expect("safe HEVC fallback");
    assert!(
        result.artifacts[0].complete_bytes,
        "HEVC must not use lossy decoder configuration"
    );
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.contains("HEVC"))
    );
    assert_eq!(
        std::fs::read(&result.artifacts[0].local_path).expect("retained HEVC"),
        source
    );
    assert!(
        decode(&result.artifacts[0].local_path, false) == decode(&original, false),
        "HEVC fallback must preserve every decoded pixel"
    );
}

fn assert_full_fragmented(
    server: &TestServer,
    url: &str,
    original: &Path,
    size: u64,
    cache: &Path,
) {
    let client = Fetcher::with_options(FetchOptions {
        cache_dir: cache.to_owned(),
        video_mode: VideoMode::FullCadence,
        ..FetchOptions::default()
    })
    .expect("full fragmented client");
    let before = server.served();
    let result = client.resolve(url).expect("full fragmented proxy");
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(result.bytes_fetched, server.served() - before);
    assert!(!result.artifacts[0].complete_bytes);
    assert!(!client.has_sparse_videos());
    let proxy = &result.artifacts[0].local_path;
    assert!(
        decode(proxy, false) == decode(original, false),
        "full fragmented frame pixels"
    );
    assert_eq!(
        frame_pts(proxy),
        frame_pts(original),
        "full fragmented frame timestamps"
    );
    eprintln!(
        "MEASURE video-fragmented-full: fetched={} served={} total={size} ratio={:.6} decoded_pixel_accuracy=1.0",
        result.bytes_fetched,
        server.served() - before,
        result.bytes_fetched as f64 / size as f64
    );
}

fn assert_refinement_budget(server: &TestServer, url: &str, cache: &Path) {
    use eidolon_core::{FileEntry, FileId, FrameRefiner, Kind};

    let cap = 640 * 1024;
    let client = Fetcher::with_options(FetchOptions {
        cache_dir: cache.to_owned(),
        max_video_bytes: cap,
        max_fetch_fraction: 1.0,
        ..FetchOptions::default()
    })
    .expect("bounded refinement client");
    let result = client.resolve(url).expect("initial sparse proxy fits cap");
    let path = &result.artifacts[0].local_path;
    let file = FileEntry {
        id: FileId(0),
        path: path.clone(),
        size: std::fs::metadata(path).expect("proxy size").len(),
        mtime_ns: 0,
        kind: Kind::Video,
    };
    let before = server.served();
    client
        .refine(&file, 17.0, 3.0)
        .expect_err("native dependencies exceed remaining cap");
    let delta = client
        .refinement_bytes(path)
        .expect("failed refinement transfer");
    assert!(delta > 0, "failed refinement retains its partial transfer");
    assert_eq!(delta, server.served() - before);
    assert_eq!(client.bytes_fetched(), result.bytes_fetched + delta);
    assert!(
        client.bytes_fetched() <= cap,
        "initial plus failed refinement exceeds cap"
    );
}

fn assert_initial_fraction_exhausted(
    server: &TestServer,
    url: &str,
    cache: &Path,
    source_size: u64,
) {
    use eidolon_core::{FileEntry, FileId, FrameRefiner, Kind};

    let limited = fetcher(cache);
    let initial = limited.resolve(url).expect("initial low-bitrate proxy");
    assert!(initial.bytes_fetched * 4 > source_size);
    let path = &initial.artifacts[0].local_path;
    let video = FileEntry {
        id: FileId(0),
        path: path.clone(),
        size: std::fs::metadata(path).expect("proxy size").len(),
        mtime_ns: 0,
        kind: Kind::Video,
    };
    let ranges = server.ranges.load(Ordering::SeqCst);
    assert!(
        limited
            .refine(&video, 5.0, 0.25)
            .expect("initial transfer already exhausted fraction")
            .is_none()
    );
    assert_eq!(server.ranges.load(Ordering::SeqCst), ranges);
    assert!(limited.budget_exhausted(path).expect("budget status"));
    assert_eq!(
        limited.refinement_bytes(path).expect("refinement transfer"),
        0
    );
}

#[test]
fn moov_at_end_keeps_sparse_provenance_after_native_refinement() {
    use eidolon_core::{FileEntry, FileId, FrameRefiner, Kind};

    let directory = tempfile::tempdir().expect("moov-at-end fixture");
    let original = directory.path().join("moov-at-end.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=24:duration=64",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-g",
            "48",
            "-keyint_min",
            "48",
            "-sc_threshold",
            "0",
        ],
        &original,
    );
    let source = std::fs::read(&original).expect("MP4 source");
    let mut position = 0;
    let mut kinds = Vec::new();
    while position < source.len() {
        let length = u32::from_be_bytes(
            source[position..position + 4]
                .try_into()
                .expect("box length"),
        ) as usize;
        assert!(length >= 8 && position + length <= source.len());
        kinds.push(&source[position + 4..position + 8]);
        position += length;
    }
    assert_eq!(kinds, [b"ftyp", b"free", b"mdat", b"moov"]);
    let server = TestServer::new();
    server.add(
        "/moov-at-end.mp4",
        Resource::new(source.clone(), "video/mp4"),
    );
    let client = Fetcher::with_options(FetchOptions {
        cache_dir: directory.path().join("cache"),
        max_fetch_fraction: 1.0,
        ..FetchOptions::default()
    })
    .expect("unlimited native-window control");
    let url = format!("{}/moov-at-end.mp4", server.base);
    let result = client.resolve(&url).expect("sparse moov-at-end proxy");
    let artifact = &result.artifacts[0];
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(artifact.mode, eidolon_fetch::FetchMode::Sparse);
    assert_eq!(artifact.mode_reason, "indexed_keyframes");
    assert_eq!(result.bytes_fetched, server.served());
    assert!(result.bytes_fetched <= source.len() as u64);
    assert_eq!(
        serde_json::to_value(artifact).expect("provenance JSON")["mode"],
        "sparse"
    );
    let video = FileEntry {
        id: FileId(0),
        path: artifact.local_path.clone(),
        size: std::fs::metadata(&artifact.local_path)
            .expect("proxy size")
            .len(),
        mtime_ns: 0,
        kind: Kind::Video,
    };
    client
        .refine(&video, 5.0, 0.25)
        .expect("native refinement")
        .expect("native window");
    let refined = client
        .refinement_bytes(&video.path)
        .expect("refinement bytes");
    assert_eq!(server.served(), result.bytes_fetched + refined);
    eprintln!(
        "MEASURE video-small-gop2-moov-at-end: initial={} native={} served={} total={} initial_ratio={:.6} total_ratio={:.6}",
        result.bytes_fetched,
        refined,
        server.served(),
        source.len(),
        result.bytes_fetched as f64 / source.len() as f64,
        server.served() as f64 / source.len() as f64
    );
    let cached = client.resolve(&url).expect("cached sparse artifact");
    assert_eq!(cached.bytes_fetched, 0);
    assert_eq!(cached.artifacts[0].mode, artifact.mode);
    assert_eq!(cached.artifacts[0].mode_reason, artifact.mode_reason);
    assert_initial_fraction_exhausted(
        &server,
        &url,
        &directory.path().join("quarter-cache"),
        source.len() as u64,
    );
}

#[test]
fn webm_cues_fetch_keyframes_and_refine_native_prefixes_without_audio() {
    use eidolon_core::{FileEntry, FileId, FrameRefiner, Kind};

    let directory = tempfile::tempdir().expect("WebM fixture");
    let original = directory.path().join("original.webm");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=8:duration=32",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=32",
            "-c:a",
            "libopus",
            "-c:v",
            "libvpx-vp9",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-b:v",
            "600k",
            "-g",
            "32",
        ],
        &original,
    );
    let source = std::fs::read(&original).expect("WebM bytes");
    let server = TestServer::new();
    server.add("/video.webm", Resource::new(source.clone(), "video/webm"));
    let client = fetcher(&directory.path().join("cache"));
    let url = format!("{}/video.webm", server.base);
    let result = client.resolve(&url).expect("Cues proxy");
    let artifact = &result.artifacts[0];
    assert_eq!(
        artifact.mode,
        eidolon_fetch::FetchMode::Sparse,
        "{:?}",
        result.warnings
    );
    assert!(!artifact.complete_bytes);
    assert!(
        result.bytes_fetched * 4 < source.len() as u64,
        "Cues proxy fetches fewer than 25%: {} / {}",
        result.bytes_fetched,
        source.len()
    );
    assert_eq!(result.bytes_fetched, server.served());
    assert_webm_keyframes(&original, artifact);
    let video = FileEntry {
        id: FileId(0),
        path: artifact.local_path.clone(),
        size: std::fs::metadata(&artifact.local_path)
            .expect("metadata")
            .len(),
        mtime_ns: 0,
        kind: Kind::Video,
    };
    let window = client
        .refine(&video, 1.0, 0.25)
        .expect("WebM native prefix")
        .expect("prefix budget");
    let native = decode(&window.path, false);
    assert!(
        native.len() >= 96 * 96 * 3,
        "native prefix contains decoded frames"
    );
    let original_pixels = decode(&original, false);
    assert_eq!(native, original_pixels[..native.len()]);
    assert!(
        frame_pts(&window.path).last().copied().expect("last frame") + window.source_offset_s
            <= 1.375
    );
    let before = client.bytes_fetched();
    client
        .refine(&video, 1.0, 0.25)
        .expect("cached window")
        .expect("window");
    assert_eq!(before, client.bytes_fetched());
    assert_eq!(client.bytes_fetched(), server.served());
    assert!(before * 4 <= source.len() as u64);
}

#[test]
fn matroska_h264_cues_keep_reordered_pixels_across_continuation_clusters() {
    use eidolon_core::{FileEntry, FileId, FrameRefiner, Kind};

    let directory = tempfile::tempdir().expect("Matroska fixture");
    let original = directory.path().join("original.mkv");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=8:duration=32",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-g",
            "32",
            "-keyint_min",
            "32",
            "-sc_threshold",
            "0",
            "-cluster_time_limit",
            "1000",
            "-cues_to_front",
            "1",
        ],
        &original,
    );
    let source = std::fs::read(&original).expect("Matroska bytes");
    let server = TestServer::new();
    server.add(
        "/video.mkv",
        Resource::new(source.clone(), "video/x-matroska"),
    );
    let client = fetcher(&directory.path().join("cache"));
    let result = client
        .resolve(&format!("{}/video.mkv", server.base))
        .expect("Cues proxy");
    let artifact = &result.artifacts[0];
    assert_eq!(
        artifact.mode,
        eidolon_fetch::FetchMode::Sparse,
        "{:?}",
        result.warnings
    );
    assert_eq!(decode(&artifact.local_path, false).len(), 8 * 96 * 96 * 3);
    let video = FileEntry {
        id: FileId(0),
        path: artifact.local_path.clone(),
        size: artifact.local_path.metadata().expect("metadata").len(),
        mtime_ns: 0,
        kind: Kind::Video,
    };
    let window = client
        .refine(&video, 1.5, 0.0)
        .expect("reordered native prefix")
        .expect("prefix budget");
    let times = frame_pts(&window.path);
    assert!(
        times
            .iter()
            .any(|time| (time + window.source_offset_s - 1.5).abs() < 0.000_001),
        "target frame remains decodable"
    );
    let source_times = frame_pts(&original);
    let source_pixels = decode(&original, false);
    let pixels = decode(&window.path, false);
    let stride = 96 * 96 * 3;
    assert_eq!(pixels.len(), times.len() * stride);
    for (frame, time) in pixels.chunks_exact(stride).zip(times) {
        let position = source_times
            .iter()
            .position(|source| (source - time - window.source_offset_s).abs() < 0.000_001)
            .expect("source PTS");
        assert!(
            frame == &source_pixels[position * stride..(position + 1) * stride],
            "reordered source pixels at {time}"
        );
    }
    assert!(client.bytes_fetched() * 4 <= source.len() as u64);
    assert_eq!(client.bytes_fetched(), server.served());
}

fn assert_webm_keyframes(original: &Path, artifact: &eidolon_fetch::Remote) {
    let original_pixels = decode(original, false);
    let sparse_pixels = decode(&artifact.local_path, false);
    let stride = 96 * 96 * 3;
    assert_eq!(
        sparse_pixels.len(),
        8 * stride,
        "exactly eight independent keyframes"
    );
    for (index, frame) in sparse_pixels.chunks_exact(stride).enumerate() {
        assert!(
            frame == &original_pixels[index * 32 * stride..(index * 32 + 1) * stride],
            "keyframe {index} pixels remain unchanged"
        );
    }
    let sidecar: serde_json::Value = serde_json::from_slice(
        &std::fs::read(format!("{}.remote.json", artifact.local_path.display()))
            .expect("source timeline"),
    )
    .expect("timeline JSON");
    let expected = frame_pts(original);
    let retained = sidecar["sample_pts_seconds"]
        .as_array()
        .expect("retained times");
    assert_eq!(retained.len(), 8);
    for (index, point) in retained.iter().enumerate() {
        assert!((point.as_f64().expect("time") - expected[index * 32]).abs() < 0.001);
    }
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
        ])
        .arg(&artifact.local_path)
        .output()
        .expect("proxy tracks");
    let tracks: serde_json::Value = serde_json::from_slice(&probe.stdout).expect("tracks");
    assert_eq!(
        tracks["streams"].as_array().expect("streams").len(),
        1,
        "no audio track"
    );
}
