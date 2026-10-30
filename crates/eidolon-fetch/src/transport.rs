use std::collections::HashMap;
use std::io::Read;
use std::ops::Deref;
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context as _, ensure};
use reqwest::StatusCode;
use reqwest::blocking::{Client, Request, RequestBuilder, Response};
use reqwest::header::{LOCATION, RETRY_AFTER};

use crate::http;

static HOSTS: LazyLock<Mutex<HashMap<String, Arc<Host>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

struct Host {
    state: Mutex<State>,
    changed: Condvar,
}

struct State {
    active: u8,
    resume: Instant,
}

struct Permit(Arc<Host>);

impl Host {
    fn for_request(request: &Request) -> anyhow::Result<Arc<Self>> {
        let name = request
            .url()
            .host_str()
            .context("HTTP request has no host")?;
        let mut hosts = HOSTS
            .lock()
            .map_err(|_| anyhow::anyhow!("host limiter poisoned"))?;
        if let Some(host) = hosts.get(name) {
            return Ok(Arc::clone(host));
        }
        let host = Arc::new(Self {
            state: Mutex::new(State {
                active: 0,
                resume: Instant::now(),
            }),
            changed: Condvar::new(),
        });
        hosts.insert(name.to_owned(), Arc::clone(&host));
        Ok(host)
    }

    fn acquire(self: &Arc<Self>, deadline: Instant) -> anyhow::Result<Permit> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("host limiter poisoned"))?;
        loop {
            let now = Instant::now();
            ensure!(now < deadline, "URL resolution timed out waiting for host");
            if state.active < 2 && now >= state.resume {
                state.active += 1;
                return Ok(Permit(Arc::clone(self)));
            }
            let wake = if now < state.resume {
                deadline.min(state.resume)
            } else {
                deadline
            };
            (state, _) = self
                .changed
                .wait_timeout(state, wake.saturating_duration_since(now))
                .map_err(|_| anyhow::anyhow!("host limiter poisoned"))?;
        }
    }

    fn defer(&self, delay: Duration) -> anyhow::Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("host limiter poisoned"))?;
        state.resume = state.resume.max(Instant::now() + delay);
        self.changed.notify_all();
        Ok(())
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        if let Ok(mut state) = self.0.state.lock() {
            state.active -= 1;
            self.0.changed.notify_all();
        }
    }
}

pub(crate) struct LimitedResponse {
    response: Response,
    _permit: Permit,
}

impl Deref for LimitedResponse {
    type Target = Response;

    fn deref(&self) -> &Self::Target {
        &self.response
    }
}

impl Read for LimitedResponse {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.response.read(buffer)
    }
}

fn retry_delay(value: Option<&str>, now: SystemTime, attempt: u32) -> Duration {
    value
        .and_then(|value| {
            value
                .trim()
                .parse::<u64>()
                .map(Duration::from_secs)
                .ok()
                .or_else(|| {
                    httpdate::parse_http_date(value)
                        .ok()
                        .map(|date| date.duration_since(now).unwrap_or_default())
                })
        })
        .unwrap_or_else(|| Duration::from_secs(1 << attempt))
        .min(Duration::from_secs(30))
}

fn execute(
    client: &Client,
    template: &Request,
    timeout: Duration,
    deadline: Instant,
) -> anyhow::Result<LimitedResponse> {
    let host = Host::for_request(template)?;
    let mut attempt = 0;
    loop {
        let permit = host.acquire(deadline)?;
        let mut request = template
            .try_clone()
            .context("HTTP request cannot be retried")?;
        *request.timeout_mut() =
            Some(timeout.min(deadline.saturating_duration_since(Instant::now())));
        let response = client.execute(request).context("HTTP request")?;
        if matches!(
            response.status(),
            StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE
        ) {
            let value = response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|value| value.to_str().ok());
            host.defer(retry_delay(value, SystemTime::now(), attempt))?;
            if attempt < 2 {
                attempt += 1;
                drop(response);
                drop(permit);
                continue;
            }
        }
        return Ok(LimitedResponse {
            response,
            _permit: permit,
        });
    }
}

pub(crate) fn send(
    client: &Client,
    request: RequestBuilder,
    timeout: Duration,
    deadline: Instant,
) -> anyhow::Result<LimitedResponse> {
    let mut request = request.build().context("building HTTP request")?;
    http::validate(request.url().as_str())?;
    let mut redirects = 0;
    loop {
        let response = execute(client, &request, timeout, deadline)?;
        if matches!(
            response.status(),
            StatusCode::MOVED_PERMANENTLY
                | StatusCode::FOUND
                | StatusCode::SEE_OTHER
                | StatusCode::TEMPORARY_REDIRECT
                | StatusCode::PERMANENT_REDIRECT
        ) && let Some(location) = response.headers().get(LOCATION)
        {
            ensure!(redirects < 10, "too many redirects");
            let target = response
                .url()
                .join(location.to_str().context("invalid redirect location")?)?;
            *request.url_mut() = http::validate(target.as_str())?;
            redirects += 1;
            continue;
        }
        return Ok(response);
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use super::retry_delay;

    #[test]
    fn retry_after_parses_seconds_dates_and_bounds() {
        let now = UNIX_EPOCH + Duration::from_secs(10);
        assert_eq!(retry_delay(Some("20"), now, 0), Duration::from_secs(20));
        assert_eq!(retry_delay(Some("3600"), now, 0), Duration::from_secs(30));
        assert_eq!(
            retry_delay(Some("Thu, 01 Jan 1970 00:00:20 GMT"), now, 0),
            Duration::from_secs(10)
        );
        assert_eq!(
            retry_delay(Some("Thu, 01 Jan 1970 00:00:05 GMT"), now, 0),
            Duration::ZERO
        );
        assert_eq!(retry_delay(Some("invalid"), now, 0), Duration::from_secs(1));
        assert_eq!(retry_delay(None, now, 1), Duration::from_secs(2));
    }
}
