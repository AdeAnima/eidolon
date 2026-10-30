use std::collections::HashSet;
use std::sync::LazyLock;

use scraper::{ElementRef, Html, Selector};
use serde_json::Value;
use url::Url;

pub(crate) struct Discovered {
    pub(crate) url: Url,
    pub(crate) role: String,
}

pub(crate) struct Extracted {
    pub(crate) media: Vec<Discovered>,
    pub(crate) text: String,
}

static META: LazyLock<Selector> =
    LazyLock::new(|| Selector::parse("meta[property], meta[name]").expect("constant CSS"));
static VIDEO: LazyLock<Selector> =
    LazyLock::new(|| Selector::parse("video, video source").expect("constant CSS"));
static JSON_LD: LazyLock<Selector> =
    LazyLock::new(|| Selector::parse("script[type='application/ld+json']").expect("constant CSS"));
static CONTENT: LazyLock<Selector> =
    LazyLock::new(|| Selector::parse("article, main, [role='main'], body").expect("constant CSS"));
static BLOCKS: LazyLock<Selector> =
    LazyLock::new(|| Selector::parse("h1, h2, h3, p, li, blockquote, pre").expect("constant CSS"));

pub(crate) fn extract(source: &str, page: &Url) -> Extracted {
    let document = Html::parse_document(source);
    let mut media = Vec::new();
    let mut seen = HashSet::new();
    let mut add = |value: &str, role: &str| {
        if let Ok(mut url) = page.join(value.trim())
            && matches!(url.scheme(), "http" | "https")
            && url.username().is_empty()
            && url.password().is_none()
        {
            url.set_fragment(None);
            if seen.insert(url.clone()) {
                media.push(Discovered {
                    url,
                    role: role.to_owned(),
                });
            }
        }
    };
    for element in document.select(&META) {
        let name = element
            .value()
            .attr("property")
            .or_else(|| element.value().attr("name"))
            .unwrap_or_default()
            .to_ascii_lowercase();
        let role = match name.as_str() {
            "og:image" | "og:image:url" | "og:image:secure_url" => "og:image",
            "og:video" | "og:video:url" | "og:video:secure_url" => "og:video",
            "twitter:image" | "twitter:image:src" => "twitter:image",
            _ => continue,
        };
        if let Some(value) = element.value().attr("content") {
            add(value, role);
        }
    }
    for element in document.select(&VIDEO) {
        if let Some(value) = element.value().attr("src") {
            add(value, "video");
        }
        if let Some(value) = element.value().attr("poster") {
            add(value, "video thumbnail");
        }
    }
    for element in document.select(&JSON_LD) {
        if let Ok(value) = serde_json::from_str::<Value>(&element.inner_html()) {
            json_media(&value, &mut add);
        }
    }
    let text = document
        .select(&CONTENT)
        .map(|element| {
            let semantic = element.value().name() != "body";
            let text = article_text(element);
            // Prefer article/main boundaries over an enclosing body with navigation.
            let score = text.len().saturating_mul(if semantic { 4 } else { 1 });
            (score, text)
        })
        .max_by_key(|(score, _)| *score)
        .map_or_else(String::new, |(_, text)| text);
    Extracted { media, text }
}

fn json_media(value: &Value, add: &mut impl FnMut(&str, &str)) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                match key.as_str() {
                    "thumbnailUrl" => json_urls(value, "video thumbnail", add),
                    "contentUrl" => json_urls(value, "JSON-LD content", add),
                    _ => json_media(value, add),
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                json_media(value, add);
            }
        }
        _ => {}
    }
}

fn json_urls(value: &Value, role: &str, add: &mut impl FnMut(&str, &str)) {
    match value {
        Value::String(url) => add(url, role),
        Value::Array(values) => {
            for value in values {
                json_urls(value, role, add);
            }
        }
        Value::Object(object) => {
            if let Some(Value::String(url)) = object.get("url") {
                add(url, role);
            }
        }
        _ => {}
    }
}

fn visible(element: ElementRef<'_>) -> bool {
    std::iter::once(element)
        .chain(element.ancestors().filter_map(ElementRef::wrap))
        .all(|parent| {
            let value = parent.value();
            !matches!(
                value.name(),
                "script"
                    | "style"
                    | "nav"
                    | "header"
                    | "footer"
                    | "aside"
                    | "form"
                    | "noscript"
                    | "svg"
                    | "template"
            ) && value.attr("hidden").is_none()
                && value.attr("aria-hidden") != Some("true")
        })
}

fn visible_text(element: ElementRef<'_>) -> String {
    element
        .descendants()
        .filter_map(|node| {
            let text = node.value().as_text()?;
            let parent = node.parent().and_then(ElementRef::wrap)?;
            visible(parent).then_some(text.as_ref())
        })
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn article_text(element: ElementRef<'_>) -> String {
    let blocks: Vec<_> = element
        .select(&BLOCKS)
        .filter(|block| visible(*block))
        .filter(|block| {
            !block
                .ancestors()
                .take_while(|parent| parent.id() != element.id())
                .filter_map(ElementRef::wrap)
                .any(|parent| BLOCKS.matches(&parent))
        })
        .map(visible_text)
        .filter(|text| !text.is_empty())
        .collect();
    if blocks.is_empty() {
        visible_text(element)
    } else {
        blocks.join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::extract;

    #[test]
    fn discovers_media_and_keeps_article_without_navigation() {
        let page = url::Url::parse("https://example.test/news/story").expect("valid fixture URL");
        let extracted = extract(include_str!("../tests/fixtures/article.html"), &page);
        let urls: Vec<_> = extracted
            .media
            .iter()
            .map(|media| media.url.as_str())
            .collect();
        assert_eq!(urls.len(), 8);
        assert!(urls.contains(&"https://example.test/cover.png"));
        assert!(urls.contains(&"https://example.test/alternate.webm"));
        assert!(urls.contains(&"https://example.test/preview.png"));
        assert!(extracted.text.contains("A patient observer"));
        assert!(!extracted.text.contains("Subscribe"));
        assert!(!extracted.text.contains("tracking"));
        assert!(!extracted.text.contains("Copyright"));
    }

    #[test]
    fn ignores_non_http_credentials_duplicates_and_script_text() {
        let page = url::Url::parse("https://example.test/").expect("valid fixture URL");
        let extracted = extract(
            r#"<meta property="og:image" content="file:///etc/passwd"><meta name="twitter:image" content="https://user:pass@example.test/a"><meta property="og:image" content="/a.png"><meta name="twitter:image" content="/a.png"><main><p>Visible &amp; useful text.</p><nav>Hidden menu</nav><div hidden>Invisible</div></main>"#,
            &page,
        );
        assert_eq!(extracted.media.len(), 1);
        assert_eq!(extracted.text, "Visible & useful text.");
    }
}
