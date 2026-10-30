//! Bounded HTML field-slot alignment, guarded by a changed document ID or amount.

use serde::{Deserialize, Serialize};

const MAX_FIELD_WORDS: usize = 8;
const MAX_FIELD_CHARACTERS: usize = 120;
const MAX_FIELD_RUN: usize = 4;
const MIN_FIXED_WORDS: usize = 20;

#[derive(Serialize, Deserialize)]
pub(super) struct Field {
    start: usize,
    end: usize,
    words: usize,
    short: bool,
    anchor: bool,
}

struct Tag<'a> {
    start: usize,
    end: usize,
    name: &'a str,
    closing: bool,
}

fn tag_at(text: &str, start: usize) -> Option<Tag<'_>> {
    let rest = text.get(start..)?.strip_prefix('<')?;
    if let Some(comment) = rest.strip_prefix("!--") {
        return Some(Tag {
            start,
            end: start + 4 + comment.find("-->")? + 3,
            name: "!",
            closing: false,
        });
    }
    let mut quote = None;
    let end = rest.char_indices().find_map(|(index, character)| {
        if quote == Some(character) {
            quote = None;
        } else if quote.is_none() {
            if matches!(character, '\'' | '"') {
                quote = Some(character);
            } else if character == '>' {
                return Some(start + index + 2);
            }
        }
        None
    })?;
    let closing = rest.starts_with('/');
    let name = rest.trim_start_matches('/');
    let length = name
        .find(|character: char| !character.is_ascii_alphanumeric())
        .unwrap_or(name.len());
    Some(Tag {
        start,
        end,
        name: &name[..length],
        closing,
    })
}

fn block(name: &str) -> bool {
    matches!(
        name,
        "p" | "div"
            | "td"
            | "th"
            | "tr"
            | "li"
            | "ul"
            | "ol"
            | "table"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "hr"
            | "section"
            | "article"
            | "header"
            | "footer"
    )
}

struct Visible<'a> {
    rest: &'a str,
    pending: Option<char>,
    started: bool,
}

impl<'a> Visible<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            rest: text,
            pending: None,
            started: false,
        }
    }

    fn raw_next(&mut self) -> Option<char> {
        loop {
            if let Some(tag) = tag_at(self.rest, 0) {
                self.rest = &self.rest[tag.end..];
            } else {
                let character = self.rest.chars().next()?;
                self.rest = &self.rest[character.len_utf8()..];
                return Some(character);
            }
        }
    }
}

impl Iterator for Visible<'_> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(character) = self.pending.take() {
            return Some(character);
        }
        let mut separated = false;
        while let Some(character) = self.raw_next() {
            if character.is_whitespace() {
                separated |= self.started;
            } else {
                if separated {
                    self.pending = Some(character);
                    return Some(' ');
                }
                self.started = true;
                return Some(character);
            }
        }
        None
    }
}

fn push_field(text: &str, start: usize, end: usize, fields: &mut Vec<Field>) {
    let value = &text[start..end];
    let mut words = 0;
    let mut characters = 0;
    let mut in_word = false;
    let mut numeric = false;
    let mut currency = false;
    let mut last = None;
    for character in Visible::new(value) {
        characters += 1;
        let word = !character.is_whitespace();
        words += usize::from(word && !in_word);
        in_word = word;
        numeric |= character.is_numeric();
        currency |= matches!(character, '€' | '$' | '£' | '¥');
        last = Some(character);
    }
    if words == 0 {
        return;
    }
    currency |= ["&euro;", "&pound;", "&yen;", " eur", " usd", " gbp"]
        .iter()
        .any(|unit| value.contains(unit));
    let sentence = words >= 3 && matches!(last, Some('.' | '!' | '?'));
    fields.push(Field {
        start,
        end,
        words,
        short: words <= MAX_FIELD_WORDS && characters <= MAX_FIELD_CHARACTERS && !sentence,
        anchor: super::template::code_field_before(&text[..start])
            || (numeric && (currency || super::template::amount_field_before(&text[..start]))),
    });
}

/// Cache borrowed ranges once per file; candidate verification never copies field text.
pub(super) fn extract(text: &str) -> Vec<Field> {
    let mut fields = Vec::new();
    let mut position = 0;
    let mut start = 0;
    let mut hidden = None;
    let mut structured = false;
    while let Some(relative) = text[position..].find('<') {
        let Some(tag) = tag_at(text, position + relative) else {
            break;
        };
        position = tag.end;
        if let Some(name) = hidden {
            if tag.closing && tag.name == name {
                hidden = None;
                start = tag.end;
            }
            continue;
        }
        if matches!(tag.name, "head" | "style" | "script") && !tag.closing {
            push_field(text, start, tag.start, &mut fields);
            hidden = Some(tag.name);
            start = tag.end;
        } else if block(tag.name) {
            structured = true;
            push_field(text, start, tag.start, &mut fields);
            start = tag.end;
        }
    }
    if structured && hidden.is_none() {
        push_field(text, start, text.len(), &mut fields);
        fields
    } else {
        Vec::new()
    }
}

fn equal(first_text: &str, first: &Field, second_text: &str, second: &Field) -> bool {
    Visible::new(&first_text[first.start..first.end])
        .eq(Visible::new(&second_text[second.start..second.end]))
}

fn resync(
    first_text: &str,
    first: &[Field],
    second_text: &str,
    second: &[Field],
) -> Option<(usize, usize)> {
    // Search only bounded runs of short fields. A changed sentence or paragraph
    // cannot be skipped, even if a nearby booking number or amount also changed.
    for total in 1..=2 * MAX_FIELD_RUN {
        for left in 0..=MAX_FIELD_RUN.min(total).min(first.len()) {
            let right = total - left;
            if right > MAX_FIELD_RUN || right > second.len() {
                continue;
            }
            if !first[..left].iter().all(|field| field.short)
                || !second[..right].iter().all(|field| field.short)
            {
                continue;
            }
            let aligned = match (first.get(left), second.get(right)) {
                (Some(a), Some(b)) => equal(first_text, a, second_text, b),
                (None, None) => true,
                _ => false,
            };
            if aligned {
                return Some((left, right));
            }
        }
    }
    None
}

/// Form alignment needs twenty fixed words and a changed anchor or typed ID-like value.
pub(super) fn same_template(
    first_text: &str,
    mut first: &[Field],
    second_text: &str,
    mut second: &[Field],
    typed_identity_change: bool,
) -> bool {
    if first.len() < 3 || second.len() < 3 {
        return false;
    }
    let mut changed_anchor = false;
    let mut fixed_words = 0;
    while !first.is_empty() || !second.is_empty() {
        if let (Some(a), Some(b)) = (first.first(), second.first())
            && equal(first_text, a, second_text, b)
        {
            fixed_words += a.words;
            first = &first[1..];
            second = &second[1..];
            continue;
        }
        let Some((left, right)) = resync(first_text, first, second_text, second) else {
            return false;
        };
        changed_anchor |= first[..left].iter().any(|a| {
            a.anchor
                && second[..right]
                    .iter()
                    .any(|b| b.anchor && !equal(first_text, a, second_text, b))
        });
        first = &first[left..];
        second = &second[right..];
    }
    (changed_anchor || typed_identity_change) && fixed_words >= MIN_FIXED_WORDS
}
