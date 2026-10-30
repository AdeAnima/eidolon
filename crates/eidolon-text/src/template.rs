//! Exact template alignment after masking typed data fields, not prose.

enum Token<'a> {
    Literal(&'a str),
    Data { value: &'a str, identity: bool },
}

struct Tokens<'a> {
    text: &'a str,
    position: usize,
    url_end: usize,
    query: bool,
    query_value: bool,
}

impl<'a> Tokens<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            position: 0,
            url_end: 0,
            query: false,
            query_value: false,
        }
    }

    fn literal(&mut self, length: usize) -> Token<'a> {
        let start = self.position;
        self.position += length;
        Token::Literal(&self.text[start..self.position])
    }
}

impl<'a> Iterator for Tokens<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let rest = &self.text[self.position..];
        if self.position >= self.url_end {
            self.query = false;
            self.query_value = false;
        }
        if rest.starts_with("https://") || rest.starts_with("http://") {
            self.url_end = self.position
                + rest
                    .find(|character: char| {
                        character.is_whitespace() || matches!(character, '<' | '>' | '\'' | '"')
                    })
                    .unwrap_or(rest.len());
        }
        if self.position < self.url_end {
            if self.query_value {
                let start = self.position;
                let length = self.text[self.position..self.url_end]
                    .find(['&', '#'])
                    .unwrap_or(self.url_end - self.position);
                self.position += length;
                self.query_value = false;
                return Some(Token::Data {
                    value: &self.text[start..self.position],
                    identity: true,
                });
            }
            if rest.starts_with('?') {
                self.query = true;
            } else if self.query && rest.starts_with('=') {
                self.query_value = true;
            } else if rest.starts_with('#') {
                self.query = false;
            }
        }
        let character = rest.chars().next()?;
        if !is_word(character) {
            return Some(self.literal(character.len_utf8()));
        }
        if self.position >= self.url_end
            && let Some(length) = email_length(rest)
        {
            self.position += length;
            return Some(Token::Data {
                value: &rest[..length],
                identity: true,
            });
        }
        let mut length = rest
            .find(|character| !is_word(character))
            .unwrap_or(rest.len());
        if rest[..length].chars().all(char::is_numeric) {
            // Keep decimal/thousands components in one value, so a cents-only
            // change retains the label before the entire amount.
            while matches!(rest.as_bytes().get(length), Some(b'.' | b','))
                && rest[length + 1..]
                    .chars()
                    .next()
                    .is_some_and(char::is_numeric)
            {
                length += 1;
                length += rest[length..]
                    .find(|character: char| !character.is_numeric())
                    .unwrap_or(rest.len() - length);
            }
        }
        let word = &rest[..length];
        let numeric = word.chars().any(char::is_numeric);
        let hexadecimal = word.len() >= 8
            && word.bytes().all(|byte| byte.is_ascii_hexdigit())
            && word.bytes().any(|byte| byte.is_ascii_alphabetic());
        let code = code_field_before(&self.text[..self.position]);
        let identity = hexadecimal || code || (numeric && word.chars().any(char::is_alphabetic));
        let data = numeric
            || hexadecimal
            || code
            || (is_date_name(word)
                && (number_before(&self.text[..self.position]) || number_after(&rest[length..])));
        if data {
            self.position += length;
            Some(Token::Data {
                value: word,
                identity,
            })
        } else {
            Some(self.literal(length))
        }
    }
}

fn is_word(character: char) -> bool {
    character.is_alphanumeric()
        || unicode_normalization::char::is_combining_mark(character)
        || matches!(character, '_' | '-')
}

fn email_length(text: &str) -> Option<usize> {
    let length = text
        .find(|character: char| {
            character.is_whitespace()
                || matches!(
                    character,
                    '<' | '>' | '\'' | '"' | '(' | ')' | ',' | ';' | ':' | '?' | '&' | '=' | '/'
                )
        })
        .unwrap_or(text.len());
    let address = &text[..length];
    let (local, domain) = address.split_once('@')?;
    (!local.is_empty()
        && domain.contains('.')
        && local.chars().all(|character| {
            character.is_alphanumeric() || matches!(character, '.' | '_' | '-' | '+' | '%')
        })
        && domain
            .chars()
            .all(|character| character.is_alphanumeric() || matches!(character, '.' | '-')))
    .then_some(length)
}

fn label_before(mut text: &str) -> Option<&str> {
    let mut labelled = false;
    loop {
        text = text.trim_end();
        if let Some(before) = text
            .strip_suffix(':')
            .or_else(|| text.strip_suffix('='))
            .or_else(|| text.strip_suffix('#'))
            .or_else(|| {
                let before = text.strip_suffix('.')?;
                matches!(
                    before.rsplit(|character| !is_word(character)).next(),
                    Some("no" | "nr")
                )
                .then_some(before)
            })
        {
            labelled = true;
            text = before;
        } else if let Some(before) = text
            .strip_suffix('€')
            .or_else(|| text.strip_suffix('$'))
            .or_else(|| text.strip_suffix('£'))
            .or_else(|| text.strip_suffix('¥'))
        {
            text = before;
        } else if let Some((before, "eur" | "usd" | "gbp" | "jpy")) = text.rsplit_once(' ') {
            text = before;
        } else if text.ends_with('>') {
            let start = text.rfind('<')?;
            labelled |= text[start..].starts_with("</");
            text = &text[..start];
        } else {
            break;
        }
    }
    labelled.then_some(text)
}

/// Explicit identifier labels distinguish document data from ordinary prose.
pub(super) fn code_field_before(text: &str) -> bool {
    let Some(label) = label_before(text) else {
        return false;
    };
    let mut words = label
        .rsplit(|character| !is_word(character))
        .filter(|word| !word.is_empty());
    let last = words.next();
    matches!(
        last,
        Some(
            "id" | "code"
                | "reference"
                | "transaction"
                | "receipt"
                | "invoice"
                | "order"
                | "rechnung"
                | "bestellung"
                | "transaction-id"
                | "transaction_id"
                | "order-id"
                | "order_id"
                | "booking-id"
                | "booking_id"
                | "bestellnummer"
                | "buchungsnummer"
                | "reservierungsnummer"
                | "bestätigungsnummer"
                | "transaktionscode"
                | "transaktions-id"
                | "referenz"
                | "referenznummer"
                | "auftragsnummer"
                | "rechnungsnummer"
                | "belegnummer"
        )
    ) || (matches!(last, Some("number" | "no" | "nr"))
        && matches!(
            words.next(),
            Some(
                "booking"
                    | "reservation"
                    | "confirmation"
                    | "invoice"
                    | "receipt"
                    | "transaction"
                    | "order"
                    | "rechnung"
                    | "bestellung"
            )
        ))
}

/// Amount labels are anchors only when their field also contains a number.
pub(super) fn amount_field_before(text: &str) -> bool {
    label_before(text).is_some_and(|label| {
        matches!(
            label.rsplit(|character| !is_word(character)).next(),
            Some(
                "amount"
                    | "total"
                    | "betrag"
                    | "gesamt"
                    | "gesamtbetrag"
                    | "summe"
                    | "price"
                    | "preis"
            )
        )
    })
}

fn date_separator(character: char) -> bool {
    character.is_whitespace() || matches!(character, '.' | ',' | '-' | '/' | ':')
}

fn number_before(text: &str) -> bool {
    text.trim_end_matches(date_separator)
        .chars()
        .next_back()
        .is_some_and(char::is_numeric)
}

fn number_after(text: &str) -> bool {
    text.trim_start_matches(date_separator)
        .chars()
        .next()
        .is_some_and(char::is_numeric)
}

fn is_date_name(word: &str) -> bool {
    matches!(
        word,
        "january"
            | "januar"
            | "jan"
            | "february"
            | "februar"
            | "feb"
            | "march"
            | "märz"
            | "mar"
            | "mär"
            | "mrz"
            | "april"
            | "apr"
            | "may"
            | "mai"
            | "june"
            | "juni"
            | "jun"
            | "july"
            | "juli"
            | "jul"
            | "august"
            | "aug"
            | "september"
            | "sep"
            | "sept"
            | "october"
            | "oktober"
            | "oct"
            | "okt"
            | "november"
            | "nov"
            | "december"
            | "dezember"
            | "dec"
            | "dez"
            | "am"
            | "pm"
    )
}

fn labelled_data_before(text: &str) -> bool {
    code_field_before(text) || amount_field_before(text)
}

/// Exact typed alignment still needs a changed labelled field or HTML field proof.
pub(super) struct Alignment {
    pub(super) labelled_change: bool,
    pub(super) identity_change: bool,
}

/// Both complete normalized texts must align; ordinary word edits remain significant.
/// Borrowed token streams retain data values so unchanged fields cannot supply proof.
pub(super) fn same_template(first: &str, second: &str) -> Option<Alignment> {
    let mut first_tokens = Tokens::new(first);
    let mut second_tokens = Tokens::new(second);
    let mut labelled_change = false;
    let mut identity_change = false;
    loop {
        let first_start = first_tokens.position;
        let second_start = second_tokens.position;
        match (first_tokens.next(), second_tokens.next()) {
            (Some(Token::Literal(a)), Some(Token::Literal(b))) if a == b => {}
            (
                Some(Token::Data {
                    value: a,
                    identity: a_identity,
                }),
                Some(Token::Data {
                    value: b,
                    identity: b_identity,
                }),
            ) => {
                labelled_change |= a != b
                    && labelled_data_before(&first[..first_start])
                    && labelled_data_before(&second[..second_start]);
                identity_change |= a != b && a_identity && b_identity;
            }
            (None, None) => {
                return Some(Alignment {
                    labelled_change,
                    identity_change,
                });
            }
            _ => return None,
        }
    }
}
