//! Public-boundary tests using files containing real toolchain license prose.

use std::path::Path;
use std::process::Command;

use eidolon_core::cache::FeatureCache;
use eidolon_core::{Config, Ctx, FileEntry, FileId, Kind, Match, Relation};

fn license(name: &str) -> String {
    let output = Command::new("rustc")
        .arg("--print")
        .arg("sysroot")
        .output()
        .expect("rustc");
    assert!(output.status.success());
    let root = String::from_utf8(output.stdout).expect("UTF-8 sysroot");
    std::fs::read_to_string(
        Path::new(root.trim())
            .join("share/doc/rust/licenses")
            .join(name),
    )
    .expect("toolchain license")
}

fn run_bytes(inputs: &[Vec<u8>], min_score: f64) -> Vec<Match> {
    let dir = tempfile::Builder::new()
        .prefix("eidolon-text-")
        .tempdir_in("/tmp")
        .expect("tempdir");
    let files: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(id, bytes)| {
            let path = dir.path().join(format!("{id}.txt"));
            std::fs::write(&path, bytes).expect("write input");
            FileEntry {
                id: FileId(id),
                path,
                size: bytes.len() as u64,
                mtime_ns: 0,
                kind: Kind::Text,
            }
        })
        .collect();
    let config = Config {
        timings: false,
        remote_overlay: true,
        min_score,
        model_dir: dir.path().into(),
        no_embeddings: true,
        no_keypoints: true,
        jobs: 2,
        fast: false,
        video_fps: 2.0,
    };
    let cache = FeatureCache::in_memory().expect("cache");
    let ctx = Ctx {
        config: &config,
        cache: &cache,
        embedder: None,
        refiner: None,
    };
    let result = eidolon_text::find(&files, &ctx).expect("detect text");
    let cached = eidolon_text::find(&files, &ctx).expect("cached detection");
    assert_eq!(
        serde_json::to_value(&result).expect("serialize"),
        serde_json::to_value(cached).expect("serialize")
    );
    for found in &result {
        assert!(!found.evidence.is_empty());
        assert!(found.score >= min_score && found.score <= 1.0);
        if !found.relation.is_directional() {
            assert!(found.a < found.b);
        }
    }
    result
}

fn run(inputs: &[&str]) -> Vec<Match> {
    run_bytes(
        &inputs
            .iter()
            .map(|text| text.as_bytes().to_vec())
            .collect::<Vec<_>>(),
        0.5,
    )
}

#[test]
fn crlf_rewrapped_license_is_same_content() {
    let original = license("Apache-2.0.txt");
    let rewrapped = format!(
        "\u{feff}{}  \r\n",
        original
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" \r\n  ")
    );
    let matches = run(&[&original, &rewrapped]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::SameContent);
}

#[test]
fn fifth_of_real_license_is_directional_excerpt() {
    let original = license("GPL-3.0-or-later.txt");
    let words: Vec<_> = original.split_whitespace().collect();
    let excerpt = words[words.len() / 5..2 * words.len() / 5].join(" ");
    let matches = run(&[&original, &excerpt]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::ExcerptOf);
    assert_eq!((matches[0].a, matches[0].b), (FileId(1), FileId(0)));
}

#[test]
fn short_verbatim_license_passage_is_directional_excerpt() {
    let original = license("Apache-2.0.txt");
    let words: Vec<_> = original.split_whitespace().collect();
    for length in [19, 39] {
        let excerpt = words[100..100 + length].join(" ");
        let matches = run(&[&original, &excerpt]);
        assert_eq!(matches.len(), 1, "{length}-word passage");
        assert_eq!(matches[0].relation, Relation::ExcerptOf);
        assert_eq!((matches[0].a, matches[0].b), (FileId(1), FileId(0)));
    }
}

#[test]
fn tiny_edited_and_repetitive_passages_are_not_excerpts() {
    let original = license("Apache-2.0.txt");
    let words: Vec<_> = original.split_whitespace().collect();
    let tiny = words[100..114].join(" ");
    let mut edited = words[100..119].to_vec();
    edited[9] = "unrelatedreplacement";
    let edited = edited.join(" ");
    for passage in [&tiny, &edited] {
        assert!(run(&[&original, passage]).is_empty());
    }
    let repeated = "permission ".repeat(19);
    let whole = format!("{original} {repeated}");
    assert!(run(&[&whole, &repeated]).is_empty());
}

#[test]
fn a_few_edited_sentences_are_edited() {
    let original = license("Apache-2.0.txt");
    let edited = original
        .replace("\"You\" (or \"Your\") shall mean an individual or Legal Entity exercising permissions granted by this License.", "A recipient is any person or legal entity using these permissions.")
        .replace("1. Definitions.", "1. Updated definitions.")
        .replace("Copyright [yyyy] [name of copyright owner]", "Copyright 2026 Example Authors");
    assert_ne!(original, edited);
    let matches = run(&[&original, &edited]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::Edited);
}

fn man_page(name: &str) -> String {
    let output = Command::new("man")
        .args(["-P", "cat", name])
        .env("MANWIDTH", "100")
        .env("TERM", "dumb")
        .output()
        .expect("man");
    assert!(output.status.success(), "man page {name}");
    let rendered = String::from_utf8_lossy(&output.stdout);
    let mut plain = String::new();
    for character in rendered.chars() {
        if character == '\u{8}' {
            plain.pop();
        } else {
            plain.push(character);
        }
    }
    assert!(plain.split_whitespace().count() > 200);
    plain
}

#[test]
fn unrelated_man_pages_do_not_match() {
    assert!(run(&[&man_page("ls"), &man_page("cat")]).is_empty());
}

#[test]
fn common_license_header_does_not_match_unrelated_bodies() {
    let header = license("MIT.txt");
    let first = format!("{header}\n\n{}", man_page("ls"));
    let second = format!("{header}\n\n{}", man_page("cat"));
    assert!(run(&[&first, &second]).is_empty());
}

#[test]
fn utf16_boms_case_folding_and_compatibility_normalization_match() {
    let original = "Straße ﬁle ＡＢＣ café\n";
    let normalized = "STRASSE file ABC cafe\u{301}\r\n";
    for little_endian in [true, false] {
        let mut encoded = if little_endian {
            vec![0xff, 0xfe]
        } else {
            vec![0xfe, 0xff]
        };
        encoded.extend(original.encode_utf16().flat_map(|unit| {
            if little_endian {
                unit.to_le_bytes()
            } else {
                unit.to_be_bytes()
            }
        }));
        let matches = run_bytes(&[encoded, normalized.as_bytes().to_vec()], 1.0);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].relation, Relation::SameContent);
    }
}

#[test]
fn invalid_utf8_decodes_lossily_without_failing() {
    let matches = run_bytes(
        &[
            b"invalid \xff byte".to_vec(),
            "invalid \u{fffd} byte".as_bytes().to_vec(),
        ],
        0.5,
    );
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::SameContent);
}

#[test]
fn short_boilerplate_overlap_is_not_a_near_match() {
    assert!(
        run(&[
            "Permission is hereby granted free of charge to use this file in the Alpha project.",
            "Permission is hereby granted free of charge to use this file in the Beta project.",
        ])
        .is_empty()
    );
}

#[test]
fn punctuation_is_ignored_for_shingles_not_for_equality() {
    let original = license("Apache-2.0.txt");
    let stripped: String = original
        .chars()
        .filter(|character| character.is_alphanumeric() || character.is_whitespace())
        .collect();
    let matches = run(&[&original, &stripped]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::Edited);
}

#[test]
fn minimum_score_filters_verified_similarities() {
    let original = license("Apache-2.0.txt");
    let edited = original.replace("License", "Permission");
    assert!(!run(&[&original, &edited]).is_empty());
    assert!(run_bytes(&[original.into_bytes(), edited.into_bytes()], 1.0).is_empty());
}

#[test]
fn distributed_edits_at_moderate_overlap_are_similar() {
    let original = license("Apache-2.0.txt");
    let edited = original
        .split_whitespace()
        .enumerate()
        .map(|(index, word)| {
            if index.is_multiple_of(20) {
                "replacementtoken"
            } else {
                word
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    let matches = run(&[&original, &edited]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::Similar);
}

#[test]
fn unspaced_text_uses_character_shingles_for_excerpt_detection() {
    let original: String = license("GPL-3.0-or-later.txt")
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let chars: Vec<_> = original.chars().collect();
    let excerpt: String = chars[chars.len() / 5..2 * chars.len() / 5].iter().collect();
    let matches = run(&[&original, &excerpt]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::ExcerptOf);
    assert_eq!((matches[0].a, matches[0].b), (FileId(1), FileId(0)));
}

#[test]
fn changed_file_identity_invalidates_cached_features() {
    let dir = tempfile::Builder::new()
        .prefix("eidolon-text-")
        .tempdir_in("/tmp")
        .expect("tempdir");
    let original = license("Apache-2.0.txt");
    let files: Vec<_> = (0..2)
        .map(|id| {
            let path = dir.path().join(format!("{id}.txt"));
            std::fs::write(&path, &original).expect("write input");
            FileEntry {
                id: FileId(id),
                path,
                size: original.len() as u64,
                mtime_ns: 0,
                kind: Kind::Text,
            }
        })
        .collect();
    let config = Config {
        timings: false,
        remote_overlay: true,
        min_score: 0.5,
        model_dir: dir.path().into(),
        no_embeddings: true,
        no_keypoints: true,
        jobs: 1,
        fast: false,
        video_fps: 2.0,
    };
    let cache = FeatureCache::in_memory().expect("cache");
    let ctx = Ctx {
        config: &config,
        cache: &cache,
        embedder: None,
        refiner: None,
    };
    let matches = eidolon_text::find(&files, &ctx).expect("initial detection");
    assert_eq!(matches.len(), 1);
    let unrelated = man_page("cat");
    std::fs::write(&files[1].path, &unrelated).expect("replace file");
    let mut changed = files;
    changed[1].size = unrelated.len() as u64;
    changed[1].mtime_ns = 1;
    assert!(
        eidolon_text::find(&changed, &ctx)
            .expect("fresh detection")
            .is_empty()
    );
}

#[test]
fn empty_and_whitespace_only_inputs_are_ignored() {
    assert!(run(&["", " \t\r\n", ""]).is_empty());
}

#[test]
fn vowel_marks_remain_significant_for_shingles() {
    let art = "कला ".repeat(30);
    let fort = "किला ".repeat(30);
    assert!(run(&[&art, &fort]).is_empty());
}

#[test]
fn cjk_paragraphs_with_minor_edits_use_character_shingles() {
    let original = [
        "夜色降临以后，研究人员打开天文台的屋顶，检查望远镜的镜面和追踪系统，准备记录遥远星系发出的微弱光线。",
        "山谷里的空气逐渐冷却，城市的灯光在地平线附近形成一层薄雾，而高处的观测平台依然保持着清晰的视野。",
        "负责校准的工程师比较昨天和今天的数据，确认温度变化没有影响设备，并把每一步操作写进详细的工作记录。",
        "在漫长的曝光期间，另一组学生讨论恒星形成的过程，他们希望通过连续几周的观察找到亮度变化的规律。",
        "午夜过后，云层从西方靠近，值班人员重新计算剩余的观测时间，优先完成最容易受到天气影响的研究项目。",
        "清晨收工之前，团队保存原始图像并制作备份，随后整理异常现象和下一次需要检查的问题，等待太阳升起。",
    ].join("\n\n");
    let edited = original.replace("望远镜", "观测仪");
    let matches = run(&[&original, &edited]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::Edited);
}

#[test]
fn uncomposed_accents_remain_significant_for_shingles() {
    let acute = "q\u{301} ".repeat(30);
    let grave = "q\u{300} ".repeat(30);
    assert!(run(&[&acute, &grave]).is_empty());
}

#[test]
fn finds_excerpt_amid_two_thousand_distinct_document_records() {
    let source = license("Apache-2.0.txt");
    let words: Vec<_> = source.split_whitespace().take(80).collect();
    let mut inputs: Vec<_> = (0..2000)
        .map(|index| {
            words
                .iter()
                .map(|word| format!("record{index}_{word}"))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    let whole = license("GPL-3.0-or-later.txt");
    let full_words: Vec<_> = whole.split_whitespace().collect();
    let excerpt = full_words[full_words.len() / 5..2 * full_words.len() / 5].join(" ");
    inputs.push(whole);
    inputs.push(excerpt);
    let matches = run(&inputs.iter().map(String::as_str).collect::<Vec<_>>());
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::ExcerptOf);
    assert_eq!((matches[0].a, matches[0].b), (FileId(2001), FileId(2000)));
}

#[test]
fn common_license_anchors_do_not_hide_a_discriminative_passage() {
    let license_text = license("GPL-3.0-or-later.txt");
    let header = license_text
        .split_whitespace()
        .take(800)
        .collect::<Vec<_>>()
        .join(" ");
    let other_license = license("Apache-2.0.txt");
    let passage = other_license
        .split_whitespace()
        .skip(100)
        .take(19)
        .collect::<Vec<_>>()
        .join(" ");
    let excerpt = format!("{header} {passage}");
    let context = license_text
        .split_whitespace()
        .cycle()
        .take(10_000)
        .enumerate()
        .map(|(index, word)| format!("context{index}_{word}"))
        .collect::<Vec<_>>()
        .join(" ");
    let whole = format!("{context} {excerpt}");
    let mut inputs = vec![header.as_str(); 34];
    inputs.extend([whole.as_str(), excerpt.as_str()]);
    let matches = run(&inputs);
    assert!(matches.iter().any(|found| {
        found.relation == Relation::ExcerptOf && (found.a, found.b) == (FileId(35), FileId(34))
    }));
}

#[test]
fn unreadable_single_input_has_a_recoverable_file_identity() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let files = [FileEntry {
        id: FileId(7),
        path: dir.path().join("removed.txt"),
        size: 32,
        mtime_ns: 0,
        kind: Kind::Text,
    }];
    let config = Config {
        timings: false,
        remote_overlay: true,
        min_score: 0.5,
        model_dir: dir.path().into(),
        no_embeddings: true,
        no_keypoints: true,
        jobs: 1,
        fast: false,
        video_fps: 2.0,
    };
    let cache = FeatureCache::in_memory()?;
    let error = eidolon_text::find(
        &files,
        &Ctx {
            config: &config,
            cache: &cache,
            embedder: None,
            refiner: None,
        },
    )
    .expect_err("missing text must fail extraction, even without another input");
    let input = error
        .downcast_ref::<eidolon_core::InputError>()
        .ok_or_else(|| anyhow::anyhow!("read failure must identify its input: {error:#}"))?;
    assert_eq!(input.file, FileId(7));
    assert!(input.reason.contains("removed.txt"));
    Ok(())
}

fn receipt(amount: &str, date: &str, transaction: &str) -> String {
    format!(
        "<html><head><title>Example payment receipt</title></head><body>\n\
         <h1>Thank you for your payment</h1>\n\
         <p>This receipt confirms that your payment has been completed successfully. \
         You can review the details below and keep this document for your records. \
         The payment service has sent a confirmation to the merchant. \
         Contact the merchant directly if you have questions about the goods or services. \
         Your account activity contains the complete history of this payment. \
         Never share your password with anyone claiming to represent the payment service.</p>\n\
         <p>Amount: {amount} EUR</p>\n\
         <p>Date: {date}</p>\n\
         <p>Transaction: {transaction}</p>\n\
         <p>Please retain this receipt as proof of payment. \
         This message was generated automatically and replies are not monitored.</p>\n\
         </body></html>"
    )
}

#[test]
fn receipts_with_different_data_are_variants_not_duplicates() {
    let first = receipt("12.50", "14 July 2026 09:31", "AB1234CD5678");
    let second = receipt("18.75", "21 August 2026 16:42", "XY9876ZW5432");
    let matches = run(&[&first, &second]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::Variant);
    assert!(
        matches[0]
            .evidence
            .iter()
            .any(|evidence| evidence.note == "same template, different data")
    );
}

#[test]
fn email_hex_codes_and_url_query_values_are_template_data() {
    let common = receipt("12.50", "14 July 2026 09:31", "AB1234CD5678");
    for (first_field, second_field) in [
        ("alice@example.invalid", "bob@other.invalid"),
        ("deadbeef", "facefeed"),
        ("INV-AB123-XY", "INV-ZZ987-PQ"),
        ("Reference: ABCDEFHIJK", "Reference: LMNOPQRSTU"),
        (
            "<table><tr><td>Bestellnummer</td><td><span>ride-abcdefghijklm</span></td></tr></table>",
            "<table><tr><td>Bestellnummer</td><td><span>ride-1234567890123</span></td></tr></table>",
        ),
        (
            "<a href=\"https://example.invalid/receipt?reference=alphabetic&amp;email=alice%40example.invalid\">Details</a>",
            "<a href=\"https://example.invalid/receipt?reference=other-value&amp;email=bob%40other.invalid\">Details</a>",
        ),
    ] {
        let first = format!("{common}<p>{first_field}</p>");
        let second = format!("{common}<p>{second_field}</p>");
        let matches = run(&[&first, &second]);
        assert_eq!(matches.len(), 1, "{first_field}");
        assert_eq!(matches[0].relation, Relation::Variant, "{first_field}");
    }
}

#[test]
fn ordinary_word_edits_remain_edited_duplicates() {
    let original = "The team followed the narrow path through the forest before sunrise. \
        Every member carried a notebook and recorded the plants growing beside the trail. \
        Near the river they stopped to compare their observations with the previous survey. \
        The water moved quietly under the old wooden bridge while birds gathered in the reeds. \
        After a short break the group continued toward the clearing where the instruments \
        had been installed earlier in the season. They checked the equipment carefully, \
        saved the measurements, and discussed the results before returning to the station. \
        Their report described the changing landscape and explained why regular observations \
        were important for understanding the health of the forest.";
    let edited = original.replace("narrow path", "winding trail");
    let matches = run(&[original, &edited]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::Edited);
}

#[test]
fn ordinary_numeric_prose_edits_remain_edited_duplicates() {
    let original = "In 2025 the research team followed the narrow path through the forest before sunrise. \
        Every member carried a notebook and recorded the plants growing beside the trail. \
        Near the river they stopped to compare their observations with the previous survey. \
        The water moved quietly under the old wooden bridge while birds gathered in the reeds. \
        They inspected the clearing and counted the total. 120 plants were recorded beside 24 instruments. \
        After a short break the group carefully saved the measurements and discussed the results \
        before returning to the station. Their report described the changing landscape and \
        explained why regular observations were important for understanding the health of the forest.";
    for edited in [
        original.replace("2025", "2026"),
        original.replace("120", "130"),
        original.replace("120", "130").replace("24", "25"),
        original.replace("120", "130.5"),
    ] {
        for (first, second) in [
            (original.to_owned(), edited.clone()),
            (format!("<p>{original}</p>"), format!("<p>{edited}</p>")),
        ] {
            let matches = run(&[&first, &second]);
            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].relation, Relation::Edited);
            assert!(
                matches[0]
                    .evidence
                    .iter()
                    .all(|evidence| evidence.signal != "text_template_alignment")
            );
        }
    }
}

#[test]
fn explicitly_labelled_plain_text_receipt_data_remains_variant() {
    for label in ["Transaction:", "Receipt #"] {
        let original = format!(
            "{label} AB1234CD5678\nAmount: 12.50 EUR\n{}",
            license("MIT.txt")
        );
        for edited in [
            original.replace("AB1234CD5678", "XY9876ZW5432"),
            original.replace("12.50", "18.75"),
        ] {
            let matches = run(&[&original, &edited]);
            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].relation, Relation::Variant);
        }
    }
}

#[test]
fn numeric_html_footer_updates_remain_edited() {
    let common = format!(
        "<html><body><header>Field observations</header><article><p>{}</p></article>",
        license("MIT.txt")
    );
    for (before, after) in [
        ("Copyright 2025", "Copyright 2026"),
        ("Updated 3 October 2026", "Updated 4 November 2026"),
        ("Visitors: 120", "Visitors: 130"),
        ("Visitors: 12345678", "Visitors: 12345679"),
    ] {
        let first = format!("{common}<footer>{before}</footer></body></html>");
        let second = format!("{common}<footer>{after}</footer></body></html>");
        let matches = run(&[&first, &second]);
        assert_eq!(matches.len(), 1, "{before}");
        assert_eq!(matches[0].relation, Relation::Edited, "{before}");
        assert!(
            matches[0]
                .evidence
                .iter()
                .all(|evidence| evidence.signal != "text_template_alignment")
        );
    }
}

#[test]
fn plain_receipt_amounts_keep_decimal_and_currency_field_context() {
    let common = license("MIT.txt");
    for label in ["Amount", "Total", "Gesamt", "Price", "Preis"] {
        for (before, after) in [
            ("12.50", "12.75"),
            ("12,50", "12,75"),
            ("€12.50", "€18.75"),
            ("€12.50", "€12.75"),
            ("€12,50", "€12,75"),
            ("EUR 12.50", "EUR 12.75"),
            ("USD 12.50", "USD 12.75"),
            ("GBP 12.50", "GBP 12.75"),
        ] {
            let first = format!("{label}: {before}\n{common}");
            let second = format!("{label}: {after}\n{common}");
            let matches = run(&[&first, &second]);
            assert_eq!(matches.len(), 1, "{label}: {before}");
            // Variant edges never join duplicate groups in the CLI.
            assert_eq!(matches[0].relation, Relation::Variant, "{label}: {before}");
        }
    }
}

#[test]
fn english_and_german_receipt_identifier_labels_are_template_evidence() {
    let common = license("MIT.txt");
    for label in [
        "Receipt number:",
        "Invoice No.",
        "Rechnung Nr.",
        "Belegnummer:",
    ] {
        for (before, after) in [("123456", "987654"), ("ALPHACODE", "OMEGACODE")] {
            let first = format!("{label} {before}\n{common}");
            let second = format!("{label} {after}\n{common}");
            let matches = run(&[&first, &second]);
            assert_eq!(matches.len(), 1, "{label}");
            assert_eq!(matches[0].relation, Relation::Variant, "{label}");
        }
    }
}

#[test]
fn hash_prefixed_document_identifiers_are_template_evidence() {
    let common = license("MIT.txt");
    for label in ["Invoice", "Order", "Rechnung", "Bestellung"] {
        for (before, after) in [("4711", "4712"), ("ALPHACODE", "OMEGACODE")] {
            let first = format!("{label} #{before}\n{common}");
            let second = format!("{label} #{after}\n{common}");
            let matches = run(&[&first, &second]);
            assert_eq!(matches.len(), 1, "{label}");
            assert_eq!(matches[0].relation, Relation::Variant, "{label}");
        }
    }
}

#[test]
fn receipt_resaved_with_crlf_is_still_a_duplicate() {
    let original = receipt("12.50", "14 July 2026 09:31", "AB1234CD5678");
    let resaved = original.replace('\n', "\r\n");
    let matches = run(&[&original, &resaved]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::SameContent);
    assert_eq!(matches[0].score, 1.0);
}

#[test]
fn changing_prose_inside_a_receipt_is_not_a_template_variant() {
    let original = receipt("12.50", "14 July 2026 09:31", "AB1234CD5678");
    let edited = original.replace("complete history", "full record");
    let matches = run(&[&original, &edited]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::Edited);
}

fn booking_receipt(
    merchant: &str,
    street: &str,
    postcode: &str,
    date: &str,
    amount: &str,
    number: &str,
) -> String {
    format!(
        "{}<table>\
         <tr><td>Booking number</td><td>{number}</td></tr>\
         <tr><td>Property</td><td>{merchant}</td></tr>\
         <tr><td>Address</td><td>{street}</td></tr>\
         <tr><td>Postcode</td><td>{postcode}</td></tr>\
         <tr><td>Arrival</td><td>{date}</td></tr>\
         <tr><td>Total</td><td>{amount} EUR</td></tr>\
         </table>",
        receipt("12.50", "14 July 2026 09:31", "AB1234CD5678")
    )
}

#[test]
fn different_merchants_addresses_and_booking_data_are_variants() {
    let first = booking_receipt(
        "Harbor House",
        "Market Street 17",
        "12345",
        "Monday 14 September 2026",
        "240",
        "1234567890",
    );
    let second = booking_receipt(
        "Garden Court Apartments",
        "Oak Avenue 8",
        "98765",
        "Tuesday 15 September 2026",
        "198.50",
        "9876543210",
    );
    let matches = run(&[&first, &second]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::Variant);
    assert!(
        matches[0]
            .evidence
            .iter()
            .any(|evidence| evidence.note == "same template, different data")
    );
}

#[test]
fn short_merchant_fields_alone_do_not_relabel_prose() {
    let original = booking_receipt(
        "Harbor House",
        "Market Street 17",
        "12345",
        "Monday 14 September 2026",
        "240",
        "1234567890",
    );
    let edited = original.replace("Harbor House", "Garden Court");
    let matches = run(&[&original, &edited]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::Edited);
}

#[test]
fn changed_amount_and_short_name_fields_need_no_identifier_change() {
    let original = booking_receipt(
        "Harbor House",
        "Market Street 17",
        "12345",
        "Monday 14 September 2026",
        "240",
        "1234567890",
    );
    let edited = original
        .replace("Harbor House", "Garden Court")
        .replace("240 EUR", "198.50 EUR");
    let matches = run(&[&original, &edited]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::Variant);
}

#[test]
fn prose_edits_stay_duplicates_even_beside_changed_booking_data() {
    let original = booking_receipt(
        "Harbor House",
        "Market Street 17",
        "12345",
        "Monday 14 September 2026",
        "240",
        "1234567890",
    );
    for (before, after) in [
        ("complete history", "full record"),
        (
            "<p>Payment was received promptly.</p>",
            "<p>Payment was recorded late.</p>",
        ),
    ] {
        let first = format!("{original}<p>Payment was received promptly.</p>");
        let edited = first
            .replace("1234567890", "9876543210")
            .replace(before, after);
        let matches = run(&[&first, &edited]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].relation, Relation::Edited);
    }
}

#[test]
fn changed_field_layout_can_split_one_short_address_slot() {
    let original = booking_receipt(
        "Harbor House",
        "Market Street 17",
        "12345",
        "Monday 14 September 2026",
        "240",
        "1234567890",
    );
    let edited = original
        .replace("1234567890", "9876543210")
        .replace("Market Street 17", "<div>Oak Avenue</div><div>8</div>");
    let matches = run(&[&original, &edited]);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, Relation::Variant);
}
