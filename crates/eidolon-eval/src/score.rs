//! Pair scoring with explicit direct and transitive-group measurements.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context as _, bail, ensure};
use serde::Serialize;

use crate::model::{ExpectedPair, Output, OutputMatch, Truth};

type Pair = (usize, usize);

/// Per-category counts and rates. Empty precision denominators yield 1; empty recall yields 0.
#[derive(Debug, Default, Serialize)]
pub struct Metrics {
    /// Annotated positive pairs.
    pub expected: usize,
    /// Annotated duplicate pairs eligible for group recall; zero for variants.
    pub group_expected: usize,
    /// Positives found by an explicit match.
    pub direct_found: usize,
    /// Duplicate positives connected by non-variant matches or declared groups.
    pub group_found: usize,
    /// Incorrect direct cross-family or explicit-negative pairs.
    pub false_positives: usize,
    /// Incorrect transitive pairs.
    pub group_false_positives: usize,
    /// Direct pair precision, excluding unannotated siblings.
    pub direct_precision: f64,
    /// Direct pair recall.
    pub direct_recall: f64,
    /// Transitive pair precision, excluding unannotated siblings.
    pub group_precision: f64,
    /// Transitive pair recall.
    pub group_recall: f64,
    /// Correct labels among annotated pairs receiving any direct report; null when absent.
    pub relation_accuracy: Option<f64>,
    /// Correct relation and part/whole order among found directional positives.
    pub direction_accuracy: Option<f64>,
    #[serde(skip)]
    labels_correct: usize,
    #[serde(skip)]
    labeled_found: usize,
    #[serde(skip)]
    directions_correct: usize,
    #[serde(skip)]
    directional_found: usize,
}

impl Metrics {
    fn finish(&mut self) {
        self.direct_precision = precision(self.direct_found, self.false_positives);
        self.group_precision = precision(self.group_found, self.group_false_positives);
        self.direct_recall = ratio(self.direct_found, self.expected);
        self.group_recall = ratio(self.group_found, self.group_expected);
        self.relation_accuracy =
            (self.labeled_found > 0).then(|| ratio(self.labels_correct, self.labeled_found));
        self.direction_accuracy = (self.directional_found > 0)
            .then(|| ratio(self.directions_correct, self.directional_found));
    }
}

/// An actionable pair-level diagnostic, ordered by confidence for false positives.
#[derive(Debug, Serialize)]
pub struct Diagnostic {
    /// First relative corpus path.
    pub a: PathBuf,
    /// Second relative corpus path.
    pub b: PathBuf,
    /// Expected or observed relation.
    pub relation: String,
    /// Prediction confidence, absent for misses and group-only errors.
    pub score: Option<f64>,
}

/// Pair recall split by input transport; missing inputs remain denominators.
#[derive(Debug, Default, Serialize)]
pub struct RecallMetrics {
    /// Annotated pairs in this stratum.
    pub expected: usize,
    /// Pairs receiving a prediction in the correct duplicate/Variant channel.
    pub direct_found: usize,
    /// Pairs connected by duplicate predictions or groups.
    pub group_found: usize,
    /// Duplicate pairs eligible for group recall; Variant pairs are excluded.
    pub group_expected: usize,
    /// Directly found pairs divided by all expected pairs.
    pub direct_recall: f64,
    /// Connected duplicate pairs divided by group-eligible expected pairs.
    pub group_recall: f64,
}

impl RecallMetrics {
    fn count(&mut self, direct: bool, group: Option<bool>) {
        self.expected += 1;
        self.direct_found += usize::from(direct);
        if let Some(group) = group {
            self.group_expected += 1;
            self.group_found += usize::from(group);
        }
    }

    fn finish(&mut self) {
        self.direct_recall = ratio(self.direct_found, self.expected);
        self.group_recall = ratio(self.group_found, self.group_expected);
    }
}

/// Machine-readable evaluation with at most 20 diagnostics per list.
#[derive(Debug, Serialize)]
pub struct Report {
    /// Evaluation schema version.
    pub version: u32,
    /// Whole-corpus duplicate metrics, excluding related non-duplicate versions.
    pub overall: Metrics,
    /// Related-version metrics, separate from duplicate detection and grouping.
    pub variants: Metrics,
    /// Client-measured body bytes for reported URLs; absent when complete sizes are unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url_traffic: Option<crate::serve::Traffic>,
    /// Independently counted loopback response-body bytes (may exceed client-consumed bytes).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub served_traffic: Option<crate::serve::Traffic>,
    /// Explicit URL runner policy; absent when scoring an independent CLI report.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_video_mode: Option<String>,
    /// Effective detector options for `run`, excluding corpus inputs; absent for standalone scoring.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eidolon_args: Option<Vec<String>>,
    /// Category metrics keyed by `kind/variant`.
    pub categories: BTreeMap<String, Metrics>,
    /// `FrameOf` target transport and transport/category recall, without invented precision.
    pub frame_of: BTreeMap<String, RecallMetrics>,
    /// Recall for local, remote, or missing-input pairs, split by kind/transformation.
    pub input_transport: BTreeMap<String, RecallMetrics>,
    /// Positives missed even after group connectivity.
    pub misses: Vec<Diagnostic>,
    /// Positives without a direct match, including group-only finds.
    pub direct_misses: Vec<Diagnostic>,
    /// Incorrect direct pairs, highest confidence first.
    pub false_positives: Vec<Diagnostic>,
    /// Incorrect connected pairs, including errors introduced by chaining.
    pub group_false_positives: Vec<Diagnostic>,
    /// Found containment pairs with wrong relation or endpoint ordering.
    pub wrong_directions: Vec<Diagnostic>,
    /// Directly detected annotated hard negatives.
    pub hard_negative_false_positives: usize,
    /// Hard negatives connected transitively.
    pub hard_negative_group_false_positives: usize,
    /// Direct sibling pairs not annotated, excluded from precision.
    pub ignored_sibling_predictions: usize,
}

/// Evaluate version-one corpus truth against version-one CLI JSON.
///
/// Missing output files are misses, never silently removed from the denominator.
/// Unknown files, duplicate IDs/paths, invalid relations, and dangling IDs are errors.
/// Annotated duplicate/design-version pairs and unrelated cross-family negatives are scored;
/// unannotated siblings may have disjoint crops and are therefore not assumed related.
///
/// # Errors
/// Returns an error for unsupported schemas or inconsistent identities/annotations.
#[allow(clippy::too_many_lines)] // Keep the complete scoring pass and its denominators together.
pub fn evaluate(truth: &Truth, output: &Output) -> anyhow::Result<Report> {
    ensure!(
        truth.version == 1 && output.version == 1,
        "unsupported schema version"
    );
    let mut paths = HashMap::new();
    for (index, file) in truth.files.iter().enumerate() {
        ensure!(
            paths
                .insert(normalize(&truth.root.join(&file.path)), index)
                .is_none(),
            "duplicate truth path: {}",
            file.path.display()
        );
    }
    let lookup = |path: &Path| -> anyhow::Result<usize> {
        paths
            .get(&normalize(&truth.root.join(path)))
            .copied()
            .with_context(|| format!("path absent from truth: {}", path.display()))
    };
    let mut expected = BTreeMap::new();
    for pair in &truth.pairs {
        validate_relation(&pair.relation)?;
        for relation in &pair.accepted_relations {
            validate_relation(relation)?;
            ensure!(
                (relation == "variant") == (pair.relation == "variant"),
                "accepted label crosses duplicate/variant channels"
            );
        }
        let a = lookup(&pair.a)?;
        let b = lookup(&pair.b)?;
        ensure!(a != b, "self pair in truth");
        ensure!(
            expected.insert(ordered(a, b), (pair, a)).is_none(),
            "duplicate truth pair"
        );
    }
    let mut negatives: HashMap<Pair, bool> = HashMap::new();
    for pair in &truth.hard_negatives {
        let key = ordered(lookup(&pair.a)?, lookup(&pair.b)?);
        ensure!(
            key.0 != key.1 && !expected.contains_key(&key),
            "conflicting hard negative"
        );
        negatives
            .entry(key)
            .and_modify(|direct_only| *direct_only &= pair.direct_only)
            .or_insert(pair.direct_only);
    }
    let mut ids = HashMap::new();
    let mut seen_paths = HashSet::new();
    let mut url_traffic = crate::serve::Traffic::default();
    let mut remote_indices = HashSet::new();
    let mut known_remote_sizes = true;
    for file in &output.files {
        let remote_path = file
            .url
            .as_ref()
            .map(|origin| crate::serve::media_path(&origin.url))
            .transpose()?;
        let index = lookup(remote_path.as_deref().unwrap_or(&file.path))?;
        if let Some(origin) = &file.url {
            remote_indices.insert(index);
            known_remote_sizes &= origin.total_size.is_some();
            url_traffic.files.insert(
                truth.files[index].path.to_string_lossy().into_owned(),
                crate::serve::FileTraffic {
                    size: origin.total_size.unwrap_or(0),
                    bytes_served: origin.bytes_fetched,
                    requests: None,
                },
            );
        }
        ensure!(
            ids.insert(file.id, index).is_none(),
            "duplicate output ID {}",
            file.id
        );
        ensure!(
            seen_paths.insert(index),
            "duplicate output path: {}",
            file.path.display()
        );
    }
    let id = |number: usize| {
        ids.get(&number)
            .copied()
            .with_context(|| format!("unknown output file ID {number}"))
    };
    let mut direct: BTreeMap<Pair, (&OutputMatch, usize)> = BTreeMap::new();
    let mut duplicates: BTreeMap<Pair, (&OutputMatch, usize)> = BTreeMap::new();
    let mut versions: BTreeMap<Pair, (&OutputMatch, usize)> = BTreeMap::new();
    let mut components = Components::new(truth.files.len());
    for item in &output.matches {
        validate_relation(&item.relation)?;
        ensure!(
            item.score.is_finite() && (0.0..=1.0).contains(&item.score),
            "invalid match score"
        );
        let a = id(item.a)?;
        let b = id(item.b)?;
        ensure!(a != b, "self match in output");
        let slot = direct.entry(ordered(a, b)).or_insert((item, a));
        if item.score > slot.0.score {
            *slot = (item, a);
        }
        let channel = if item.relation == "variant" {
            &mut versions
        } else {
            &mut duplicates
        };
        let slot = channel.entry(ordered(a, b)).or_insert((item, a));
        if item.score > slot.0.score {
            *slot = (item, a);
        }
        if item.relation != "variant" {
            components.join(a, b);
        }
    }
    for group in &output.groups {
        if let Some(first) = group.files.first() {
            let first = id(*first)?;
            for member in &group.files {
                components.join(first, id(*member)?);
            }
        }
    }
    let category_names: Vec<_> = truth
        .files
        .iter()
        .map(|file| format!("{}/{}", file.kind, file.variant))
        .collect();
    let variant_categories: Vec<_> = truth
        .files
        .iter()
        .map(|file| format!("{}/variant", file.kind))
        .collect();
    let mut report = Report {
        version: 1,
        overall: Metrics::default(),
        variants: Metrics::default(),
        url_traffic: if known_remote_sizes && !url_traffic.files.is_empty() {
            url_traffic.finish(true);
            Some(url_traffic)
        } else {
            None
        },
        categories: category_names
            .iter()
            .chain(&variant_categories)
            .map(|key| (key.clone(), Metrics::default()))
            .collect(),
        served_traffic: None,
        remote_video_mode: None,
        eidolon_args: None,
        input_transport: BTreeMap::new(),
        frame_of: BTreeMap::new(),
        misses: Vec::new(),
        direct_misses: Vec::new(),
        false_positives: Vec::new(),
        group_false_positives: Vec::new(),
        wrong_directions: Vec::new(),
        hard_negative_false_positives: 0,
        hard_negative_group_false_positives: 0,
        ignored_sibling_predictions: 0,
    };
    for (&key, &(pair, expected_a)) in &expected {
        let is_variant = pair.relation == "variant";
        let found = if is_variant {
            versions.get(&key)
        } else {
            duplicates.get(&key)
        };
        let labeled = direct.get(&key);
        let connected = if is_variant {
            found.is_some()
        } else {
            components.same(key.0, key.1)
        };
        if pair.relation == "frame_of" {
            let video = lookup(&pair.b)?;
            let transport = if !seen_paths.contains(&video) {
                "missing_video"
            } else if remote_indices.contains(&video) {
                "remote_video"
            } else {
                "local_video"
            };
            for name in [
                transport.to_owned(),
                format!("{transport}/{}", pair.variant),
            ] {
                let metric = report.frame_of.entry(name).or_default();
                metric.count(found.is_some(), Some(connected));
            }
        }
        let transport = if !seen_paths.contains(&key.0) || !seen_paths.contains(&key.1) {
            "missing"
        } else if remote_indices.contains(&key.0) || remote_indices.contains(&key.1) {
            "remote"
        } else {
            "local"
        };
        report
            .input_transport
            .entry(format!("{transport}/{}/{}", pair.kind, pair.variant))
            .or_default()
            .count(found.is_some(), (!is_variant).then_some(connected));
        let category = report
            .categories
            .entry(format!(
                "{}/{}",
                pair.kind,
                if is_variant { "variant" } else { &pair.variant }
            ))
            .or_default();
        let overall = if is_variant {
            &mut report.variants
        } else {
            &mut report.overall
        };
        for metric in [overall, category] {
            metric.expected += 1;
            metric.group_expected += usize::from(!is_variant);
            metric.group_found += usize::from(!is_variant && connected);
            metric.direct_found += usize::from(found.is_some());
            if let Some(&(item, actual_a)) = labeled {
                metric.labeled_found += 1;
                metric.labels_correct += usize::from(
                    item.relation == pair.relation
                        || pair.accepted_relations.contains(&item.relation),
                );
                if directional(&pair.relation) {
                    metric.directional_found += 1;
                    metric.directions_correct +=
                        usize::from(actual_a == expected_a && item.relation == pair.relation);
                }
            }
        }
        if found.is_none() {
            push_limited(&mut report.direct_misses, || expected_diagnostic(pair));
        }
        if !connected {
            push_limited(&mut report.misses, || expected_diagnostic(pair));
        }
        if let Some(&(item, actual_a)) = labeled
            && directional(&pair.relation)
            && (actual_a != expected_a || item.relation != pair.relation)
        {
            push_limited(&mut report.wrong_directions, || expected_diagnostic(pair));
        }
    }
    let is_negative = |key: Pair, variant: bool| {
        expected.get(&key).map_or_else(
            || {
                let a = &truth.files[key.0];
                let b = &truth.files[key.1];
                negatives.contains_key(&key)
                    || (a.family != b.family && !(a.design.is_some() && a.design == b.design))
            },
            |(pair, _)| (pair.relation == "variant") != variant,
        )
    };
    let mut false_direct = Vec::new();
    let mut hard_negative_predictions = HashSet::new();
    for (variant, predictions) in [(false, &duplicates), (true, &versions)] {
        for (&key, &(item, _)) in predictions {
            if is_negative(key, variant) {
                let metric = if variant {
                    &mut report.variants
                } else {
                    &mut report.overall
                };
                metric.false_positives += 1;
                if negatives.contains_key(&key) {
                    hard_negative_predictions.insert(key);
                }
                count_negative_categories(
                    &mut report.categories,
                    if variant {
                        &variant_categories
                    } else {
                        &category_names
                    },
                    key,
                    false,
                );
                false_direct.push((key, item));
            }
        }
    }
    report.hard_negative_false_positives = hard_negative_predictions.len();
    report.ignored_sibling_predictions = direct
        .keys()
        .filter(|key| !expected.contains_key(key) && !is_negative(**key, false))
        .count();
    false_direct.sort_by(|a, b| b.1.score.total_cmp(&a.1.score));
    report.false_positives = false_direct
        .into_iter()
        .take(20)
        .map(|(key, item)| Diagnostic {
            a: truth.files[key.0].path.clone(),
            b: truth.files[key.1].path.clone(),
            relation: item.relation.clone(),
            score: Some(item.score),
        })
        .collect();
    for a in 0..truth.files.len() {
        for b in a + 1..truth.files.len() {
            if is_negative((a, b), false)
                && components.same(a, b)
                && negatives.get(&(a, b)) != Some(&true)
            {
                report.overall.group_false_positives += 1;
                report.hard_negative_group_false_positives +=
                    usize::from(negatives.contains_key(&(a, b)));
                count_negative_categories(&mut report.categories, &category_names, (a, b), true);
                push_limited(&mut report.group_false_positives, || Diagnostic {
                    a: truth.files[a].path.clone(),
                    b: truth.files[b].path.clone(),
                    relation: "connected".into(),
                    score: None,
                });
            }
        }
    }
    report.overall.finish();
    report.variants.finish();
    for metric in report
        .frame_of
        .values_mut()
        .chain(report.input_transport.values_mut())
    {
        metric.finish();
    }
    for metrics in report.categories.values_mut() {
        metrics.finish();
    }
    Ok(report)
}

fn count_negative_categories(
    categories: &mut BTreeMap<String, Metrics>,
    names: &[String],
    pair: Pair,
    group: bool,
) {
    for index in [pair.0, pair.1] {
        if index == pair.1 && names[pair.0] == names[pair.1] {
            continue;
        }
        if let Some(metric) = categories.get_mut(&names[index]) {
            if group {
                metric.group_false_positives += 1;
            } else {
                metric.false_positives += 1;
            }
        }
    }
}

fn expected_diagnostic(pair: &ExpectedPair) -> Diagnostic {
    Diagnostic {
        a: pair.a.clone(),
        b: pair.b.clone(),
        relation: pair.relation.clone(),
        score: None,
    }
}

fn push_limited(list: &mut Vec<Diagnostic>, item: impl FnOnce() -> Diagnostic) {
    if list.len() < 20 {
        list.push(item());
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn precision(found: usize, false_positives: usize) -> f64 {
    if found + false_positives == 0 {
        1.0
    } else {
        ratio(found, found + false_positives)
    }
}

fn ordered(a: usize, b: usize) -> Pair {
    if a < b { (a, b) } else { (b, a) }
}

fn directional(relation: &str) -> bool {
    matches!(relation, "crop_of" | "clip_of" | "excerpt_of" | "frame_of")
}

fn validate_relation(relation: &str) -> anyhow::Result<()> {
    if !matches!(
        relation,
        "identical"
            | "same_content"
            | "reencoded"
            | "edited"
            | "crop_of"
            | "clip_of"
            | "excerpt_of"
            | "frame_of"
            | "similar"
            | "variant"
    ) {
        bail!("unknown relation {relation}");
    }
    Ok(())
}

fn normalize(path: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            _ => result.push(component),
        }
    }
    result
}

struct Components {
    parent: Vec<usize>,
}
impl Components {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
        }
    }
    fn root(&mut self, mut node: usize) -> usize {
        while self.parent[node] != node {
            self.parent[node] = self.parent[self.parent[node]];
            node = self.parent[node];
        }
        node
    }
    fn join(&mut self, a: usize, b: usize) {
        let a = self.root(a);
        let b = self.root(b);
        self.parent[a] = b;
    }
    fn same(&mut self, a: usize, b: usize) -> bool {
        self.root(a) == self.root(b)
    }
}
