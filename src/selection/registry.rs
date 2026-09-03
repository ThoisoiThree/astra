use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::molecule::Molecule;

use super::{Selection, SelectionExpr, evaluate_with_named, parse_selection};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionStatus {
    Valid,
    Broken(Vec<String>),
    Cyclic,
    Stale(String),
}

impl SelectionStatus {
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Valid => "valid",
            Self::Broken(_) => "broken",
            Self::Cyclic => "cyclic",
            Self::Stale(_) => "stale",
        }
    }

    pub const fn is_valid(&self) -> bool {
        matches!(self, Self::Valid)
    }
}

#[derive(Debug, Clone)]
pub struct SelectionResolution {
    pub selections: BTreeMap<String, Selection>,
    pub statuses: BTreeMap<String, SelectionStatus>,
}

pub fn canonical_name(name: &str) -> String {
    name.trim().to_lowercase()
}

pub fn find_display_name<'a, T>(items: &'a BTreeMap<String, T>, name: &str) -> Option<&'a str> {
    let canonical = canonical_name(name);
    items
        .keys()
        .find(|candidate| canonical_name(candidate) == canonical)
        .map(String::as_str)
}

pub fn validate_unique_name<T>(
    items: &BTreeMap<String, T>,
    candidate: &str,
    except: Option<&str>,
) -> Result<(), String> {
    let candidate = candidate.trim();
    if candidate.is_empty() {
        return Err("selection name cannot be empty".into());
    }
    if !candidate
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
    {
        return Err("selection names may contain only letters, digits, '_' and '-'".into());
    }
    let key = canonical_name(candidate);
    if items.keys().any(|name| {
        canonical_name(name) == key
            && except.is_none_or(|except| !name.eq_ignore_ascii_case(except))
    }) {
        return Err(format!(
            "selection name '{candidate}' conflicts with an existing name (names are case-insensitive)"
        ));
    }
    Ok(())
}

pub fn resolve_named_expressions(
    molecule: &Molecule,
    expressions: &BTreeMap<String, String>,
    previous: &BTreeMap<String, Selection>,
) -> SelectionResolution {
    let mut names = BTreeMap::<String, String>::new();
    for name in previous.keys().chain(expressions.keys()) {
        names
            .entry(canonical_name(name))
            .or_insert_with(|| name.clone());
    }

    let mut parsed = BTreeMap::<String, SelectionExpr>::new();
    let mut statuses = BTreeMap::<String, SelectionStatus>::new();
    for (name, source) in expressions {
        match parse_selection(source) {
            Ok(expression) => {
                parsed.insert(name.clone(), expression);
            }
            Err(error) => {
                statuses.insert(name.clone(), SelectionStatus::Stale(error.to_string()));
            }
        }
    }

    let mut selections = previous
        .iter()
        .filter(|(name, _)| !expressions.contains_key(*name))
        .map(|(name, selection)| (name.clone(), selection.clone()))
        .collect::<BTreeMap<_, _>>();
    for name in selections.keys() {
        statuses.insert(name.clone(), SelectionStatus::Valid);
    }

    let mut states = HashMap::<String, VisitState>::new();
    let expression_names: Vec<_> = expressions.keys().cloned().collect();
    for name in expression_names {
        resolve_one(
            &name,
            molecule,
            &names,
            &parsed,
            previous,
            &mut selections,
            &mut statuses,
            &mut states,
            &mut Vec::new(),
        );
    }
    for name in expressions.keys() {
        selections
            .entry(name.clone())
            .or_insert_with(|| Selection::from_flags(vec![false; molecule.atoms.len()]));
    }

    SelectionResolution {
        selections,
        statuses,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisitState {
    Visiting,
    Complete,
}

#[allow(clippy::too_many_arguments)]
fn resolve_one(
    name: &str,
    molecule: &Molecule,
    names: &BTreeMap<String, String>,
    parsed: &BTreeMap<String, SelectionExpr>,
    previous: &BTreeMap<String, Selection>,
    selections: &mut BTreeMap<String, Selection>,
    statuses: &mut BTreeMap<String, SelectionStatus>,
    states: &mut HashMap<String, VisitState>,
    stack: &mut Vec<String>,
) {
    if statuses
        .get(name)
        .is_some_and(|status| matches!(status, SelectionStatus::Stale(_) | SelectionStatus::Cyclic))
    {
        if let Some(selection) = previous.get(name) {
            selections.insert(name.to_owned(), selection.clone());
        }
        return;
    }
    let key = canonical_name(name);
    if states.get(&key) == Some(&VisitState::Complete) {
        return;
    }
    if states.get(&key) == Some(&VisitState::Visiting) {
        if let Some(start) = stack.iter().position(|candidate| candidate == &key) {
            for cyclic_key in &stack[start..] {
                if let Some(display_name) = names.get(cyclic_key) {
                    statuses.insert(display_name.clone(), SelectionStatus::Cyclic);
                    selections.remove(display_name);
                }
            }
        }
        return;
    }
    let Some(expression) = parsed.get(name) else {
        return;
    };
    states.insert(key.clone(), VisitState::Visiting);
    stack.push(key.clone());

    let dependencies = named_dependencies(expression);
    let mut missing = BTreeSet::new();
    for dependency in dependencies {
        let dependency_key = canonical_name(&dependency);
        let Some(display_name) = names.get(&dependency_key) else {
            missing.insert(dependency);
            continue;
        };
        if parsed.contains_key(display_name) {
            resolve_one(
                display_name,
                molecule,
                names,
                parsed,
                previous,
                selections,
                statuses,
                states,
                stack,
            );
        }
        match statuses.get(display_name) {
            Some(SelectionStatus::Valid) => {}
            Some(SelectionStatus::Cyclic) => {
                statuses.insert(name.to_owned(), SelectionStatus::Cyclic);
            }
            Some(SelectionStatus::Broken(values)) => missing.extend(values.iter().cloned()),
            Some(SelectionStatus::Stale(_)) | None => {
                missing.insert(display_name.clone());
            }
        }
    }

    if !matches!(statuses.get(name), Some(SelectionStatus::Cyclic)) {
        if missing.is_empty() {
            match evaluate_with_named(expression, molecule, selections) {
                Ok(selection) => {
                    selections.insert(name.to_owned(), selection);
                    statuses.insert(name.to_owned(), SelectionStatus::Valid);
                }
                Err(error) => {
                    statuses.insert(name.to_owned(), SelectionStatus::Stale(error.to_string()));
                    if let Some(selection) = previous.get(name) {
                        selections.insert(name.to_owned(), selection.clone());
                    }
                }
            }
        } else {
            selections.remove(name);
            statuses.insert(
                name.to_owned(),
                SelectionStatus::Broken(missing.into_iter().collect()),
            );
        }
    }
    stack.pop();
    states.insert(key, VisitState::Complete);
}

pub fn named_dependencies(expression: &SelectionExpr) -> BTreeSet<String> {
    let mut dependencies = BTreeSet::new();
    collect_dependencies(expression, &mut dependencies);
    dependencies
}

fn collect_dependencies(expression: &SelectionExpr, output: &mut BTreeSet<String>) {
    match expression {
        SelectionExpr::Named(name) => {
            output.insert(name.clone());
        }
        SelectionExpr::Not(inner) => collect_dependencies(inner, output),
        SelectionExpr::And(left, right)
        | SelectionExpr::Xor(left, right)
        | SelectionExpr::Or(left, right) => {
            collect_dependencies(left, output);
            collect_dependencies(right, output);
        }
        _ => {}
    }
}

pub fn rename_named_reference(
    source: &str,
    old_name: &str,
    new_name: &str,
) -> Result<String, super::SelectionParseError> {
    let mut expression = parse_selection(source)?;
    rename_in_ast(&mut expression, old_name, new_name);
    Ok(format_expression(&expression))
}

fn rename_in_ast(expression: &mut SelectionExpr, old_name: &str, new_name: &str) {
    match expression {
        SelectionExpr::Named(name) if name.eq_ignore_ascii_case(old_name) => {
            *name = new_name.to_owned();
        }
        SelectionExpr::Not(inner) => rename_in_ast(inner, old_name, new_name),
        SelectionExpr::And(left, right)
        | SelectionExpr::Xor(left, right)
        | SelectionExpr::Or(left, right) => {
            rename_in_ast(left, old_name, new_name);
            rename_in_ast(right, old_name, new_name);
        }
        _ => {}
    }
}

pub fn format_expression(expression: &SelectionExpr) -> String {
    format_with_precedence(expression, 0)
}

fn format_with_precedence(expression: &SelectionExpr, parent: u8) -> String {
    let (precedence, text) = match expression {
        SelectionExpr::All => (5, "all".into()),
        SelectionExpr::None => (5, "none".into()),
        SelectionExpr::Element(value) => (5, format!("element {}", value.symbol())),
        SelectionExpr::AtomName(value) => (5, format!("name {value}")),
        SelectionExpr::AtomNamePattern(value) => (5, format!("name {value}")),
        SelectionExpr::ResidueName(value) => (5, format!("resn {value}")),
        SelectionExpr::ResidueNamePattern(value) => (5, format!("resn {value}")),
        SelectionExpr::ResidueNumber(value) => (5, format!("resi {value}")),
        SelectionExpr::ResidueRange(start, end) => (5, format!("resi {start}-{end}")),
        SelectionExpr::Chain(value) => (5, format!("chain {value}")),
        SelectionExpr::ChainPattern(value) => (5, format!("chain {value}")),
        SelectionExpr::Serial(value) => (5, format!("serial {value}")),
        SelectionExpr::Named(value) => (5, format!("selection {value}")),
        SelectionExpr::Hetatm => (5, "hetatm".into()),
        SelectionExpr::Polymer => (5, "polymer".into()),
        SelectionExpr::Not(inner) => (4, format!("not {}", format_with_precedence(inner, 4))),
        SelectionExpr::And(left, right) => (
            3,
            format!(
                "{} and {}",
                format_with_precedence(left, 3),
                format_with_precedence(right, 3)
            ),
        ),
        SelectionExpr::Xor(left, right) => (
            2,
            format!(
                "{} xor {}",
                format_with_precedence(left, 2),
                format_with_precedence(right, 2)
            ),
        ),
        SelectionExpr::Or(left, right) => (
            1,
            format!(
                "{} or {}",
                format_with_precedence(left, 1),
                format_with_precedence(right, 1)
            ),
        ),
    };
    if precedence < parent {
        format!("({text})")
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::molecule::{Atom, Element, Molecule};

    fn molecule() -> Molecule {
        Molecule {
            atoms: vec![Atom {
                serial: 1,
                name: "CA".into(),
                element: Element::C,
                residue_name: "ALA".into(),
                residue_number: 1,
                insertion_code: None,
                chain_id: "A".into(),
                position: Vec3::ZERO,
                occupancy: 1.0,
                b_factor: 0.0,
                hetero: false,
            }],
            bonds: Vec::new(),
        }
    }

    #[test]
    fn resolves_dependencies_and_marks_missing_and_cycles() {
        let expressions = BTreeMap::from([
            ("A".into(), "chain A".into()),
            ("B".into(), "selection a and element C".into()),
            ("Broken".into(), "selection missing".into()),
            ("Cycle1".into(), "selection Cycle2".into()),
            ("Cycle2".into(), "selection Cycle1".into()),
        ]);
        let result = resolve_named_expressions(&molecule(), &expressions, &BTreeMap::new());
        assert_eq!(result.selections["B"].count(), 1);
        assert!(matches!(
            result.statuses["Broken"],
            SelectionStatus::Broken(_)
        ));
        assert_eq!(result.statuses["Cycle1"], SelectionStatus::Cyclic);
        assert_eq!(result.statuses["Cycle2"], SelectionStatus::Cyclic);
    }

    #[test]
    fn rename_uses_ast_and_preserves_unrelated_names() {
        let source = rename_named_reference(
            "selection old and (selection older or selection OLD)",
            "old",
            "new",
        )
        .unwrap();
        assert_eq!(
            source,
            "selection new and (selection older or selection new)"
        );
    }

    #[test]
    fn names_are_case_insensitively_unique() {
        let existing = BTreeMap::from([("Alpha".into(), ())]);
        assert!(validate_unique_name(&existing, "alpha", None).is_err());
    }
}
