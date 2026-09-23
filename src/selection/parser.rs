use std::{fmt, str::FromStr};

use crate::molecule::Element;

use super::{
    ast::{MAX_SELECTION_DISTANCE, SelectionExpr, distance_to_milli},
    lexer::{Token, TokenKind, lex},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionParseError {
    pub position: usize,
    pub expected: String,
    pub found: String,
}

impl SelectionParseError {
    pub(crate) fn new(
        position: usize,
        expected: impl Into<String>,
        found: impl Into<String>,
    ) -> Self {
        Self {
            position,
            expected: expected.into(),
            found: found.into(),
        }
    }
}

impl fmt::Display for SelectionParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "selection syntax error at character {}: expected {}, found {}",
            self.position, self.expected, self.found
        )
    }
}

impl std::error::Error for SelectionParseError {}

pub fn parse_selection(input: &str) -> Result<SelectionExpr, SelectionParseError> {
    if input.contains('/') || input.contains('[') {
        return parse_mask_expression(input, 0);
    }
    parse_standard_selection(input)
}

fn parse_standard_selection(input: &str) -> Result<SelectionExpr, SelectionParseError> {
    let tokens = lex(input)?;
    let mut parser = Parser {
        tokens,
        cursor: 0,
        input_len: input.len(),
    };
    let expression = parser.parse_or()?;
    if let Some(token) = parser.peek() {
        return Err(SelectionParseError::new(
            token.position,
            "'and', 'xor', 'or', or end of input",
            describe(token),
        ));
    }
    Ok(expression)
}

fn parse_mask_expression(
    input: &str,
    position: usize,
) -> Result<SelectionExpr, SelectionParseError> {
    let leading = input.len() - input.trim_start().len();
    let input = input.trim();
    let position = position + leading;
    if input.is_empty() {
        return Err(SelectionParseError::new(
            position,
            "a selection predicate",
            "end of input",
        ));
    }

    if fully_parenthesized(input) {
        return parse_mask_expression(&input[1..input.len() - 1], position + 1);
    }

    for (keyword, constructor) in [
        (
            "or",
            SelectionExpr::Or as fn(Box<SelectionExpr>, Box<SelectionExpr>) -> SelectionExpr,
        ),
        ("xor", SelectionExpr::Xor),
        ("and", SelectionExpr::And),
    ] {
        if let Some(index) = find_last_top_level_keyword(input, keyword) {
            let right_position = position + index + keyword.len();
            let left = parse_mask_expression(&input[..index], position)?;
            let right = parse_mask_expression(&input[index + keyword.len()..], right_position)?;
            return Ok(constructor(Box::new(left), Box::new(right)));
        }
    }

    if let Some((operator, rest)) = strip_prefix_operator(input, position)? {
        let rest_position = position + input.len() - rest.len();
        return Ok(operator.apply(parse_mask_expression(rest, rest_position)?));
    }

    if input.starts_with('[') {
        return parse_residue_list(input, position);
    }

    if input.contains('/') {
        let mask = strip_keyword_prefix(input, "chain").unwrap_or(input);
        let mask_position = position + input.len() - mask.len();
        return parse_chain_mask_body(mask, mask_position);
    }

    parse_standard_selection(input).map_err(|mut error| {
        error.position += position;
        error
    })
}

/// Unary prefix operators shared by the token parser and the path-mask parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefixOperator {
    Not,
    Within(u32),
    Around(u32),
    ByResidue,
    ByChain,
}

impl PrefixOperator {
    fn apply(self, operand: SelectionExpr) -> SelectionExpr {
        let operand = Box::new(operand);
        match self {
            Self::Not => SelectionExpr::Not(operand),
            Self::Within(distance) => SelectionExpr::Within(distance, operand),
            Self::Around(distance) => SelectionExpr::Around(distance, operand),
            Self::ByResidue => SelectionExpr::ByResidue(operand),
            Self::ByChain => SelectionExpr::ByChain(operand),
        }
    }
}

fn parse_distance(value: &str, position: usize) -> Result<u32, SelectionParseError> {
    match value.parse::<f32>() {
        Ok(distance)
            if distance.is_finite() && (0.0..=MAX_SELECTION_DISTANCE).contains(&distance) =>
        {
            Ok(distance_to_milli(distance))
        }
        _ => Err(SelectionParseError::new(
            position,
            format!("a distance in Å between 0 and {MAX_SELECTION_DISTANCE}"),
            format!("'{value}'"),
        )),
    }
}

/// Splits a leading prefix operator off a path-mask expression.
fn strip_prefix_operator(
    input: &str,
    position: usize,
) -> Result<Option<(PrefixOperator, &str)>, SelectionParseError> {
    if let Some(rest) = strip_keyword_prefix(input, "not") {
        return Ok(Some((PrefixOperator::Not, rest)));
    }
    for (keyword, operator) in [
        ("byres", PrefixOperator::ByResidue),
        ("bychain", PrefixOperator::ByChain),
    ] {
        if let Some(rest) = strip_keyword_prefix(input, keyword) {
            return Ok(Some((operator, rest)));
        }
    }
    for (phrase, operator) in [
        (["same", "residue", "as"], PrefixOperator::ByResidue),
        (["same", "chain", "as"], PrefixOperator::ByChain),
    ] {
        let mut rest = input;
        let mut matched = true;
        for word in phrase {
            match strip_keyword_prefix(rest, word) {
                Some(next) => rest = next,
                None => {
                    matched = false;
                    break;
                }
            }
        }
        if matched {
            return Ok(Some((operator, rest)));
        }
    }
    for keyword in ["within", "around"] {
        let Some(rest) = strip_keyword_prefix(input, keyword) else {
            continue;
        };
        let distance_position = position + input.len() - rest.len();
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        let distance = parse_distance(&rest[..end], distance_position)?;
        let after = rest[end..].trim_start();
        let Some(operand) = strip_keyword_prefix(after, "of") else {
            return Err(SelectionParseError::new(
                position + input.len() - after.len(),
                "'of'",
                after.split_whitespace().next().unwrap_or("end of input"),
            ));
        };
        let operator = if keyword == "within" {
            PrefixOperator::Within(distance)
        } else {
            PrefixOperator::Around(distance)
        };
        return Ok(Some((operator, operand)));
    }
    Ok(None)
}

fn strip_keyword_prefix<'a>(input: &'a str, keyword: &str) -> Option<&'a str> {
    let prefix = input.get(..keyword.len())?;
    if !prefix.eq_ignore_ascii_case(keyword) {
        return None;
    }
    let rest = &input[keyword.len()..];
    rest.chars()
        .next()
        .is_some_and(|character| character.is_whitespace() || character == '(')
        .then(|| rest.trim_start())
}

fn find_last_top_level_keyword(input: &str, keyword: &str) -> Option<usize> {
    let mut bracket_depth = 0_u32;
    let mut parenthesis_depth = 0_u32;
    let mut found = None;
    for (index, character) in input.char_indices() {
        match character {
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            '(' if bracket_depth == 0 => parenthesis_depth += 1,
            ')' if bracket_depth == 0 => parenthesis_depth = parenthesis_depth.saturating_sub(1),
            _ if bracket_depth == 0 && parenthesis_depth == 0 => {
                let Some(candidate) = input.get(index..index + keyword.len()) else {
                    continue;
                };
                let before = input[..index].chars().next_back();
                let after = input[index + keyword.len()..].chars().next();
                if candidate.eq_ignore_ascii_case(keyword)
                    && before.is_some_and(|value| value.is_whitespace() || value == ')')
                    && after.is_some_and(|value| value.is_whitespace() || value == '(')
                {
                    found = Some(index);
                }
            }
            _ => {}
        }
    }
    found
}

fn fully_parenthesized(input: &str) -> bool {
    if !input.starts_with('(') || !input.ends_with(')') {
        return false;
    }
    let mut depth = 0_u32;
    for (index, character) in input.char_indices() {
        match character {
            '(' => depth += 1,
            ')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 && index + character.len_utf8() != input.len() {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

fn parse_chain_mask_body(
    mask: &str,
    position: usize,
) -> Result<SelectionExpr, SelectionParseError> {
    let segments: Vec<_> = mask.split('/').map(str::trim).collect();
    if !(2..=3).contains(&segments.len())
        || segments.iter().any(|segment| {
            segment.is_empty()
                || (!segment.starts_with('[') && segment.contains(char::is_whitespace))
        })
    {
        return Err(SelectionParseError::new(
            position,
            "Chain <chain>/<residue-mask>[/<atom-mask>]",
            mask,
        ));
    }
    let mut expression = if matches!(segments[0], ".." | "*") {
        SelectionExpr::All
    } else {
        string_mask_expression(segments[0], MaskField::Chain)
    };
    let residue = if segments[1].starts_with('[') {
        parse_residue_list(segments[1], position + mask.find('/').unwrap_or(0) + 1)?
    } else {
        string_mask_expression(segments[1], MaskField::Residue)
    };
    expression = SelectionExpr::And(Box::new(expression), Box::new(residue));
    if let Some(atom) = segments.get(2) {
        expression = SelectionExpr::And(
            Box::new(expression),
            Box::new(string_mask_expression(atom, MaskField::Atom)),
        );
    }
    Ok(expression)
}

#[derive(Clone, Copy)]
enum MaskField {
    Chain,
    Residue,
    Atom,
}

fn string_mask_expression(value: &str, field: MaskField) -> SelectionExpr {
    let wildcard = value.contains('*') || value.contains('?');
    let value = value.to_ascii_uppercase();
    match (field, wildcard) {
        (MaskField::Chain, false) => SelectionExpr::Chain(value),
        (MaskField::Chain, true) => SelectionExpr::ChainPattern(value),
        (MaskField::Residue, false) => SelectionExpr::ResidueName(value),
        (MaskField::Residue, true) => SelectionExpr::ResidueNamePattern(value),
        (MaskField::Atom, false) => SelectionExpr::AtomName(value),
        (MaskField::Atom, true) => SelectionExpr::AtomNamePattern(value),
    }
}

fn parse_residue_list(value: &str, position: usize) -> Result<SelectionExpr, SelectionParseError> {
    let Some(inner) = value
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
    else {
        return Err(SelectionParseError::new(
            position,
            "a residue list such as [20:22, 70:71]",
            value,
        ));
    };
    let mut expressions = Vec::new();
    for item in inner.split(',').map(str::trim) {
        if item.is_empty() {
            return Err(SelectionParseError::new(
                position,
                "a residue number or range",
                value,
            ));
        }
        let expression = if let Some((start, end)) = item.split_once(':') {
            let start = parse_mask_number(start, position)?;
            let end = parse_mask_number(end, position)?;
            if start > end {
                return Err(SelectionParseError::new(
                    position,
                    "an ascending residue range",
                    item,
                ));
            }
            SelectionExpr::ResidueRange(start, end)
        } else {
            SelectionExpr::ResidueNumber(parse_mask_number(item, position)?)
        };
        expressions.push(expression);
    }
    expressions
        .into_iter()
        .reduce(|left, right| SelectionExpr::Or(Box::new(left), Box::new(right)))
        .ok_or_else(|| SelectionParseError::new(position, "a non-empty residue list", value))
}

fn parse_mask_number(value: &str, position: usize) -> Result<i32, SelectionParseError> {
    value
        .trim()
        .parse::<i32>()
        .map_err(|_| SelectionParseError::new(position, "an integer residue number", value.trim()))
}

struct Parser {
    tokens: Vec<Token>,
    cursor: usize,
    input_len: usize,
}

impl Parser {
    fn parse_or(&mut self) -> Result<SelectionExpr, SelectionParseError> {
        let mut expression = self.parse_xor()?;
        while self.consume_keyword("or") {
            let right = self.parse_xor()?;
            expression = SelectionExpr::Or(Box::new(expression), Box::new(right));
        }
        Ok(expression)
    }

    fn parse_xor(&mut self) -> Result<SelectionExpr, SelectionParseError> {
        let mut expression = self.parse_and()?;
        while self.consume_keyword("xor") {
            let right = self.parse_and()?;
            expression = SelectionExpr::Xor(Box::new(expression), Box::new(right));
        }
        Ok(expression)
    }

    fn parse_and(&mut self) -> Result<SelectionExpr, SelectionParseError> {
        let mut expression = self.parse_not()?;
        while self.consume_keyword("and") {
            let right = self.parse_not()?;
            expression = SelectionExpr::And(Box::new(expression), Box::new(right));
        }
        Ok(expression)
    }

    /// Unary level: `not`, `within D of`, `around D of`, `byres`, `bychain` and
    /// `same residue|chain as` all take one unary operand.
    fn parse_not(&mut self) -> Result<SelectionExpr, SelectionParseError> {
        let operator = if self.consume_keyword("not") {
            PrefixOperator::Not
        } else if self.consume_keyword("byres") {
            PrefixOperator::ByResidue
        } else if self.consume_keyword("bychain") {
            PrefixOperator::ByChain
        } else if self.peek_keyword(0, "same")
            && (self.peek_keyword(1, "residue") || self.peek_keyword(1, "chain"))
        {
            self.cursor += 1;
            let chain = self.consume_keyword("chain");
            if !chain {
                self.cursor += 1;
            }
            if !self.consume_keyword("as") {
                return Err(self.error_here("'as'"));
            }
            if chain {
                PrefixOperator::ByChain
            } else {
                PrefixOperator::ByResidue
            }
        } else if self.peek_keyword(0, "within") || self.peek_keyword(0, "around") {
            let around = self.peek_keyword(0, "around");
            self.cursor += 1;
            let (value, position) = self.value("a distance in Å")?;
            let distance = parse_distance(&value, position)?;
            if !self.consume_keyword("of") {
                return Err(self.error_here("'of'"));
            }
            if around {
                PrefixOperator::Around(distance)
            } else {
                PrefixOperator::Within(distance)
            }
        } else {
            return self.parse_primary();
        };
        Ok(operator.apply(self.parse_not()?))
    }

    fn peek_keyword(&self, offset: usize, keyword: &str) -> bool {
        matches!(
            self.tokens.get(self.cursor + offset),
            Some(Token { kind: TokenKind::Word(word), .. }) if word.eq_ignore_ascii_case(keyword)
        )
    }

    fn parse_primary(&mut self) -> Result<SelectionExpr, SelectionParseError> {
        if self.consume_kind(&TokenKind::LeftParen) {
            let expression = self.parse_or()?;
            if !self.consume_kind(&TokenKind::RightParen) {
                return Err(self.error_here("')'"));
            }
            return Ok(expression);
        }

        let token = self.next().ok_or_else(|| {
            SelectionParseError::new(self.input_len, "a selection predicate", "end of input")
        })?;
        let TokenKind::Word(keyword) = token.kind else {
            return Err(SelectionParseError::new(
                token.position,
                "a selection predicate",
                describe(&token),
            ));
        };
        match keyword.to_ascii_lowercase().as_str() {
            "all" => Ok(SelectionExpr::All),
            "none" => Ok(SelectionExpr::None),
            "hetatm" => Ok(SelectionExpr::Hetatm),
            "polymer" => Ok(SelectionExpr::Polymer),
            "protein" => Ok(SelectionExpr::Protein),
            "nucleic" => Ok(SelectionExpr::Nucleic),
            "water" | "waters" | "solvent" => Ok(SelectionExpr::Water),
            "ion" | "ions" => Ok(SelectionExpr::Ion),
            "ligand" | "ligands" | "organic" => Ok(SelectionExpr::Ligand),
            "backbone" | "bb" => Ok(SelectionExpr::Backbone),
            "sidechain" | "sc" => Ok(SelectionExpr::Sidechain),
            "hydrogen" | "hydrogens" => Ok(SelectionExpr::Hydrogen),
            "element" => {
                let (value, position) = self.value("an element symbol")?;
                let element = Element::from_str(&value).map_err(|()| {
                    SelectionParseError::new(position, "a supported element symbol", value)
                })?;
                Ok(SelectionExpr::Element(element))
            }
            "name" => self
                .value("an atom name")
                .map(|(value, _)| SelectionExpr::AtomName(value.to_ascii_uppercase())),
            "resn" => self
                .value("a residue name")
                .map(|(value, _)| SelectionExpr::ResidueName(value.to_ascii_uppercase())),
            "chain" => self
                .value("a chain identifier")
                .map(|(value, _)| SelectionExpr::Chain(value.to_ascii_uppercase())),
            "serial" => {
                let (value, position) = self.value("a positive atom serial")?;
                let serial = value.parse::<u32>().map_err(|_| {
                    SelectionParseError::new(position, "a positive atom serial", value)
                })?;
                Ok(SelectionExpr::Serial(serial))
            }
            "selection" => self
                .value("a named selection")
                .map(|(value, _)| SelectionExpr::Named(value)),
            "resi" => {
                let (value, position) = self.value("a residue number or range")?;
                parse_residue(value, position)
            }
            _ => Err(SelectionParseError::new(
                token.position,
                "a known selection predicate",
                keyword,
            )),
        }
    }

    fn value(&mut self, expected: &str) -> Result<(String, usize), SelectionParseError> {
        let token = self
            .next()
            .ok_or_else(|| SelectionParseError::new(self.input_len, expected, "end of input"))?;
        match token.kind {
            TokenKind::Word(value) => Ok((value, token.position)),
            _ => Err(SelectionParseError::new(
                token.position,
                expected,
                describe(&token),
            )),
        }
    }

    fn consume_keyword(&mut self, keyword: &str) -> bool {
        if matches!(self.peek(), Some(Token { kind: TokenKind::Word(word), .. }) if word.eq_ignore_ascii_case(keyword))
        {
            self.cursor += 1;
            true
        } else {
            false
        }
    }

    fn consume_kind(&mut self, kind: &TokenKind) -> bool {
        if self.peek().is_some_and(|token| token.kind == *kind) {
            self.cursor += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.cursor)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.cursor).cloned()?;
        self.cursor += 1;
        Some(token)
    }

    fn error_here(&self, expected: &str) -> SelectionParseError {
        self.peek().map_or_else(
            || SelectionParseError::new(self.input_len, expected, "end of input"),
            |token| SelectionParseError::new(token.position, expected, describe(token)),
        )
    }
}

fn parse_residue(value: String, position: usize) -> Result<SelectionExpr, SelectionParseError> {
    if let Some(separator) = value[1..].find('-').map(|index| index + 1) {
        let start = value[..separator].parse::<i32>();
        let end = value[separator + 1..].parse::<i32>();
        match (start, end) {
            (Ok(start), Ok(end)) if start <= end => Ok(SelectionExpr::ResidueRange(start, end)),
            _ => Err(SelectionParseError::new(
                position,
                "an ascending residue range such as 10-30",
                value,
            )),
        }
    } else {
        value
            .parse::<i32>()
            .map(SelectionExpr::ResidueNumber)
            .map_err(|_| SelectionParseError::new(position, "a residue number or range", value))
    }
}

fn describe(token: &Token) -> String {
    match &token.kind {
        TokenKind::Word(word) => format!("'{word}'"),
        TokenKind::LeftParen => "'('".into(),
        TokenKind::RightParen => "')'".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_predicates() {
        assert_eq!(parse_selection("all").unwrap(), SelectionExpr::All);
        assert_eq!(
            parse_selection("element C").unwrap(),
            SelectionExpr::Element(Element::C)
        );
        assert_eq!(
            parse_selection("resi 10-20").unwrap(),
            SelectionExpr::ResidueRange(10, 20)
        );
    }

    #[test]
    fn parses_filesystem_style_chain_masks() {
        assert!(matches!(
            parse_selection("Chain A/LEU*").unwrap(),
            SelectionExpr::And(left, right)
                if matches!(*left, SelectionExpr::Chain(ref chain) if chain == "A")
                    && matches!(*right, SelectionExpr::ResidueNamePattern(ref name) if name == "LEU*")
        ));
        let parsed = parse_selection("Chain B/[20:22, 70:71]").unwrap();
        assert!(matches!(
            parsed,
            SelectionExpr::And(left, right)
                if matches!(*left, SelectionExpr::Chain(ref chain) if chain == "B")
                    && matches!(*right, SelectionExpr::Or(_, _))
        ));
        assert!(parse_selection("Chain A/[22:20]").is_err());
    }

    #[test]
    fn honors_not_and_parentheses() {
        let parsed = parse_selection("chain A and (element N or element O)").unwrap();
        assert!(
            matches!(parsed, SelectionExpr::And(_, right) if matches!(*right, SelectionExpr::Or(_, _)))
        );
        assert!(matches!(
            parse_selection("not (chain A or chain B)").unwrap(),
            SelectionExpr::Not(inner) if matches!(*inner, SelectionExpr::Or(_, _))
        ));
    }

    #[test]
    fn and_has_higher_precedence_than_or() {
        let parsed = parse_selection("chain A or chain B and element C").unwrap();
        assert!(
            matches!(parsed, SelectionExpr::Or(_, right) if matches!(*right, SelectionExpr::And(_, _)))
        );
    }

    #[test]
    fn xor_has_precedence_between_and_and_or() {
        let parsed = parse_selection("all or none xor all and none").unwrap();
        assert!(matches!(
            parsed,
            SelectionExpr::Or(_, right)
                if matches!(&*right, SelectionExpr::Xor(_, and) if matches!(&**and, SelectionExpr::And(_, _)))
        ));
    }

    #[test]
    fn combines_path_masks_with_boolean_operators() {
        let parsed = parse_selection("../LEU* AND [20:30, 45:50]").unwrap();
        assert!(
            matches!(parsed, SelectionExpr::And(_, right) if matches!(*right, SelectionExpr::Or(_, _)))
        );
        let parsed = parse_selection("Chain A/LEU* XOR (Chain B/HOH* OR element O)").unwrap();
        assert!(
            matches!(parsed, SelectionExpr::Xor(_, right) if matches!(*right, SelectionExpr::Or(_, _)))
        );
    }

    #[test]
    fn rejects_incomplete_and_unknown_syntax_with_position() {
        let error = parse_selection("chain A and").unwrap_err();
        assert_eq!(error.position, 11);
        assert!(error.expected.contains("predicate"));
        assert!(parse_selection("banana A").is_err());
        assert!(parse_selection("(chain A").is_err());
    }

    #[test]
    fn parses_spatial_and_expansion_operators() {
        assert_eq!(
            parse_selection("within 4.5 of resn HEM and not water").unwrap(),
            SelectionExpr::And(
                Box::new(SelectionExpr::Within(
                    4500,
                    Box::new(SelectionExpr::ResidueName("HEM".into()))
                )),
                Box::new(SelectionExpr::Not(Box::new(SelectionExpr::Water))),
            )
        );
        assert_eq!(
            parse_selection("byres around 3 of (ligand or ion)").unwrap(),
            SelectionExpr::ByResidue(Box::new(SelectionExpr::Around(
                3000,
                Box::new(SelectionExpr::Or(
                    Box::new(SelectionExpr::Ligand),
                    Box::new(SelectionExpr::Ion)
                ))
            )))
        );
        assert_eq!(
            parse_selection("same residue as serial 5").unwrap(),
            SelectionExpr::ByResidue(Box::new(SelectionExpr::Serial(5)))
        );
        assert_eq!(
            parse_selection("same chain as hydrogen").unwrap(),
            SelectionExpr::ByChain(Box::new(SelectionExpr::Hydrogen))
        );
        assert_eq!(
            parse_selection("bychain within 5 of Chain A/[10:12]").unwrap(),
            SelectionExpr::ByChain(Box::new(SelectionExpr::Within(
                5000,
                Box::new(SelectionExpr::And(
                    Box::new(SelectionExpr::Chain("A".into())),
                    Box::new(SelectionExpr::ResidueRange(10, 12))
                ))
            )))
        );
        assert!(parse_selection("within five of protein").is_err());
        assert!(parse_selection("within 5 protein").is_err());
        assert!(parse_selection("within -1 of protein").is_err());
        assert!(parse_selection("around 5 of Chain A/LEU* and").is_err());
    }

    #[test]
    fn formatting_round_trips_through_the_parser() {
        for source in [
            "within 4.5 of resn HEM and not water",
            "byres (within 3 of ligand or ion)",
            "not around 2.25 of chain A",
            "bychain (protein and backbone) or sidechain xor hydrogen",
        ] {
            let parsed = parse_selection(source).unwrap();
            let formatted = super::super::format_expression(&parsed);
            assert_eq!(parse_selection(&formatted).unwrap(), parsed, "{formatted}");
        }
    }
}
