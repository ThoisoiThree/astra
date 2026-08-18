use std::{fmt, str::FromStr};

use crate::molecule::Element;

use super::{
    ast::SelectionExpr,
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

    if let Some(rest) = strip_keyword_prefix(input, "not") {
        let rest_position = position + input.len() - rest.len();
        return Ok(SelectionExpr::Not(Box::new(parse_mask_expression(
            rest,
            rest_position,
        )?)));
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
    if !(2..=3).contains(&segments.len()) || segments.iter().any(|segment| segment.is_empty()) {
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

    fn parse_not(&mut self) -> Result<SelectionExpr, SelectionParseError> {
        if self.consume_keyword("not") {
            Ok(SelectionExpr::Not(Box::new(self.parse_not()?)))
        } else {
            self.parse_primary()
        }
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
}
