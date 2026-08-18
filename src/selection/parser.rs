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
            "'and', 'or', or end of input",
            describe(token),
        ));
    }
    Ok(expression)
}

struct Parser {
    tokens: Vec<Token>,
    cursor: usize,
    input_len: usize,
}

impl Parser {
    fn parse_or(&mut self) -> Result<SelectionExpr, SelectionParseError> {
        let mut expression = self.parse_and()?;
        while self.consume_keyword("or") {
            let right = self.parse_and()?;
            expression = SelectionExpr::Or(Box::new(expression), Box::new(right));
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
    fn rejects_incomplete_and_unknown_syntax_with_position() {
        let error = parse_selection("chain A and").unwrap_err();
        assert_eq!(error.position, 11);
        assert!(error.expected.contains("predicate"));
        assert!(parse_selection("banana A").is_err());
        assert!(parse_selection("(chain A").is_err());
    }
}
