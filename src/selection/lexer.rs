use super::parser::SelectionParseError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TokenKind {
    Word(String),
    LeftParen,
    RightParen,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Token {
    pub kind: TokenKind,
    pub position: usize,
}

pub(crate) fn lex(input: &str) -> Result<Vec<Token>, SelectionParseError> {
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < input.len() {
        let character = input[index..]
            .chars()
            .next()
            .ok_or_else(|| SelectionParseError::new(index, "a selection token", "end of input"))?;
        if character.is_whitespace() {
            index += character.len_utf8();
            continue;
        }
        let kind = match character {
            '(' => {
                index += 1;
                TokenKind::LeftParen
            }
            ')' => {
                index += 1;
                TokenKind::RightParen
            }
            _ if character.is_ascii_alphanumeric()
                || matches!(character, '_' | '-' | '+' | '\'' | '*') =>
            {
                let start = index;
                while index < input.len() {
                    let current = input[index..].chars().next().ok_or_else(|| {
                        SelectionParseError::new(index, "a selection token", "end of input")
                    })?;
                    if current.is_whitespace() || matches!(current, '(' | ')') {
                        break;
                    }
                    if !(current.is_ascii_alphanumeric()
                        || matches!(current, '_' | '-' | '+' | '\'' | '*'))
                    {
                        return Err(SelectionParseError::new(
                            index,
                            "an alphanumeric selection token",
                            current.to_string(),
                        ));
                    }
                    index += current.len_utf8();
                }
                TokenKind::Word(input[start..index].to_string())
            }
            _ => {
                return Err(SelectionParseError::new(
                    index,
                    "a selection token or parenthesis",
                    character.to_string(),
                ));
            }
        };
        let position = match kind {
            TokenKind::Word(_) => index.saturating_sub(match &kind {
                TokenKind::Word(word) => word.len(),
                _ => 0,
            }),
            _ => index - 1,
        };
        tokens.push(Token { kind, position });
    }
    Ok(tokens)
}
