use thiserror::Error;

use crate::selection::{SelectionExpr, SelectionParseError, parse_selection};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Representation {
    Spheres,
    Sticks,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color(pub [f32; 4]);

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Select {
        name: Option<String>,
        selection: SelectionExpr,
    },
    Color {
        color: Color,
        selection: SelectionExpr,
    },
    Show {
        representation: Representation,
        selection: SelectionExpr,
    },
    Hide {
        representation: Representation,
        selection: SelectionExpr,
    },
}

#[derive(Debug, Error, PartialEq)]
pub enum CommandError {
    #[error("empty command; expected select, color, show, or hide")]
    Empty,
    #[error("unknown command '{0}'; expected select, color, show, or hide")]
    UnknownCommand(String),
    #[error("{command} expects '{usage}'")]
    InvalidShape {
        command: &'static str,
        usage: &'static str,
    },
    #[error("unknown representation '{0}'; expected spheres or sticks")]
    UnknownRepresentation(String),
    #[error("invalid color '{0}'; use a name such as red or #RRGGBB")]
    InvalidColor(String),
    #[error("invalid selection name '{0}'; use letters, numbers, '_' or '-'")]
    InvalidSelectionName(String),
    #[error(transparent)]
    Selection(#[from] SelectionParseError),
}

pub fn parse_command(input: &str) -> Result<Command, CommandError> {
    let input = input.trim();
    if input.is_empty() {
        return Err(CommandError::Empty);
    }
    let split = input.find(char::is_whitespace).unwrap_or(input.len());
    let command = &input[..split];
    let arguments = input[split..].trim();
    match command.to_ascii_lowercase().as_str() {
        "select" => {
            if arguments.is_empty() {
                return Err(CommandError::InvalidShape {
                    command: "select",
                    usage: "select <expression>",
                });
            }
            if arguments.contains(',') {
                let (name, selection) =
                    comma_arguments(arguments, "select", "select <name>, <expression>")?;
                if !valid_selection_name(name) {
                    return Err(CommandError::InvalidSelectionName(name.to_string()));
                }
                Ok(Command::Select {
                    name: Some(name.to_string()),
                    selection: parse_selection(selection)?,
                })
            } else {
                Ok(Command::Select {
                    name: None,
                    selection: parse_selection(arguments)?,
                })
            }
        }
        "color" => {
            let (color, selection) =
                comma_arguments(arguments, "color", "color <color>, <expression>")?;
            Ok(Command::Color {
                color: parse_color(color)?,
                selection: parse_selection(selection)?,
            })
        }
        "show" | "hide" => {
            let static_command = if command.eq_ignore_ascii_case("show") {
                "show"
            } else {
                "hide"
            };
            let usage = if static_command == "show" {
                "show <representation>, <expression>"
            } else {
                "hide <representation>, <expression>"
            };
            let (representation, selection) = comma_arguments(arguments, static_command, usage)?;
            let representation = parse_representation(representation)?;
            let selection = parse_selection(selection)?;
            if static_command == "show" {
                Ok(Command::Show {
                    representation,
                    selection,
                })
            } else {
                Ok(Command::Hide {
                    representation,
                    selection,
                })
            }
        }
        _ => Err(CommandError::UnknownCommand(command.to_string())),
    }
}

fn comma_arguments<'a>(
    arguments: &'a str,
    command: &'static str,
    usage: &'static str,
) -> Result<(&'a str, &'a str), CommandError> {
    let (first, second) = arguments
        .split_once(',')
        .ok_or(CommandError::InvalidShape { command, usage })?;
    let first = first.trim();
    let second = second.trim();
    if first.is_empty() || second.is_empty() {
        return Err(CommandError::InvalidShape { command, usage });
    }
    Ok((first, second))
}

fn parse_representation(value: &str) -> Result<Representation, CommandError> {
    match value.to_ascii_lowercase().as_str() {
        "spheres" => Ok(Representation::Spheres),
        "sticks" => Ok(Representation::Sticks),
        _ => Err(CommandError::UnknownRepresentation(value.to_string())),
    }
}

fn parse_color(value: &str) -> Result<Color, CommandError> {
    let rgba = match value.to_ascii_lowercase().as_str() {
        "red" => [0.95, 0.10, 0.10, 1.0],
        "green" => [0.12, 0.80, 0.25, 1.0],
        "blue" => [0.15, 0.35, 0.95, 1.0],
        "yellow" => [1.00, 0.85, 0.10, 1.0],
        "orange" => [1.00, 0.45, 0.08, 1.0],
        "magenta" => [0.95, 0.15, 0.80, 1.0],
        "cyan" => [0.10, 0.85, 0.90, 1.0],
        "white" => [0.95, 0.95, 0.95, 1.0],
        "gray" | "grey" => [0.55, 0.55, 0.55, 1.0],
        hexadecimal if hexadecimal.len() == 7 && hexadecimal.starts_with('#') => {
            let parsed = u32::from_str_radix(&hexadecimal[1..], 16)
                .map_err(|_| CommandError::InvalidColor(value.to_string()))?;
            [
                ((parsed >> 16) & 0xff) as f32 / 255.0,
                ((parsed >> 8) & 0xff) as f32 / 255.0,
                (parsed & 0xff) as f32 / 255.0,
                1.0,
            ]
        }
        _ => return Err(CommandError::InvalidColor(value.to_string())),
    };
    Ok(Color(rgba))
}

fn valid_selection_name(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|character| character.is_ascii_alphabetic() || character == '_')
        && characters
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{molecule::Element, selection::SelectionExpr};

    #[test]
    fn parses_required_commands() {
        assert_eq!(
            parse_command("select chain A").unwrap(),
            Command::Select {
                name: None,
                selection: SelectionExpr::Chain("A".into()),
            }
        );
        assert_eq!(
            parse_command("select active_site, chain A and resi 10-20").unwrap(),
            Command::Select {
                name: Some("active_site".into()),
                selection: SelectionExpr::And(
                    Box::new(SelectionExpr::Chain("A".into())),
                    Box::new(SelectionExpr::ResidueRange(10, 20)),
                ),
            }
        );
        assert!(matches!(
            parse_command("color red, chain A and element C").unwrap(),
            Command::Color {
                color: Color([0.95, 0.10, 0.10, 1.0]),
                selection: SelectionExpr::And(_, _)
            }
        ));
        assert_eq!(
            parse_command("color #ff00aa, hetatm").unwrap(),
            Command::Color {
                color: Color([1.0, 0.0, 170.0 / 255.0, 1.0]),
                selection: SelectionExpr::Hetatm,
            }
        );
        assert_eq!(
            parse_command("show sticks, chain B").unwrap(),
            Command::Show {
                representation: Representation::Sticks,
                selection: SelectionExpr::Chain("B".into()),
            }
        );
        assert_eq!(
            parse_command("hide spheres, element H").unwrap(),
            Command::Hide {
                representation: Representation::Spheres,
                selection: SelectionExpr::Element(Element::H),
            }
        );
    }

    #[test]
    fn rejects_bad_commands() {
        assert!(matches!(
            parse_command("color red chain A"),
            Err(CommandError::InvalidShape { .. })
        ));
        assert!(matches!(
            parse_command("show cartoon, all"),
            Err(CommandError::UnknownRepresentation(_))
        ));
        assert!(matches!(
            parse_command("color #xyzxyz, all"),
            Err(CommandError::InvalidColor(_))
        ));
        assert!(matches!(
            parse_command("select bad name, all"),
            Err(CommandError::InvalidSelectionName(_))
        ));
    }
}
