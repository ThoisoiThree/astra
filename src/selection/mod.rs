mod ast;
mod eval;
mod lexer;
mod parser;
mod registry;

pub use ast::{MAX_SELECTION_DISTANCE, SelectionExpr, distance_to_milli, milli_to_distance};
pub use eval::{Selection, SelectionEvaluationError, evaluate, evaluate_with_named};
pub use parser::{SelectionParseError, parse_selection};
pub use registry::{
    SelectionResolution, SelectionStatus, canonical_name, find_display_name, format_expression,
    named_dependencies, rename_named_reference, resolve_named_expressions, validate_unique_name,
};
