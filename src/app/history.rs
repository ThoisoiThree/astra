use super::*;

#[derive(Debug, Clone, PartialEq)]
/// Snapshot of undoable editor state. Camera, focus, pivot, and viewport state are intentionally
/// absent: they are saved in a scene but camera interaction must never consume an undo step.
pub(super) struct EditTransaction {
    pub(super) display: Option<DisplayStateData>,
    pub(super) named_selections: BTreeMap<String, NamedSelectionRecord>,
    pub(super) measurements: BTreeMap<u64, IndexedMeasurement>,
    pub(super) hierarchy_names: BTreeMap<InspectionTarget, String>,
    pub(super) workspace: WorkspaceSelection,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct NamedSelectionRecord {
    pub(super) selection: Selection,
    pub(super) expression: Option<String>,
    pub(super) style: Option<NamedSelectionStyle>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct IndexedMeasurement {
    pub(super) index: usize,
    pub(super) line: MeasurementLine,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct WorkspaceSelection {
    pub(super) inspection: Option<InspectionTarget>,
    pub(super) hierarchy_selection: BTreeSet<InspectionTarget>,
    pub(super) hierarchy_selection_anchor: Option<InspectionTarget>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ValueChange<T> {
    pub(super) before: T,
    pub(super) after: T,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct NamedSelectionChange {
    pub(super) name: String,
    pub(super) value: ValueChange<Option<NamedSelectionRecord>>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct MeasurementChange {
    pub(super) id: u64,
    pub(super) value: ValueChange<Option<IndexedMeasurement>>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct HierarchyNameChange {
    pub(super) target: InspectionTarget,
    pub(super) value: ValueChange<Option<String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum EditChange {
    Display(Box<ValueChange<Option<DisplayStateData>>>),
    NamedSelection(Box<NamedSelectionChange>),
    Measurement(Box<MeasurementChange>),
    HierarchyName(Box<HierarchyNameChange>),
    Workspace(Box<ValueChange<WorkspaceSelection>>),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct EditOperation {
    pub(super) changes: Vec<EditChange>,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum HistoryDirection {
    Undo,
    Redo,
}

impl EditOperation {
    pub(super) fn between(before: EditTransaction, after: EditTransaction) -> Self {
        let mut changes = Vec::new();
        if before.display != after.display {
            changes.push(EditChange::Display(Box::new(ValueChange {
                before: before.display,
                after: after.display,
            })));
        }

        let selection_names: BTreeSet<_> = before
            .named_selections
            .keys()
            .chain(after.named_selections.keys())
            .cloned()
            .collect();
        for name in selection_names {
            let old = before.named_selections.get(&name).cloned();
            let new = after.named_selections.get(&name).cloned();
            if old != new {
                changes.push(EditChange::NamedSelection(Box::new(NamedSelectionChange {
                    name,
                    value: ValueChange {
                        before: old,
                        after: new,
                    },
                })));
            }
        }

        let measurement_ids: BTreeSet<_> = before
            .measurements
            .keys()
            .chain(after.measurements.keys())
            .copied()
            .collect();
        for id in measurement_ids {
            let old = before.measurements.get(&id).cloned();
            let new = after.measurements.get(&id).cloned();
            if old != new {
                changes.push(EditChange::Measurement(Box::new(MeasurementChange {
                    id,
                    value: ValueChange {
                        before: old,
                        after: new,
                    },
                })));
            }
        }

        let hierarchy_targets: BTreeSet<_> = before
            .hierarchy_names
            .keys()
            .chain(after.hierarchy_names.keys())
            .copied()
            .collect();
        for target in hierarchy_targets {
            let old = before.hierarchy_names.get(&target).cloned();
            let new = after.hierarchy_names.get(&target).cloned();
            if old != new {
                changes.push(EditChange::HierarchyName(Box::new(HierarchyNameChange {
                    target,
                    value: ValueChange {
                        before: old,
                        after: new,
                    },
                })));
            }
        }

        if before.workspace != after.workspace {
            changes.push(EditChange::Workspace(Box::new(ValueChange {
                before: before.workspace,
                after: after.workspace,
            })));
        }
        Self { changes }
    }
}

pub(super) fn history_value<T>(change: &ValueChange<T>, direction: HistoryDirection) -> &T {
    match direction {
        HistoryDirection::Undo => &change.before,
        HistoryDirection::Redo => &change.after,
    }
}

pub(super) fn set_optional_map_value<K: Ord, V>(
    map: &mut BTreeMap<K, V>,
    key: K,
    value: Option<V>,
) {
    if let Some(value) = value {
        map.insert(key, value);
    } else {
        map.remove(&key);
    }
}

pub(super) fn push_history(history: &mut VecDeque<EditOperation>, operation: EditOperation) {
    if history.len() == EDIT_HISTORY_LIMIT {
        history.pop_front();
    }
    history.push_back(operation);
}
