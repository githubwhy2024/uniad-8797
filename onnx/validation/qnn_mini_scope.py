"""Explicit frozen mini dataset coverage, independent of full/planning task scope."""
import copy


def dataset_scope(selection='all'):
    if selection not in ('all', 'mini_val'):
        raise ValueError('unknown validation split selection')
    splits = ['mini_val', 'mini_train'] if selection == 'all' else ['mini_val']
    return dict(schema='qnn-mini-dataset-scope-v1', selection=selection,
                splits=splits, frames=404 if selection == 'all' else 81,
                complete_stage='mini404_records' if selection == 'all' else 'mini_val81_records')


def scope_of(container):
    # Historical records lacking a marker remain exclusively 404 records.
    value = container.get('dataset_scope', dataset_scope())
    if not isinstance(value, dict) or value != dataset_scope(value.get('selection')):
        raise ValueError('noncanonical mini dataset scope')
    return copy.deepcopy(value)


def check_counts(scope, task_scope):
    if scope != dataset_scope(scope['selection']) or task_scope not in ('full', 'planning'):
        raise ValueError('unknown dataset/task coverage')
    return (96 if task_scope == 'full' else 9) * len(scope['splits']), 6 * len(scope['splits'])


def require_complete(mini, *companions):
    scope = scope_of(mini)
    if mini['status'] != 'pass' or mini['execution_status'] != 'complete' or mini['stage'] != scope['complete_stage'] or mini['frames'] != scope['frames']:
        raise ValueError('complete declared mini dataset required')
    if set(mini['split_summaries']) != set(scope['splits']):
        raise ValueError('saved split coverage differs')
    for split in scope['splits']:
        row = mini['split_summaries'][split]
        if row['status'] != 'pass' or row['frames'] != (81 if split == 'mini_val' else 323):
            raise ValueError('saved split is incomplete')
    for companion in companions:
        if scope_of(companion) != scope or ('frames' in companion and companion['frames'] != scope['frames']):
            raise ValueError('dataset coverage differs across evidence')
    return scope
