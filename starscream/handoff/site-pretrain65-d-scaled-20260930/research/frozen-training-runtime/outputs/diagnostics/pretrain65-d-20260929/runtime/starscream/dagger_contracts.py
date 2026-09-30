"""Fail closed when weights or statistics use different observation units."""


def previous_action_mapping(payload):
    if 'previous_action_feature_mapping' in payload:
        return payload['previous_action_feature_mapping']
    config = payload.get('training_config', {})
    settings = config.get('dagger', config)
    return settings.get('previous_action_feature_mapping', 'legacy_linear')


def validate_previous_action_mapping(payload, settings):
    requested = settings.get('previous_action_feature_mapping', 'legacy_linear')
    if previous_action_mapping(payload) != requested:
        raise ValueError('previous-action feature mapping mismatch; collect new statistics and initialize from scratch')
