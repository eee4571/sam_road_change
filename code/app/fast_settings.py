"""User-facing Fast settings; no dependency on inference modules."""
import json

COMPENSATION_LABELS = (
    ('patch_radiometric_normalization', '候选影像局部辐射校正'),
    ('stable_appearance_calibration', '稳定道路外观基线校正'),
    ('surface_probability_calibration', '道路面与概率跨期校正'),
    ('width_temporal_bias_correction', '道路宽度跨期偏差校正'),
)
SETTING_DEFAULTS = dict(irmad='0', irmad_reference='20250118',
                        **{name: '1' for name, _ in COMPENSATION_LABELS})


def settings_values(variables):
    return {key: variables[key].get() if key in variables else default
            for key, default in SETTING_DEFAULTS.items()}


def restore_settings(variables, settings):
    for key, default in SETTING_DEFAULTS.items():
        if key in variables:
            variables[key].set(default if key == "irmad_reference" else str(settings.get(key, default)))


def compensation_values(variables):
    values = settings_values(variables)
    return {name: values[name] == '1' for name, _ in COMPENSATION_LABELS}


def compensation_arguments(config):
    return [] if config is None else ['--fast2-compensation', json.dumps(config, sort_keys=True)]
