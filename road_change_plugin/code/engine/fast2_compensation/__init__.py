"""Optional domain compensation; geometry, evidence and publication stay outside."""
from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class Fast2CompensationConfig:
    patch_radiometric_normalization: bool = True
    stable_appearance_calibration: bool = True
    surface_probability_calibration: bool = True
    width_temporal_bias_correction: bool = True

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not bool: raise TypeError(f'{name} must be a boolean')

    @classmethod
    def from_preset(cls, preset='raw_input', **overrides):
        if preset not in ('raw_input', 'normalized_input', 'custom'):
            raise ValueError(f'Unknown Fast2 compensation preset: {preset}')
        if overrides and preset != 'custom':
            raise ValueError('Individual overrides require the custom preset')
        defaults={name:preset!='normalized_input' for name in cls.__dataclass_fields__}
        return cls(**(defaults | overrides))

    @classmethod
    def resolve(cls, value=None):
        if value is None: return cls()
        if isinstance(value, cls): return value
        if isinstance(value, str):
            return cls.resolve(json.loads(value)) if value.lstrip().startswith('{') else cls.from_preset(value)
        if isinstance(value, dict):
            options=dict(value);preset=options.pop('preset','custom')
            return cls.from_preset(preset, **options)
        raise TypeError('Expected Fast2CompensationConfig, preset name or configuration mapping')

    def to_dict(self): return asdict(self)

    @property
    def cache_identity(self):
        # Include every requested switch, even an identity-only implementation.
        return 'fast2_compensation_v1:' + ''.join(str(int(v)) for v in asdict(self).values())


class Fast2Compensation:
    """One strategy bundle per comparison; all enable checks live in modules."""
    def __init__(self, config=None):
        from .patch_radiometric_normalization import PatchRadiometricNormalization
        from .stable_appearance_calibration import StableAppearanceCalibration
        from .surface_probability_calibration import SurfaceProbabilityCalibration
        from .width_temporal_bias_correction import WidthTemporalBiasCorrection
        self.config=Fast2CompensationConfig.resolve(config)
        self.patch=PatchRadiometricNormalization(self.config.patch_radiometric_normalization)
        self.appearance=StableAppearanceCalibration(self.config.stable_appearance_calibration)
        self.surface_probability=SurfaceProbabilityCalibration(self.config.surface_probability_calibration)
        self.width=WidthTemporalBiasCorrection(self.config.width_temporal_bias_correction)

    def metadata(self):
        return dict(config=self.config.to_dict(), cache_identity=self.config.cache_identity,
                    surface_probability_implementation='identity_no_existing_cross_period_correction')


def cache_matches(result, config=None):
    """Old unlabelled results are only compatible with the exact raw default."""
    from ..fast_multitemporal import AUTO_REVISION
    config=Fast2CompensationConfig.resolve(config)
    return (result.get('fast_auto_revision')==AUTO_REVISION and
            result.get('fast2_compensation_identity',Fast2CompensationConfig().cache_identity)==config.cache_identity)
