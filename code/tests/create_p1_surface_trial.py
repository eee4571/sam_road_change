"""Create an isolated P1 source snapshot; never modify production code."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]/'engine/fast_auto_change.py'
    text = source.read_text(encoding='utf-8')
    evidence = '    surface = scene.surface(axis)\n'
    assert text.count(evidence) == 1
    text = text.replace(evidence, '    surface = scene.surface(axis, tolerance+2.0)\n')
    width = '''                    if width_surfaces is None:
                        width_surfaces = ((source_surface, source_surface.buffer(.1)),
                                          (target_surface, target_surface.buffer(.1)))'''
    assert text.count(width) == 1
    text = text.replace(width, '''                    if width_surfaces is None:
                        width_surface_started = time.perf_counter()
                        source_width_surface = source.surface(axis)
                        target_width_surface = target.surface(axis)
                        width_surfaces = ((source_width_surface, source_width_surface.buffer(.1)),
                                          (target_width_surface, target_width_surface.buffer(.1)))
                        timing['width_surface_lazy_build'] += time.perf_counter()-width_surface_started''')
    path = args.directory/'p1_trial_source.py'
    path.write_text(text,encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
