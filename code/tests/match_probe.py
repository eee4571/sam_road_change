"""Experiment-only RoadScene.match timings; never changes algorithm statistics."""
import ast
from collections import Counter
import inspect
import textwrap
import time


def install(module, *, detailed=False):
    """Install a per-module probe and return its independent report callable."""
    seconds, counts = Counter(), Counter()
    original = module.RoadScene.match
    sampled = _instrument(original, module.__dict__, seconds, counts) if detailed else original

    def match(self, *args, **kwargs):
        started = time.perf_counter()
        counts['match_calls'] += 1
        try:
            return sampled(self, *args, **kwargs)
        finally:
            seconds['match_total'] += time.perf_counter()-started

    module.RoadScene.match = match

    def report():
        return dict(mode='detailed' if detailed else 'trace',
                    seconds=dict(seconds), counts=dict(counts))
    return report


def _instrument(function, globals_, seconds, counts):
    """Wrap original statement groups; preserve their expressions and order.

    This accepts both the original combined direction/overlap rejection and
    an earlier direction continue, as well as a hoisted source corridor. No
    production expression is recreated from a hand-maintained implementation.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    method = tree.body[0]
    assert isinstance(method, ast.FunctionDef)
    assert not any(isinstance(node, ast.Name) and node.id.startswith('_match_probe_')
                   for node in ast.walk(method)), 'reserved probe name in method'

    def statements(source):
        return ast.parse(source).body

    def count(name, value='1'):
        return statements(f"_match_probe_counts[{name!r}] += {value}")

    def timed(stage, body):
        return (statements('_match_probe_tick = _match_probe_clock()') +
                [ast.Try(body=body, handlers=[], orelse=[], finalbody=statements(
                    f"_match_probe_seconds[{stage!r}] += _match_probe_clock()-_match_probe_tick"))])

    def assigned(node):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            return node.target.id
        return None

    def stage_for(node, *, target):
        name = assigned(node)
        if target:
            if name in ('target', 'target_station', 'target_point', 'distance'):
                return 'target_project_interpolate_distance'
            if name == 'cosine':
                return 'target_normal_cosine'
            if name == 'target_local':
                return 'target_substring'
            if name == 'overlap':
                return 'overlap_buffer_intersection'
            if name == 'target_width':
                return 'width_lookup'
            if name == 'corridor_overlap':
                return 'corridor_buffer_intersection'
            if name in ('compatibility', 'score'):
                return 'score_rank'
        else:
            if name == 'point':
                return 'source_interpolate'
            if name == 'local':
                return 'source_substring'
            if name == 'normal':
                return 'source_normal'
            if name == 'local_length':
                return 'source_length'
        if name == 'source_corridor':
            return 'source_corridor_buffer'
        if isinstance(node, ast.If):
            names = {part.id for part in ast.walk(node.test) if isinstance(part, ast.Name)}
            if 'source_corridor' in names:
                return 'source_corridor_buffer'
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
                isinstance(node.value.func, ast.Attribute) and
                ast.unparse(node.value.func) == 'ranked.append'):
            return 'score_rank'
        return None

    def grouped(body, *, target):
        result, pending, previous_stage = [], [], None

        def flush():
            if pending:
                result.extend(timed(previous_stage, pending[:]) if previous_stage else pending)
                pending.clear()

        for node in body:
            stage = stage_for(node, target=target)
            if stage != previous_stage:
                flush()
                previous_stage = stage
            pending.append(node)
            if target and isinstance(node, ast.If) and any(isinstance(item, ast.Continue) for item in node.body):
                names = {part.id for part in ast.walk(node.test) if isinstance(part, ast.Name)}
                if names == {'cosine', 'overlap'}:
                    node.body[:0] = statements("_match_probe_counts['direction_rejected' if cosine < .90 else 'overlap_rejected'] += 1")
                elif names == {'cosine'}:
                    node.body[:0] = count('direction_rejected')
                elif names == {'overlap'}:
                    node.body[:0] = count('overlap_rejected')
                else:
                    raise AssertionError(f'unknown match rejection: {ast.unparse(node.test)}')
            if target and stage == 'score_rank' and isinstance(node, ast.Expr):
                pending.extend(count('ranked_candidates'))
        flush()
        return result

    loops = [i for i, node in enumerate(method.body) if isinstance(node, ast.For)]
    assert len(loops) == 1, 'expected one target query loop'
    index = loops[0]
    loop = method.body[index]
    assert isinstance(loop.iter, ast.Call) and ast.unparse(loop.iter.func) == 'self.tree.query'
    query = ast.Assign(targets=[ast.Name(id='_match_probe_candidates', ctx=ast.Store())], value=loop.iter)
    loop.iter = ast.Name(id='_match_probe_candidates', ctx=ast.Load())
    loop.body = count('candidate_visits') + grouped(loop.body, target=True)
    method.body = (grouped(method.body[:index], target=False) + timed('strtree_query', [query]) +
                   count('query_candidates', 'len(_match_probe_candidates)') + [loop] +
                   timed('sort_ambiguity', method.body[index+1:]))
    namespace = dict(globals_, _match_probe_seconds=seconds, _match_probe_counts=counts,
                     _match_probe_clock=time.perf_counter)
    exec(compile(ast.fix_missing_locations(tree), '<match-stage-probe>', 'exec'), namespace)
    return namespace[function.__name__]
