"""Opt-in coverage requirements. No detection or tracking performance model."""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math

import numpy as np

from src.core.evaluation import objective_from_density
from src.core.objective import ObjectiveWeights


def _keys(value, allowed, context):
    if not isinstance(value, dict):
        raise ValueError(f'{context} must be an object')
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f'Unknown {context} keys: {sorted(unknown)}')


def _number(value, name, lower=0., upper=math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be numeric')
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f'{name} must be finite in [{lower}, {upper}]')


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _bounds(value):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError('bounds must be [xmin, ymin, xmax, ymax]')
    for coordinate in value:
        _number(coordinate, 'bounds', -math.inf)
    if value[0] >= value[2] or value[1] >= value[3]:
        raise ValueError('bounds must have positive width and height')


def _inside(points, bounds):
    lo, hi = np.asarray(bounds[:2]), np.asarray(bounds[2:])
    return np.all((points[:, :2] >= lo) & (points[:, :2] <= hi), axis=1)


def _list(value, name):
    if not isinstance(value, list):
        raise ValueError(f'{name} must be an array')
    return value


def _thresholds(data):
    if 'min_coverage_pct' in data:
        _number(data['min_coverage_pct'], 'min_coverage_pct', upper=100.)
    for name in ('min_overlap_ratio', 'max_overlap_ratio'):
        if name in data:
            _number(data[name], name, upper=1.)
    if data.get('min_overlap_ratio', 0.) > data.get('max_overlap_ratio', 1.):
        raise ValueError('Minimum overlap exceeds maximum overlap')


@dataclass(frozen=True)
class CoverageAssessment:
    objective_score: float
    coverage_pct: float
    overlap_ratio: float
    components: dict
    regions: dict
    feasible: bool
    violations: tuple
    constraint_residuals: dict = field(default_factory=dict)
    robustness: dict = field(default_factory=dict)
    infrastructure: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class CoverageRequirements:
    """Strict, JSON-serializable specification with rectangle-based region masks."""
    def __init__(self, specification):
        _keys(specification, ('schema', 'budget_k', 'weights', 'cost', 'constraints', 'regions'), 'specification')
        if specification.get('schema') not in ('constrained_coverage_v1', 'constrained_coverage_v2'):
            raise ValueError('Expected schema constrained_coverage_v1 or constrained_coverage_v2')
        extended = specification['schema'] == 'constrained_coverage_v2'
        _integer(specification.get('budget_k'), 'budget_k')
        self.specification = deepcopy(specification)
        weights = specification.get('weights', {})
        _keys(weights, ObjectiveWeights.__dataclass_fields__, 'weights')
        for key, value in weights.items():
            _number(value, key)
        self.weights = ObjectiveWeights(**weights)
        cost = specification.get('cost', {})
        _keys(cost, ('base_per_sensor', 'per_height_m', 'regions'), 'cost')
        for key in ('base_per_sensor', 'per_height_m'):
            _number(cost.get(key, 0.), key)
        for region in _list(cost.get('regions', []), 'cost.regions'):
            _keys(region, ('bounds', 'surcharge'), 'cost region')
            _bounds(region.get('bounds'))
            _number(region.get('surcharge'), 'surcharge')
        constraints = specification.get('constraints', {})
        extra = ('mounting_z_range_m', 'allowed_installation_regions', 'service_hubs',
                 'min_single_failure_coverage_pct') if extended else ()
        _keys(constraints, ('max_installation_cost', 'min_xy_separation_m', 'max_sensors_per_pole',
            'forbidden_regions', 'min_coverage_pct', 'min_overlap_ratio', 'max_overlap_ratio') + extra, 'constraints')
        _thresholds(constraints)
        if 'min_single_failure_coverage_pct' in constraints:
            _number(constraints['min_single_failure_coverage_pct'], 'min_single_failure_coverage_pct', upper=100.)
        if 'mounting_z_range_m' in constraints:
            levels = constraints['mounting_z_range_m']
            if not isinstance(levels, list) or len(levels) != 2:
                raise ValueError('mounting_z_range_m must be [min, max]')
            for value in levels:
                _number(value, 'mounting_z_range_m')
            if levels[0] > levels[1]:
                raise ValueError('Minimum mounting height exceeds maximum')
        for bounds in _list(constraints.get('allowed_installation_regions', []), 'allowed_installation_regions'):
            _bounds(bounds)
        if 'allowed_installation_regions' in constraints and not constraints['allowed_installation_regions']:
            raise ValueError('An enabled installation whitelist must be nonempty')
        hubs = _list(constraints.get('service_hubs', []), 'service_hubs')
        if 'service_hubs' in constraints and not hubs:
            raise ValueError('An enabled service-hub constraint must be nonempty')
        hub_names = set()
        for hub in hubs:
            _keys(hub, ('name', 'xy', 'max_horizontal_distance_m', 'capacity'), 'service hub')
            name, xy = hub.get('name'), hub.get('xy')
            if not isinstance(name, str) or not name or name in hub_names:
                raise ValueError('Service hub names must be nonempty and unique')
            hub_names.add(name)
            if not isinstance(xy, list) or len(xy) != 2:
                raise ValueError('Service hub xy must have two coordinates')
            for value in xy:
                _number(value, 'hub xy', -math.inf)
            _number(hub.get('max_horizontal_distance_m'), 'max_horizontal_distance_m')
            _integer(hub.get('capacity'), 'hub capacity')
        for key in ('max_installation_cost', 'min_xy_separation_m'):
            if key in constraints:
                _number(constraints[key], key)
        if 'max_sensors_per_pole' in constraints:
            _integer(constraints['max_sensors_per_pole'], 'max_sensors_per_pole')
        for bounds in _list(constraints.get('forbidden_regions', []), 'forbidden_regions'):
            _bounds(bounds)
        names = set()
        for region in _list(specification.get('regions', []), 'regions'):
            _keys(region, ('name', 'bounds', 'importance_factor', 'min_coverage_pct',
                          'min_overlap_ratio', 'max_overlap_ratio') +
                          (('min_single_failure_coverage_pct',) if extended else ()), 'region')
            name = region.get('name')
            if not isinstance(name, str) or not name or name in names:
                raise ValueError('Region names must be nonempty and unique')
            names.add(name)
            _bounds(region.get('bounds'))
            _number(region.get('importance_factor', 1.), 'importance_factor', lower=1.)
            _thresholds(region)
            if 'min_single_failure_coverage_pct' in region:
                _number(region['min_single_failure_coverage_pct'], 'min_single_failure_coverage_pct', upper=100.)

    def bind(self, target_points):
        return BoundCoverageRequirements(self, target_points)


class BoundCoverageRequirements:
    def __init__(self, requirements, target_points):
        self.requirements = requirements
        self.spec = deepcopy(requirements.specification)
        self.targets = np.asarray(target_points, dtype=float)
        self.needs_sensor_masks = ('min_single_failure_coverage_pct' in self.spec.get('constraints', {})
            or any('min_single_failure_coverage_pct' in r for r in self.spec.get('regions', [])))
        if (self.targets.ndim != 2 or self.targets.shape[1] != 3 or not len(self.targets)
                or not np.isfinite(self.targets).all()):
            raise ValueError('Expected nonempty finite [N,3] targets')
        self.regions = []
        self.importance = np.ones(len(self.targets))
        for region in self.spec.get('regions', []):
            mask = _inside(self.targets, region['bounds'])
            if not mask.any():
                raise ValueError(f'Region {region["name"]} has no targets')
            self.regions.append((region, mask))
            self.importance[mask] = np.maximum(self.importance[mask], region.get('importance_factor', 1.))

    def installation_costs(self, poses):
        model = self.spec.get('cost', {})
        costs = np.full(len(poses), float(model.get('base_per_sensor', 0.)))
        costs += model.get('per_height_m', 0.) * poses[:, 2]
        for region in model.get('regions', []):
            costs[_inside(poses, region['bounds'])] += region['surcharge']
        return costs

    def evaluate(self, poses, density, *, sensor_masks=None):
        poses, density = np.asarray(poses, dtype=float), np.asarray(density, dtype=float)
        if (poses.ndim != 2 or poses.shape[1] != 5 or not np.isfinite(poses).all()
                or np.any(poses[:, 2] < 0)):
            raise ValueError('Expected finite [K,5] poses with nonnegative height')
        if (density.shape != (len(self.targets),) or not np.isfinite(density).all()
                or np.any(density < 0) or np.any(density > len(poses))
                or np.any(density != np.floor(density))):
            raise ValueError('Density must contain one integer count in [0, sensor_count] per target')
        density = density.astype(np.int64)
        masks = None
        if sensor_masks is not None:
            masks = np.asarray(sensor_masks)
            if (masks.shape != (len(poses), len(self.targets)) or not np.isin(masks, [0, 1]).all()
                    or not np.array_equal(masks.sum(axis=0), density)):
                raise ValueError('Sensor masks must be binary [K,N] and sum to density')
            masks = masks.astype(bool)
        if self.needs_sensor_masks and masks is None:
            raise ValueError('Single-failure coverage requires per-sensor masks, not density alone')
        costs = self.installation_costs(poses)
        score, components = objective_from_density(density, self.importance, costs, self.requirements.weights)
        if not np.isfinite(costs).all() or not math.isfinite(score):
            raise ValueError('Cost or objective overflow; use smaller finite coefficients')
        coverage = float(100. * np.count_nonzero(density) / len(density))
        overlap = float(np.count_nonzero(density >= 2) / len(density))
        violations = []
        residuals = {}

        def check(condition, name, residual):
            if condition:
                violations.append(name)
            residuals[name] = float(max(residual, 0.)) if condition else 0.

        def thresholds(data, cov, double, prefix=''):
            min_cov = data.get('min_coverage_pct', 0.)
            min_double, max_double = data.get('min_overlap_ratio', 0.), data.get('max_overlap_ratio', 1.)
            check(cov + 1e-9 < min_cov, prefix + 'coverage', (min_cov - cov) / 100.)
            check(double + 1e-9 < min_double, prefix + 'overlap_min', min_double - double)
            check(double > max_double + 1e-9, prefix + 'overlap_max', double - max_double)

        c = self.spec.get('constraints', {})
        n, k = len(poses), self.spec['budget_k']
        duplicates = n - len(np.unique(poses, axis=0))
        check(n > k, 'sensor_budget', (n - k) / max(k, 1))
        check(duplicates > 0, 'duplicate_pose', duplicates / max(n, 1))
        budget = c.get('max_installation_cost')
        if budget is not None:
            excess_cost = components['installation_cost'] - budget
            check(excess_cost > 1e-9, 'installation_cost', excess_cost / max(budget, 1.))
        if len(poses) > 1:
            distances = np.linalg.norm(poses[:, None, :2] - poses[None, :, :2], axis=2)
            distances = distances[np.triu_indices(n, k=1)]
            minimum = c.get('min_xy_separation_m', 0.)
            deficit = np.maximum(minimum - distances, 0.) / max(minimum, 1.)
            check(float(distances.min()) + 1e-9 < minimum, 'min_xy_separation', deficit.mean())
        if len(poses):
            _, counts = np.unique(poses[:, :2], axis=0, return_counts=True)
            capacity = c.get('max_sensors_per_pole', n)
            excess = int(np.maximum(counts - capacity, 0).sum())
            check(excess > 0, 'pole_capacity', excess / n)
        forbidden = np.zeros(n, dtype=bool)
        for bounds in c.get('forbidden_regions', []):
            forbidden |= _inside(poses, bounds)
        check(forbidden.any(), 'forbidden_installation', forbidden.sum() / max(n, 1))
        if 'mounting_z_range_m' in c:
            lo, hi = c['mounting_z_range_m']
            deficit = np.maximum(lo - poses[:, 2], 0.) + np.maximum(poses[:, 2] - hi, 0.)
            check(bool(np.any(deficit > 1e-9)), 'mounting_height', float(deficit.sum()) / max(n * (hi - lo), 1.))
        if 'allowed_installation_regions' in c:
            allowed = np.zeros(n, dtype=bool)
            for bounds in c['allowed_installation_regions']:
                allowed |= _inside(poses, bounds)
            check(bool(np.any(~allowed)), 'installation_whitelist', float((~allowed).sum()) / max(n, 1))
        infrastructure = {}
        if 'service_hubs' in c:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import maximum_bipartite_matching
            hubs = c['service_hubs']
            distances = np.linalg.norm(poses[:, None, :2] - np.asarray([h['xy'] for h in hubs])[None], axis=2)
            slots = [i for i, hub in enumerate(hubs) for _ in range(min(hub['capacity'], n))]
            reachable = distances <= np.asarray([h['max_horizontal_distance_m'] for h in hubs]) + 1e-9
            assignments = maximum_bipartite_matching(csr_matrix(reachable[:, slots]), perm_type='column') if n else []
            matched = sum(int(slot >= 0) for slot in assignments)
            check(matched < n, 'service_connection', (n - matched) / max(n, 1))
            infrastructure = dict(matched_sensors=matched, required_sensors=n,
                assigned_hubs=[hubs[slots[slot]]['name'] if slot >= 0 else None for slot in assignments],
                model='capacity-constrained horizontal reachability; not cable routing or radio QoS')
        thresholds(c, coverage, overlap)
        robustness = {}

        def failure_metrics(mask):
            values = density[mask]
            # Removing sensor i loses exactly its uniquely observed targets.
            losses = (masks[:, mask] & (values == 1)).sum(axis=1)
            surviving = np.count_nonzero(values) - losses
            percentages = 100. * surviving / len(values)
            return dict(single_failure_coverage_pct=float(percentages.min()) if n else 0.,
                        coverage_after_each_failure_pct=percentages.tolist())

        if self.needs_sensor_masks:
            robustness = failure_metrics(np.ones(len(density), dtype=bool))
            minimum = c.get('min_single_failure_coverage_pct', 0.)
            actual = robustness['single_failure_coverage_pct']
            check(actual + 1e-9 < minimum, 'single_failure_coverage', (minimum - actual) / 100.)
        region_metrics = {}
        for region, mask in self.regions:
            values = density[mask]
            cov = float(100. * np.count_nonzero(values) / len(values))
            double = float(np.count_nonzero(values >= 2) / len(values))
            region_metrics[region['name']] = dict(target_count=len(values), coverage_pct=cov,
                                                 overlap_ratio=double)
            thresholds(region, cov, double, f'region:{region["name"]}:')
            if 'min_single_failure_coverage_pct' in region:
                metrics = failure_metrics(mask)
                region_metrics[region['name']].update(metrics)
                minimum, actual = region['min_single_failure_coverage_pct'], metrics['single_failure_coverage_pct']
                check(actual + 1e-9 < minimum, f'region:{region["name"]}:single_failure_coverage',
                      (minimum - actual) / 100.)
        return CoverageAssessment(score, coverage, overlap, components, region_metrics,
                                  not violations, tuple(violations), residuals, robustness, infrastructure)
