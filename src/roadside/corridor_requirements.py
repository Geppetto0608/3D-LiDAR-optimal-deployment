"""Road-coordinate constraints composed with the shared coverage evaluator."""
from copy import deepcopy
from dataclasses import replace

import numpy as np

from src.core.coverage_requirements import BoundCoverageRequirements, CoverageRequirements


def validate_profile(profile, length_m):
    if (not isinstance(profile, dict)
            or set(profile) != {'name', 'interface', 'forbidden_installation'}
            or not isinstance(profile['name'], str) or not profile['name']):
        raise ValueError('Expected named corridor profile with interface and forbidden_installation')
    thresholds = profile['interface']
    if not isinstance(thresholds, dict) or set(thresholds) - {
            'min_coverage_pct', 'min_overlap_ratio', 'max_overlap_ratio'}:
        raise ValueError('Unsupported interface threshold')
    CoverageRequirements(dict(schema='constrained_coverage_v2', budget_k=2, constraints=thresholds))
    if not isinstance(profile['forbidden_installation'], list):
        raise ValueError('forbidden_installation must be an array')
    for zone in profile['forbidden_installation']:
        if (not isinstance(zone, dict) or set(zone) != {'station_range_m', 'side'}
                or zone['side'] not in ('left', 'right', 'both')):
            raise ValueError('Forbidden corridor zone requires station_range_m and left/right/both side')
        bounds = zone['station_range_m']
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in bounds)
                or not np.isfinite(bounds).all() or not 0 <= bounds[0] < bounds[1] <= length_m):
            raise ValueError('Forbidden station range must fit inside the corridor')
    return deepcopy(profile)


class CorridorRequirements(BoundCoverageRequirements):
    """Discrete pose lookup avoids confusing road stations with world XY ROIs."""
    def __init__(self, specification, targets, target_coordinates, canonical, poses,
                 profile, *, length_m, interface_half_length_m):
        super().__init__(CoverageRequirements(specification), targets)
        self.profile = validate_profile(profile, length_m)
        self.canonical = np.asarray(canonical, dtype=float)
        self.poses = np.asarray(poses, dtype=float)
        coordinates = np.asarray(target_coordinates, dtype=float)
        if (self.poses.shape != self.canonical.shape or self.poses.ndim != 2 or self.poses.shape[1] != 5
                or coordinates.shape != self.targets.shape or not np.isfinite(coordinates).all()
                or not np.isfinite(self.canonical).all() or not np.isfinite(self.poses).all()):
            raise ValueError('Expected matching canonical/world candidates and target coordinates')
        self.lookup = {tuple(pose): i for i, pose in enumerate(self.poses)}
        if len(self.lookup) != len(self.poses):
            raise ValueError('Corridor candidates must have unique world poses')
        self.interface_mask = np.abs(coordinates[:, 0] - length_m/2) <= interface_half_length_m
        if not self.interface_mask.any() or self.interface_mask.all():
            raise ValueError('Interface must be a nonempty proper target subset')
        self.allowed = np.ones(len(poses), dtype=bool)
        for zone in self.profile['forbidden_installation']:
            lo, hi = zone['station_range_m']
            inside = (self.canonical[:, 0] >= lo) & (self.canonical[:, 0] <= hi)
            if zone['side'] != 'both':
                inside &= self.canonical[:, 1] > 0 if zone['side'] == 'left' else self.canonical[:, 1] < 0
            self.allowed &= ~inside

    def eligible_ids(self, ids):
        return [int(i) for i in ids if self.allowed[int(i)]]

    def filter_visibility_pairs(self, first, others):
        """Necessary/exact interface tests for the fixed-two-sensor selector."""
        thresholds = self.profile['interface']
        a, b = first[self.interface_mask], others[:, self.interface_mask]
        covered = 100 * np.mean(b | a, axis=1)
        double = np.mean(b & a, axis=1)
        return ((covered + 1e-9 >= thresholds.get('min_coverage_pct', 0))
                & (double + 1e-9 >= thresholds.get('min_overlap_ratio', 0))
                & (double <= thresholds.get('max_overlap_ratio', 1) + 1e-9))

    def evaluate(self, poses, density, *, sensor_masks=None):
        assessment = super().evaluate(poses, density, sensor_masks=sensor_masks)
        try:
            ids = [self.lookup[tuple(pose)] for pose in np.asarray(poses, dtype=float)]
        except KeyError as exc:
            raise ValueError('Corridor assessment requires poses from its declared discrete grid') from exc
        values = np.asarray(density)[self.interface_mask]
        coverage, overlap = 100*float(np.mean(values > 0)), float(np.mean(values >= 2))
        regions = {**assessment.regions, 'interface': dict(target_count=len(values),
            coverage_pct=coverage, overlap_ratio=overlap)}
        violations, residuals = list(assessment.violations), dict(assessment.constraint_residuals)

        def check(name, deficit):
            if deficit > 1e-9:
                violations.append(name)
            residuals[name] = float(max(0., deficit)) if deficit > 1e-9 else 0.

        threshold = self.profile['interface']
        # Coverage tolerance is applied in percent units, like the shared evaluator.
        gap = threshold.get('min_coverage_pct', 0) - coverage
        if gap > 1e-9:
            violations.append('region:interface:coverage')
        residuals['region:interface:coverage'] = max(0., gap/100) if gap > 1e-9 else 0.
        check('region:interface:overlap_min', threshold.get('min_overlap_ratio', 0) - overlap)
        check('region:interface:overlap_max', overlap - threshold.get('max_overlap_ratio', 1))
        forbidden = int(np.count_nonzero(~self.allowed[ids]))
        check('forbidden_corridor_installation', forbidden/max(len(ids), 1))
        return replace(assessment, regions=regions, feasible=not violations,
                       violations=tuple(violations), constraint_residuals=residuals)
