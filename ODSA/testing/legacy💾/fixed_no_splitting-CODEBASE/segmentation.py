import numpy as np
from scipy.ndimage import uniform_filter1d
from config import SMOOTHING_LENGTH, GRADIENT_THRESHOLD


def detect_data_gaps(distance, gap_threshold=2000):
    steps = np.diff(distance)
    gap_indices = np.where(steps > gap_threshold)[0]
    gap_mask = np.zeros(len(distance), dtype=bool)
    gap_mask[gap_indices] = True
    gap_mask[gap_indices + 1] = True
    return gap_mask


def split_into_segments(datafile, distance, gap_threshold=2000, min_segment_length=50, min_segment_km=10):
    steps = np.diff(distance)
    gap_indices = np.where(steps > gap_threshold)[0]

    split_points = [0]
    for gap_idx in gap_indices:
        split_points.append(gap_idx + 1)
        split_points.append(gap_idx + 1)
    split_points.append(len(distance))

    segments = []
    for i in range(0, len(split_points) - 1, 2):
        start = split_points[i]
        end = split_points[i + 1]
        length_km = (distance[end-1] - distance[start]) / 1000
        if end - start >= min_segment_length and length_km >= min_segment_km:
            print(f"    > Segment {len(segments)+1}: Rows {start} to {end} ({end-start} points), Length: {length_km:.2f} km")
            segments.append((datafile.iloc[start:end].copy(), distance[start:end]))

    return segments


def split_by_landscape(segment_data, segment_distance, smoothing_length=SMOOTHING_LENGTH,
                       gradient_threshold=GRADIENT_THRESHOLD,
                       min_segment_km=10, min_segment_pts=50):
    elev = segment_data['bedrock_altitude (m)'].values
    dist = np.asarray(segment_distance, dtype=float)

    if len(dist) < 2:
        return [(segment_data, segment_distance, False)]

    # Distance repeats where positions are duplicated. np.gradient below divides by
    # dist/1000 (km), so a repeated point divides by ~0 and yields a spurious ~1e5
    # m/km spike. Build a strictly increasing copy for the gradient only, nudging by
    # the median real sample spacing. Everything else (zone extents, length gates,
    # the returned arrays) uses the measured distance, so none is invented.
    _diffs = np.diff(dist)
    _pos = _diffs[_diffs > 0]
    min_step = float(np.median(_pos)) if _pos.size else 15.0
    grad_dist = dist.copy()
    for i in range(1, len(grad_dist)):
        if grad_dist[i] <= grad_dist[i - 1]:
            grad_dist[i] = grad_dist[i - 1] + min_step

    kernel_pts = int(smoothing_length / min_step)
    kernel_pts = max(3, kernel_pts if kernel_pts % 2 == 1 else kernel_pts + 1)

    smoothed = uniform_filter1d(elev, size=kernel_pts, mode='nearest')
    grad = np.gradient(smoothed, grad_dist / 1000)
    in_transition = np.abs(grad) > gradient_threshold

    if not np.any(in_transition):
        return [(segment_data, segment_distance, False)]

    changes = np.diff(in_transition.astype(int))
    t_starts = np.where(changes == 1)[0] + 1
    t_ends = np.where(changes == -1)[0] + 1

    if in_transition[0]:
        t_starts = np.concatenate([[0], t_starts])
    if in_transition[-1]:
        t_ends = np.concatenate([t_ends, [len(in_transition)]])

    merge_gap_km = 5 # [2, 5*, 10]km for sensitivity testing
    merged_starts, merged_ends = [t_starts[0]], [t_ends[0]]
    for s, e in zip(t_starts[1:], t_ends[1:]):
        gap_km = (dist[s] - dist[merged_ends[-1]]) / 1000
        if gap_km < merge_gap_km:
            merged_ends[-1] = e
        else:
            merged_starts.append(s)
            merged_ends.append(e)

    transition_set = set()
    for s, e in zip(merged_starts, merged_ends):
        peak_grad_idx = s + np.argmax(np.abs(grad[s:e]))
        print(f"      transition zone km {dist[s]/1000:.1f}-{dist[min(e,len(dist)-1)]/1000:.1f}, "
              f"peak gradient = {grad[peak_grad_idx]:.1f} m/km")
        transition_set.add((s, e))

    boundaries = sorted({0, len(dist)} | {s for s, _ in transition_set} | {e for _, e in transition_set})

    sub_segments = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e <= s:
            continue
        is_trans = (s, e) in transition_set
        length_km = (dist[e - 1] - dist[s]) / 1000
        if e - s >= min_segment_pts and length_km >= min_segment_km:
            sub_segments.append((segment_data.iloc[s:e].copy(), dist[s:e], is_trans))

    if not sub_segments:
        return [(segment_data, segment_distance, False)]

    return sub_segments
