from wallball.audio import FLOOR, POP, THUD, Onset
from wallball.events import ImpactSequencer


def on(t, label, p=0.9):
    other = (1 - p) / 2
    scores = {POP: other, THUD: other, FLOOR: other}
    scores[label] = p
    return Onset(t, 10.0, 0.0, (0, 0, 0), label, scores)


def run(seq, onsets):
    done = []
    for o in onsets:
        done += seq.push(o)
    done += seq.flush()
    return done


def test_rally_pairs_pop_and_thud():
    seq = ImpactSequencer(distance_m=5.0)
    onsets = []
    t = 1.0
    for _ in range(5):
        onsets += [on(t, POP), on(t + 0.28, THUD)]
        t += 1.3
    done = run(seq, onsets)
    assert len(done) == 5
    assert all(s.t_thud is not None and abs(s.flight_s - 0.28) < 1e-9 for s in done)
    assert [s.idx for s in done] == [1, 2, 3, 4, 5]


def test_volley_rhythm_short_distance():
    seq = ImpactSequencer(distance_m=1.8)
    onsets = []
    t = 0.5
    for _ in range(8):
        onsets += [on(t, POP), on(t + 0.1, THUD)]
        t += 0.2
    done = run(seq, onsets)
    assert len(done) == 8 and all(s.t_thud is not None for s in done)


def test_bounce_before_wall_and_return_bounce():
    seq = ImpactSequencer(distance_m=4.0)
    onsets = [on(1.0, POP), on(1.2, FLOOR), on(1.45, THUD),  # bounced before the wall
              on(2.4, POP), on(2.7, THUD), on(3.1, FLOOR),   # normal, then bounce on the way back
              on(3.9, POP), on(4.2, THUD)]
    done = run(seq, onsets)
    assert len(done) == 3
    assert done[0].bounce_before_wall and done[0].t_thud == 1.45
    assert not done[1].bounce_before_wall and done[1].bounces_after == 1
    assert not done[2].bounce_before_wall


def test_missed_thud_and_missed_pop():
    seq = ImpactSequencer(distance_m=4.0)
    onsets = [on(1.0, POP), on(2.2, POP), on(2.5, THUD),  # first stroke never reached the wall
              on(4.0, THUD)]                              # thud whose pop was not heard: no stroke invented
    done = run(seq, onsets)
    assert len(done) == 2
    assert done[0].t_thud is None and done[1].t_thud == 2.5
    assert seq.count == 2


def test_echo_after_pop_is_ignored():
    seq = ImpactSequencer(distance_m=5.0)
    onsets = [on(1.0, POP), on(1.04, POP, 0.6), on(1.3, THUD), on(2.5, POP), on(2.55, POP, 0.5), on(2.8, THUD)]
    done = run(seq, onsets)
    assert len(done) == 2 and all(s.t_thud is not None for s in done)


def test_uncertain_labels_resolved_by_timing():
    seq = ImpactSequencer(distance_m=4.0)
    onsets = [on(1.0, POP), on(1.3, THUD, 0.4), on(2.5, POP, 0.45), on(2.8, THUD, 0.4)]
    done = run(seq, onsets)
    assert len(done) == 2 and all(s.t_thud is not None for s in done)


def test_stale_pop_closes_as_miss():
    seq = ImpactSequencer(distance_m=4.0)
    onsets = [on(1.0, POP), on(3.5, POP), on(3.8, THUD)]
    done = run(seq, onsets)
    assert len(done) == 2 and done[0].t_thud is None
