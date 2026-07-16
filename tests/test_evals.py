from evals.run_evals import MockJudge, metrics_table, render_report, run_evals


async def test_eval_runner_smoke_on_three_goldens():
    summary = await run_evals(limit=3, judge=MockJudge())
    assert len(summary.items) == 3
    assert summary.documents == 4  # bundled corpus
    assert summary.chunks >= summary.documents

    # metrics land in sane ranges on the bundled corpus
    assert 0.0 <= summary.hit_rate <= 1.0
    assert summary.hit_rate >= 2 / 3
    assert summary.citation_presence >= 2 / 3
    assert 1.0 <= summary.avg_judge_score <= 5.0

    report = render_report(summary)
    assert "hit_rate" in report and "citation_presence" in report
    assert "hit_rate" in metrics_table(summary)


async def test_mock_judge_monotonic_in_overlap():
    judge = MockJudge()
    reference = "pgvector uses the cosine distance operator with vector_cosine_ops"
    full = await judge.score("q", reference, reference)
    partial = await judge.score(
        "q", reference, "pgvector supports cosine distance operators"
    )
    none = await judge.score("q", reference, "espresso crema depends on fresh beans")
    assert full == 5
    assert full >= partial >= none
    assert none == 1
    assert all(1 <= s <= 5 for s in (full, partial, none))
