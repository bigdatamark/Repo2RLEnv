# [MODULE DOC] CLI test for 'repo2rlenv campaign init/status': re-initializing must never reset an
# existing budget.
from __future__ import annotations

import json

from repo2rlenv.cli import main


def test_campaign_initialization_never_resets_an_existing_budget(tmp_path, capsys):
    # [FUNCTION DOC] Runs cli.main for 'campaign init' with $7, then again with $9 (expects
    # exit 2 and an
    # 'implicit budget reset' JSON error), then 'campaign status' showing the $7 limit.
    # Inputs: tmp_path, capsys (reads JSON stdout). Side effects: creates ledger in tmp_path.
    assert main(["campaign", "init", str(tmp_path), "--budget-usd", "7", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["remaining_usd"] == "7.000000"
    assert main(["campaign", "init", str(tmp_path), "--budget-usd", "9", "--json"]) == 2
    assert "implicit budget reset" in json.loads(capsys.readouterr().out)["message"]
    assert main(["campaign", "status", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["limit_usd"] == "7.000000"
