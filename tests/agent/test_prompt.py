from datetime import date

from wanderlust.agent.prompt import NUDGE, SUBMIT_TOOL, request_message, system_prompt


def test_system_prompt_states_today_and_the_rules():
    prompt = system_prompt(date(2026, 11, 13))
    assert "Today is Friday 13 November 2026." in prompt
    assert f"call {SUBMIT_TOOL} exactly once" in prompt
    assert "cite every stop" in prompt
    assert "{" not in prompt  # every placeholder was filled


def test_request_message_wraps_the_request():
    assert request_message("  4 days in Lisbon \n") == (
        "<trip_request>\n4 days in Lisbon\n</trip_request>"
    )


def test_nudge_names_the_submit_tool():
    assert SUBMIT_TOOL in NUDGE
