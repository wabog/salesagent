from sales_agent.adapters.crm_notion import NotionCRMAdapter


def test_pick_active_page_skips_trashed_results():
    trashed = {"id": "lead-trashed", "archived": True, "in_trash": True}
    active = {"id": "lead-active", "archived": False, "in_trash": False}

    selected = NotionCRMAdapter._pick_active_page([trashed, active])  # noqa: SLF001

    assert selected == active


def test_pick_active_page_returns_none_when_all_results_are_trashed():
    selected = NotionCRMAdapter._pick_active_page(  # noqa: SLF001
        [
            {"id": "lead-1", "archived": True, "in_trash": True},
            {"id": "lead-2", "archived": False, "in_trash": True},
        ]
    )

    assert selected is None
