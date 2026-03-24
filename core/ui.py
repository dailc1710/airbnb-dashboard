from __future__ import annotations

import json

import streamlit.components.v1 as components

from core.i18n import nav_label


def inject_page_navigation(target_page: str) -> None:
    target_label = nav_label(target_page)
    components.html(
        f"""
        <script>
        const targetLabel = {json.dumps(target_label)};
        const clickSidebarOption = () => {{
            const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) {{
                return false;
            }}

            const candidates = Array.from(sidebar.querySelectorAll('label, label *, [role="radiogroup"] *'));
            const match = candidates.find((node) => (node.textContent || '').trim() === targetLabel);
            const clickable = match ? (match.closest('label') || match) : null;
            if (!clickable) {{
                return false;
            }}

            clickable.click();
            return true;
        }};

        if (!clickSidebarOption()) {{
            let attempts = 0;
            const timer = window.setInterval(() => {{
                attempts += 1;
                if (clickSidebarOption() || attempts >= 20) {{
                    window.clearInterval(timer);
                }}
            }}, 150);
        }}
        </script>
        """,
        height=0,
    )
