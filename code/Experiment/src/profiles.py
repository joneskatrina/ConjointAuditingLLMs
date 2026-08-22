"""
src/profiles.py

Functions that render two sampled profiles into a string for the LLM prompt.
Add new render functions here and register them in src/config.py.

TO DO: check in notebook if rendering works corecctly!

"""

import random


# This will be set once the config is loaded
FIELD_LABELS: list[tuple[str, str]] = []

def set_field_labels_from_attributes(attributes: dict[str, list[str]]):
    global FIELD_LABELS
    FIELD_LABELS = [
        (key.replace("_", " ").capitalize(), key)
        for key in attributes.keys()
    ]

def render_profiles_markdown(profile_a: dict, profile_b: dict) -> str:
    fields = FIELD_LABELS.copy()
    random.shuffle(fields)  # optional random order

    header    = "| Attribute | Candidate A | Candidate B |"
    separator = "|-----------|-------------|-------------|"
    rows = [
        f"| {label} | {profile_a[key]} | {profile_b[key]} |"
        for label, key in fields
    ]
    #return "\n".join([header, separator] + rows)
    table = "\n".join([header, separator] + rows)

    '''
    print(f"\n{'─' * 60}")
    print(table)
    print('─' * 60)
    '''
    return table

def render_profiles_list(profile_a: dict, profile_b: dict) -> str:
    """
    Render two profiles as bullet lists with the SAME randomized attribute order.
    """

    # Shuffle field order once
    fields = FIELD_LABELS.copy()
    random.shuffle(fields)

    def render_one(label: str, profile: dict) -> str:
        lines = [
            f"- {human}: {profile[key]}"
            for human, key in fields
        ]
        return f"Candidate {label}:\n" + "\n".join(lines)

    #return render_one("A", profile_a) + "\n\n" + render_one("B", profile_b)
    result = render_one("A", profile_a) + "\n\n" + render_one("B", profile_b)

    '''
    print(f"\n{'─' * 60}")
    print(result)
    print('─' * 60)
    '''
    return result


def render_profiles_flowtext(profile_a: dict, profile_b: dict) -> str:
    """
    Render two profiles as natural flowing prose.
    """
    fields = FIELD_LABELS.copy()
    random.shuffle(fields)

    def render_one(label: str, profile: dict) -> str:
        parts = [f"{human}: {profile[key]}" for human, key in fields]
        return f"Candidate {label} — " + ", ".join(parts) + "."

    result = render_one("A", profile_a) + "\n\n" + render_one("B", profile_b)

    '''
    print(f"\n{'─' * 60}")
    print(result)
    print('─' * 60)
    '''
    return result

def render_single_profile_markdown(profile: dict) -> str:
    fields = FIELD_LABELS.copy()
    random.shuffle(fields)
    header = "| Attribute | Candidate |"
    separator = "|-----------|-----------|"
    rows = [f"| {label} | {profile[key]} |" for label, key in fields]
    return "\n".join([header, separator] + rows)

def render_single_profile_list(profile: dict) -> str:
    fields = FIELD_LABELS.copy()
    random.shuffle(fields)
    rows = [f"- {label}: {profile[key]}" for label, key in fields]
    return "Candidate profile:\n" + "\n".join(rows)

def render_single_profile_flowtext(profile: dict) -> str:
    fields = FIELD_LABELS.copy()
    random.shuffle(fields)
    parts = [f"{label}: {profile[key]}" for label, key in fields]
    return "Candidate profile — " + ", ".join(parts) + "."
