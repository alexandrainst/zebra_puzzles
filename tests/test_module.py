"""Tests for the `module` module."""

from zebra_puzzles.pipeline import run_pipeline


# An end-to-end test of build_dataset
def test_pipeline() -> None:
    """Test build_dataset."""
    # Define the attributes
    attributes = {
        "jobs": {
            "baker": ["the baker", "is a baker", "is not a baker"],
            "minister": ["the minister", "is a minister", "is not a minister"],
            "police officer": [
                "the police officer",
                "is a police officer",
                "is not a police officer",
            ],
            "nurse": ["the nurse", "is a nurse", "is not a nurse"],
        },
        "pets": {
            "cat": ["the cat owner", "has a cat", "does not have a cat"],
            "dog": ["the dog owner", "has a dog", "does not have a dog"],
            "rabbit": ["the rabbit owner", "has a rabbit", "does not have a rabbit"],
            "zebra": ["the zebra owner", "has a zebra", "does not have a zebra"],
        },
        "drinks": {
            "juice": ["the juice drinker", "drinks juice", "does not drink juice"],
            "coffee": ["the coffee drinker", "drinks coffee", "does not drink coffee"],
            "milk": ["the milk drinker", "drinks milk", "does not drink milk"],
            "smoothie": [
                "the smoothie drinker",
                "drinks smoothie",
                "does not drink smoothie",
            ],
        },
    }

    n_objects = 4
    n_attributes = 3
    n_red_herring_clues = 2

    prompt_templates = [
        "Test prompt template, \nattributes: \n{chosen_attributes_str} \nhints: \n{chosen_clues_str}",
        "Test prompt 2",
    ]
    prompt_and = "and"
    clues_dict = {"found_at": "{attribute_desc} lives in house no. {i_object}."}

    n_red_herring_clues = 2
    red_herring_clues_dict = {
        "same_herring": "{attribute_desc} {attribute_desc_herring}.",
        "fact": "{fact}.",
    }
    red_herring_attributes = {
        "red_hair": ["the person with red hair", "has red hair"],
        "glasses": ["the person with glasses", "wears glasses"],
    }
    red_herring_facts = {
        "herring": "herrings are fish",
        "solar_system": "the solar system moves at a speed of about 200 km/s around the center of the galaxy",
    }

    prompt, solution_str = run_pipeline(
        n_objects=n_objects,
        n_attributes=n_attributes,
        attributes=attributes,
        clues_dict=clues_dict,
        prompt_templates=prompt_templates,
        prompt_and=prompt_and,
        n_red_herring_clues=n_red_herring_clues,
        red_herring_clues_dict=red_herring_clues_dict,
        red_herring_attributes=red_herring_attributes,
        red_herring_facts=red_herring_facts,
        verbose=False,
        eval=False,
    )

    assert len(prompt) > len(prompt_templates[0])
    assert len(solution_str) > 0
    assert isinstance(prompt, str)
    assert isinstance(solution_str, str)
