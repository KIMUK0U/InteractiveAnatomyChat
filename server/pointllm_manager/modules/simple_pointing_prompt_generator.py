"""
simple_pointing_prompt_generator.py - Simple Pointing Region Identification

Generates simple Q&A pairs for basic region identification tasks.
Designed to teach the model to understand <point_patch> tokens and respond
with direct anatomical labels.

Examples:
    Q: "What is this region?"
    A: "This is FDI 11: Upper right 1st tooth from midline (Central incisor)"
    
    Q: "What region does this index finger point to?"
    A: "This is Right Upper Posterior of Maxilla (Class: Maxilla & Upper Skull)"
"""

import random
from typing import Tuple, Optional


class SimplePointingPromptGenerator:
    """Generates simple pointing region identification prompts."""
    
    # Question templates for simple region identification
    SIMPLE_QUESTIONS = [
        "What is this region?",
        "What region is this?",
        "What anatomical structure is this?",
        "Identify this region.",
        "What structure is this?",
        "What is this anatomical region?",
    ]
    
    # Question templates that mention finger/pointing
    FINGER_POINTING_QUESTIONS = [
        "What region does this index finger point to?",
        "What region am I pointing at?",
        "Which structure am I pointing at?",
        "What region is being pointed at?",
        "What anatomical structure is being indicated?",
        "Identify the region I'm pointing to.",
    ]
    
    # Answer templates for subclass (with full details)
    SUBCLASS_ANSWER_TEMPLATES = [
        "This is {subclass_name}",
        "This region is {subclass_name}",
        "The anatomical structure is {subclass_name}",
        "This is the {subclass_name}",
    ]
    
    # Answer templates when class context is important
    CLASS_AWARE_ANSWER_TEMPLATES = [
        "This is {subclass_name} (Class: {class_name})",
        "This region is {subclass_name}, which belongs to {class_name}",
        "The structure is {subclass_name} from the {class_name} region",
    ]
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
    
    def generate(
        self,
        primary_region,
        include_class_info: bool = False,
        variation_index: int = 0,
    ) -> Tuple[str, str]:
        """
        Generate a simple pointing Q&A pair.
        
        Args:
            primary_region: RegionOfInterest object for the main target
            include_class_info: Whether to include class name in answer
            variation_index: Index for deterministic variation selection
        
        Returns:
            Tuple of (question, answer)
        """
        # Select question type based on variation index
        all_questions = self.SIMPLE_QUESTIONS + self.FINGER_POINTING_QUESTIONS
        question = all_questions[variation_index % len(all_questions)]
        
        # Generate answer
        subclass_name = primary_region.anatomical_subclass_name
        class_name = primary_region.anatomical_class_name
        
        if include_class_info:
            # Use class-aware template
            templates = self.CLASS_AWARE_ANSWER_TEMPLATES
            answer_template = templates[variation_index % len(templates)]
            answer = answer_template.format(
                subclass_name=subclass_name,
                class_name=class_name
            )
        else:
            # Use simple template
            templates = self.SUBCLASS_ANSWER_TEMPLATES
            answer_template = templates[variation_index % len(templates)]
            answer = answer_template.format(subclass_name=subclass_name)
        
        return question, answer
    
    def generate_batch(
        self,
        primary_region,
        num_variations: int = 4,
        include_class_info: bool = False,
    ) -> list[Tuple[str, str]]:
        """
        Generate multiple simple pointing Q&A pairs.
        
        Args:
            primary_region: RegionOfInterest object for the main target
            num_variations: Number of variations to generate
            include_class_info: Whether to include class name in answers
        
        Returns:
            List of (question, answer) tuples
        """
        results = []
        for i in range(num_variations):
            qa_pair = self.generate(
                primary_region=primary_region,
                include_class_info=include_class_info,
                variation_index=i
            )
            results.append(qa_pair)
        return results


if __name__ == "__main__":
    # Test the generator
    from dataclasses import dataclass
    
    @dataclass
    class MockRegion:
        """Mock region for testing."""
        anatomical_subclass_name: str = "FDI 11: Upper right 1st tooth from midline (Central incisor)"
        anatomical_class_name: str = "Upper Teeth"
    
    generator = SimplePointingPromptGenerator(seed=42)
    test_region = MockRegion()
    
    print("=" * 60)
    print("Simple Pointing Prompt Generator - Test")
    print("=" * 60)
    
    print("\nSimple answers (without class):")
    print("-" * 60)
    for i, (q, a) in enumerate(generator.generate_batch(test_region, num_variations=6)):
        print(f"\nVariation {i}:")
        print(f"Q: {q}")
        print(f"A: {a}")
    
    print("\n\nClass-aware answers:")
    print("-" * 60)
    for i, (q, a) in enumerate(generator.generate_batch(test_region, num_variations=6, include_class_info=True)):
        print(f"\nVariation {i}:")
        print(f"Q: {q}")
        print(f"A: {a}")
