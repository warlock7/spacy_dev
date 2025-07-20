#!/usr/bin/env python3
"""
Proper test script with assertions to verify classification improvements
"""

from nlp_analyzer_refactored import NLPAnalyzer
import json


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def assert_equal(self, actual, expected, test_name):
        if actual == expected:
            self.passed += 1
            print(f"✅ {test_name}: PASS")
        else:
            self.failed += 1
            self.failures.append(f"{test_name}: Expected '{expected}', got '{actual}'")
            print(f"❌ {test_name}: FAIL - Expected '{expected}', got '{actual}'")

    def assert_contains(self, container, item, test_name):
        if item in container:
            self.passed += 1
            print(f"✅ {test_name}: PASS")
        else:
            self.failed += 1
            self.failures.append(f"{test_name}: Expected '{item}' in {container}")
            print(f"❌ {test_name}: FAIL - Expected '{item}' in {container}")

    def assert_not_contains(self, container, item, test_name):
        if item not in container:
            self.passed += 1
            print(f"✅ {test_name}: PASS")
        else:
            self.failed += 1
            self.failures.append(f"{test_name}: Expected '{item}' NOT in {container}")
            print(f"❌ {test_name}: FAIL - Expected '{item}' NOT in {container}")

    def print_summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"TEST SUMMARY: {self.passed}/{total} tests passed")
        print(f"{'='*70}")

        if self.failures:
            print("FAILURES:")
            for failure in self.failures:
                print(f"  - {failure}")
        else:
            print("🎉 ALL TESTS PASSED!")
        print(f"{'='*70}")


def verify_classification():
    """Verify the classification improvements with actual assertions"""
    analyzer = NLPAnalyzer()
    results = TestResults()

    print("=" * 70)
    print("AUTOMATED VERIFICATION OF CLASSIFICATION FIXES")
    print("=" * 70)

    # Test Case 1: dog sketch
    print(f"\n🔍 Testing: 'dog sketch'")
    result = analyzer.analyze_query("dog sketch")
    results.assert_equal(result["main_objective"], "dog", "Dog sketch - main objective")
    results.assert_contains(
        result["attributes"].get("format_types", []),
        "sketch",
        "Dog sketch - format detection",
    )
    results.assert_not_contains(
        result["attributes"], "entities", "Dog sketch - no entities"
    )

    # Test Case 2: apple sketch
    print(f"\n🔍 Testing: 'apple sketch'")
    result = analyzer.analyze_query("apple sketch")
    results.assert_equal(
        result["main_objective"], "apple", "Apple sketch - main objective"
    )
    results.assert_contains(
        result["attributes"].get("format_types", []),
        "sketch",
        "Apple sketch - format detection",
    )

    # Test Case 3: tiger minimal illustration
    print(f"\n🔍 Testing: 'tiger minimal illustration'")
    result = analyzer.analyze_query("tiger minimal illustration")
    results.assert_equal(
        result["main_objective"], "tiger", "Tiger illustration - main objective"
    )
    results.assert_contains(
        result["attributes"].get("format_types", []),
        "illustration",
        "Tiger illustration - format detection",
    )
    results.assert_contains(
        result["attributes"].get("adjectives", []),
        "minimal",
        "Tiger illustration - adjective detection",
    )

    # Test Case 4: dog pencil sketch
    print(f"\n🔍 Testing: 'dog pencil sketch'")
    result = analyzer.analyze_query("dog pencil sketch")
    results.assert_equal(
        result["main_objective"], "dog", "Dog pencil sketch - main objective"
    )
    results.assert_contains(
        result["attributes"].get("format_types", []),
        "sketch",
        "Dog pencil sketch - sketch format",
    )
    results.assert_contains(
        result["attributes"].get("format_types", []),
        "pencil",
        "Dog pencil sketch - pencil format",
    )

    # Test Case 5: cat drawing
    print(f"\n🔍 Testing: 'cat drawing'")
    result = analyzer.analyze_query("cat drawing")
    results.assert_equal(
        result["main_objective"], "cat", "Cat drawing - main objective"
    )
    results.assert_contains(
        result["attributes"].get("format_types", []),
        "drawing",
        "Cat drawing - format detection",
    )

    # Test Case 6: bird watercolor painting
    print(f"\n🔍 Testing: 'bird watercolor painting'")
    result = analyzer.analyze_query("bird watercolor painting")
    results.assert_equal(
        result["main_objective"], "bird", "Bird painting - main objective"
    )
    materials = result["attributes"].get("materials", [])
    results.assert_contains(
        materials, "watercolor", "Bird painting - watercolor material"
    )
    format_types = result["attributes"].get("format_types", [])
    results.assert_contains(format_types, "painting", "Bird painting - painting format")

    # Test Case 7: Ensure complex queries still work with entities
    print(f"\n🔍 Testing: 'large oil painting by Pablo Picasso'")
    result = analyzer.analyze_query("large oil painting by Pablo Picasso")
    results.assert_equal(
        result["main_objective"], "painting", "Complex query - main objective"
    )
    results.assert_contains(
        result["attributes"].get("sizes", []), "large", "Complex query - size detection"
    )
    entities = result["attributes"].get("entities", [])
    entity_names = [ent["text"] for ent in entities] if entities else []
    results.assert_contains(
        entity_names, "Pablo Picasso", "Complex query - entity detection"
    )

    # Test Case 8: moon shape vector
    print(f"\n🔍 Testing: 'moon shape vector'")
    result = analyzer.analyze_query("moon shape vector")
    results.assert_equal(
        result["main_objective"], "moon", "Moon shape vector - main objective"
    )
    results.assert_contains(
        result["attributes"].get("shapes", []),
        "shape",
        "Moon shape vector - shape detection",
    )
    results.assert_contains(
        result["attributes"].get("format_types", []),
        "vector",
        "Moon shape vector - format detection",
    )

    # Test Case 9: christmas tree
    print(f"\n🔍 Testing: 'christmas tree'")
    result = analyzer.analyze_query("christmas tree")
    results.assert_equal(
        result["main_objective"], "christmas", "Christmas tree - main objective"
    )
    results.assert_contains(
        result["attributes"].get("modifiers", []),
        "tree",
        "Christmas tree - 'tree' as modifier",
    )
    results.assert_not_contains(
        result["sub_objectives"], "tree", "Christmas tree - no sub-objective 'tree'"
    )
    results.assert_not_contains(
        result["attributes"], "entities", "Christmas tree - no entities"
    )

    # Test Case 10: christmas tree decoration
    print(f"\n🔍 Testing: 'christmas tree decoration'")
    result = analyzer.analyze_query("christmas tree decoration")
    results.assert_equal(
        result["main_objective"],
        "christmas",
        "Christmas tree decoration - main objective",
    )
    results.assert_contains(
        result["attributes"].get("modifiers", []),
        "tree",
        "Christmas tree decoration - 'tree' as modifier",
    )
    results.assert_contains(
        result["attributes"].get("format_types", []),
        "decoration",
        "Christmas tree decoration - 'decoration' as format type",
    )
    results.assert_not_contains(
        result["sub_objectives"],
        "tree",
        "Christmas tree decoration - no sub-objective 'tree'",
    )
    results.assert_not_contains(
        result["sub_objectives"],
        "decoration",
        "Christmas tree decoration - no sub-objective 'decoration'",
    )
    results.assert_not_contains(
        result["attributes"], "entities", "Christmas tree decoration - no entities"
    )

    results.print_summary()
    return results.failed == 0


if __name__ == "__main__":
    success = verify_classification()
    exit(0 if success else 1)
