#!/usr/bin/env python3
"""Test query matching logic"""

import json

# Load first data entry
with open('data.json', 'r') as f:
    line = f.readline()
    entry = json.loads(line)
    source = entry['_source']

print("Sample data entry:")
print(f"eventID: {source.get('eventID')} (type: {type(source.get('eventID'))})")
print(f"targetUsername: {source.get('targetUsername')}")
print(f"eventType: {source.get('eventType')}")
print()

# Test the query structure we're using
test_query = {
    "bool": {
        "filter": [
            {
                "bool": {
                    "should": [
                        {"terms": {"eventID": [4740, 4647, 4648, 4624, 4625, 4720, 4722, 4725, 4634]}},
                        {"terms": {"eventId": [4740, 4647, 4648, 4624, 4625, 4720, 4722, 4725, 4634]}}
                    ],
                    "minimum_should_match": 1
                }
            },
            {
                "bool": {
                    "should": [
                        {"term": {"targetUsername.keyword": "jdoe"}},
                        {"term": {"targetUsername": "jdoe"}},
                        {"term": {"username.keyword": "jdoe"}},
                        {"term": {"username": "jdoe"}}
                    ],
                    "minimum_should_match": 1
                }
            }
        ]
    }
}

def _matches_terms_query(source, terms_query):
    """Test terms query logic"""
    for field, values in terms_query.items():
        source_value = source.get(field)
        print(f"  Checking field '{field}': source='{source_value}', values={values}")
        if source_value and source_value in values:
            print(f"    MATCH found!")
            return True
        else:
            print(f"    No match")
    return False

def _matches_term_query(source, term_query):
    """Test term query logic"""
    for field, value in term_query.items():
        source_value = source.get(field)
        print(f"  Checking field '{field}': source='{source_value}', expected='{value}'")
        if source_value == value:
            print(f"    MATCH found!")
            return True
        else:
            print(f"    No match")
    return False

def _matches_bool_query(source, bool_query):
    """Test bool query logic"""
    print("Testing bool query...")
    
    # Handle 'should' clauses (OR logic)
    if 'should' in bool_query:
        print("  Found 'should' clauses")
        should_clauses = bool_query['should']
        for i, clause in enumerate(should_clauses):
            print(f"  Testing should clause {i}: {list(clause.keys())}")
            if 'terms' in clause:
                if _matches_terms_query(source, clause['terms']):
                    return True
            elif 'term' in clause:
                if _matches_term_query(source, clause['term']):
                    return True
        return False
    
    # Handle 'filter' clauses
    if 'filter' in bool_query:
        print("  Found 'filter' clauses")
        filter_clauses = bool_query['filter']
        if isinstance(filter_clauses, list):
            for i, clause in enumerate(filter_clauses):
                print(f"  Testing filter clause {i}: {list(clause.keys())}")
                if 'bool' in clause:
                    if not _matches_bool_query(source, clause['bool']):
                        return False
        return True
    
    return True

# Test the query
print("Testing query matching...")
print("Query structure:", json.dumps(test_query, indent=2))
print()

result = _matches_bool_query(source, test_query['bool'])
print(f"\nFinal result: {result}")
