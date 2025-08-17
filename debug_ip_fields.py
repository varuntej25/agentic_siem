#!/usr/bin/env python3
"""Check what IP fields exist in the data for IP 192.168.1.104"""

import json
import sys

print("=== Checking IP Fields in Data ===")

# Load the data.json file
print("Loading data.json...")
with open('data.json', 'r') as f:
    content = f.read()

# Find the specific event for alert 15
target_event_id = "c0e932ba-10d3-44c6-a013-b0305644e608"
target_ip = "192.168.1.104"

print(f"Looking for event ID: {target_event_id}")
print(f"Looking for IP: {target_ip}")

lines = content.strip().split('\n')
found_event = None

for line_num, line in enumerate(lines):
    try:
        entry = json.loads(line)
        if entry.get('_id') == target_event_id:
            found_event = entry
            print(f"✅ Found target event at line {line_num + 1}")
            break
    except json.JSONDecodeError:
        continue

if found_event:
    print("\n📋 Event details:")
    print(f"   _id: {found_event.get('_id')}")
    print(f"   _index: {found_event.get('_index')}")
    
    source = found_event.get('_source', {})
    print(f"\n🔍 All fields in _source:")
    for field, value in source.items():
        print(f"   {field}: {value}")
    
    print(f"\n🎯 Searching for IP {target_ip} in all fields:")
    ip_fields = []
    for field, value in source.items():
        if target_ip in str(value):
            ip_fields.append((field, value))
            print(f"   ✅ FOUND in {field}: {value}")
    
    if not ip_fields:
        print(f"   ❌ IP {target_ip} not found in any field")
        print(f"   💡 Available IP-like fields:")
        for field, value in source.items():
            if any(keyword in field.lower() for keyword in ['ip', 'addr', 'host', 'src', 'dest', 'client', 'server']):
                print(f"      {field}: {value}")
    
    print(f"\n🔎 IP fields we're searching for:")
    search_fields = ['srcIP', 'src_ip', 'sourceIP', 'source_ip', 'destIP', 'dest_ip', 
                    'destinationIP', 'destination_ip', 'clientIP', 'client_ip', 
                    'serverIP', 'server_ip']
    for field in search_fields:
        value = source.get(field)
        if value:
            print(f"   ✅ {field}: {value}")
        else:
            print(f"   ❌ {field}: not found")

else:
    print(f"❌ Target event {target_event_id} not found in data.json")

print(f"\n🔍 Let's also search for any entries containing IP {target_ip}:")
matching_entries = 0
sample_matches = []

for line_num, line in enumerate(lines[:1000]):  # Check first 1000 entries
    try:
        entry = json.loads(line)
        source = entry.get('_source', {})
        entry_str = json.dumps(source).lower()
        if target_ip in entry_str:
            matching_entries += 1
            if len(sample_matches) < 3:
                sample_matches.append((line_num + 1, entry.get('_id', 'unknown'), source))
    except json.JSONDecodeError:
        continue

print(f"Found {matching_entries} entries containing IP {target_ip} in first 1000 entries")

if sample_matches:
    print(f"\n📋 Sample matches:")
    for line_num, entry_id, source in sample_matches:
        print(f"   Line {line_num}, ID: {entry_id}")
        for field, value in source.items():
            if target_ip in str(value):
                print(f"      {field}: {value}")
        print()
