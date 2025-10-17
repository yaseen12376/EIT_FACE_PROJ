#!/usr/bin/env python3
"""Fix entry_type in results"""

with open('cctv_attendance_system.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace patterns
content = content.replace(
    '"bbox": scaled_bbox,\n                                        "status": status\n                                    })',
    '"bbox": scaled_bbox,\n                                        "status": status,\n                                        "entry_type": entry_type\n                                    })'
)

content = content.replace(
    '"bbox": scaled_bbox,\n                                        "status": "unknown"\n                                    })',
    '"bbox": scaled_bbox,\n                                        "status": "unknown",\n                                        "entry_type": entry_type\n                                    })'
)

with open('cctv_attendance_system.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed entry_type in all results")
