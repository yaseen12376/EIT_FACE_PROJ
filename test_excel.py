#!/usr/bin/env python3
"""
Quick test to verify Excel functionality is working
"""

try:
    import pandas as pd
    import openpyxl
    from datetime import datetime
    
    print("✅ All Excel modules imported successfully!")
    
    # Test creating a simple Excel file
    test_data = {
        'Name': ['Zayd', 'Darun', 'Iyaaa', 'Lokesh'],
        'Employee_ID': ['1001', '1002', '1003', '1004'],
        'Department': ['AI&DS', 'AI&DS', 'AI&DS', 'AI&DS'],
        'Test_Time': [datetime.now().strftime('%H:%M:%S')] * 4
    }
    
    df = pd.DataFrame(test_data)
    
    # Test Excel creation with openpyxl engine
    test_file = "test_excel_output.xlsx"
    with pd.ExcelWriter(test_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Test_Sheet', index=False)
    
    print(f"✅ Excel file created successfully: {test_file}")
    print("🎉 Excel functionality is working properly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Excel creation error: {e}")