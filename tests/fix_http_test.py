# Simple script to add the missing initialize_pipeline method
import os
import sys

# Determine path relative to this script
# This script is in tests/, target is in tests/integration/
script_dir = os.path.dirname(os.path.abspath(__file__))
# Try integration subdirectory first (if script is in tests/)
target_file = os.path.join(script_dir, 'integration', 'test_api_http.py')

print(f"Current script dir: {script_dir}")
print(f"Looking for target at: {target_file}")

if not os.path.exists(target_file):
    # Try assuming script is in root (fallback)
    target_file = os.path.join(script_dir, 'tests', 'integration', 'test_api_http.py')
    print(f"Fallback target check: {target_file}")

if not os.path.exists(target_file):
    print("❌ Could not find test_api_http.py. Please ensure you are running this from the correct directory.")
    sys.exit(1)

print(f"✅ Found target file: {target_file}")

with open(target_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where to insert (after print_header method, line 50)
insert_at = -1
for i, line in enumerate(lines):
    if i > 40 and line.strip() == 'print()' and 'def print_header' in lines[i-7]:
        insert_at = i + 2  # Insert after the blank line following print_header
        break

if insert_at > 0:
    new_method = '''    def initialize_pipeline(self) -> bool:
        """Initialize pipeline for testing"""
        print("\\n" + "-" * 80)
        print("Initializing Pipeline")
        print("-" * 80)
        
        try:
            from pipeline import RAGPipeline
            
            config = {'final_top_k': 3, 'max_new_tokens': 1024}
            self.pipeline = RAGPipeline(config=config)
            success = self.pipeline.initialize()
            
            if success:
                print("✓ Pipeline initialized")
                self.logger.success("Pipeline ready for endpoint testing")
            else:
                print("✗ Pipeline initialization failed")
                
            return success
            
        except Exception as e:
            print(f"✗ Init error: {e}")
            self.logger.error(f"Init error: {e}")
            return False
    
'''
    # Check if method already exists to avoid duplication
    content = "".join(lines)
    if "def initialize_pipeline" in content:
        print("Method initialize_pipeline already exists. No changes made.")
    else:
        lines.insert(insert_at, new_method)
        
        with open(target_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"✅ Method successfully added to {target_file}!")
else:
    print("❌ Could not find insertion point in the file structure.")
