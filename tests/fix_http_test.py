# Simple script to add the missing initialize_pipeline method
with open(r'd:\Antigravity\06_ID_Legal\tests\integration\test_api_http.py', 'r', encoding='utf-8') as f:
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
    lines.insert(insert_at, new_method)
    
    with open(r'd:\Antigravity\06_ID_Legal\tests\integration\test_api_http.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Method successfully added!")
else:
    print("Could not find insertion point")
