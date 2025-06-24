import os
import sys
from pathlib import Path

print("🔧 Debugging Python imports...")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {__file__}")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
print(f"Project root: {PROJECT_ROOT}")
print(f"Project root exists: {PROJECT_ROOT.exists()}")

# Check if agents directory exists
agents_dir = PROJECT_ROOT / "agents"
print(f"Agents directory: {agents_dir}")
print(f"Agents directory exists: {agents_dir.exists()}")

# Check files in agents directory
if agents_dir.exists():
    print("Files in agents directory:")
    for item in agents_dir.iterdir():
        print(f"  {item.name} ({'dir' if item.is_dir() else 'file'})")

# Add to Python path
sys.path.insert(0, str(PROJECT_ROOT))
print(f"Python path: {sys.path[:3]}...")

# Try importing
try:
    import agents
    print("✅ Successfully imported agents")
    
    try:
        import agents.base
        print("✅ Successfully imported agents.base")
        
        try:
            from agents.base.message_system import AgentMessage
            print("✅ Successfully imported AgentMessage")
        except ImportError as e:
            print(f"❌ Failed to import AgentMessage: {e}")
            
    except ImportError as e:
        print(f"❌ Failed to import agents.base: {e}")
        
except ImportError as e:
    print(f"❌ Failed to import agents: {e}")

# Check current directory contents
print(f"\nContents of current directory:")
for item in Path('.').iterdir():
    print(f"  {item.name}")