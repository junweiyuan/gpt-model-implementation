import os
from pathlib import Path
from mobile_gui_agent import MobileGUIAgent
from mobile_gui_agent.vlm import ClaudeVLM, GeminiVLM, OpenAIVLM


def example_claude_with_reasoning():
    print("Example 1: Claude 3.7 Sonnet with Reasoning")
    print("=" * 60)
    
    agent = MobileGUIAgent(
        vlm=ClaudeVLM(
            model="claude-3-7-sonnet-20250219",
            use_reasoning=True,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        ),
        prompting_strategy="direct",
        verbose=True
    )
    
    result = agent.execute_step(
        screenshot="screenshot.png",
        task_instruction="Set my DM Spam filter to 'Do not filter direct messages' on Discord app"
    )
    
    print(f"\nResult: {result['action'].to_dict()}")
    if result['reasoning']:
        print(f"Reasoning: {result['reasoning'][:200]}...")


def example_gemini_with_som():
    print("\n\nExample 2: Gemini 2.0 Flash with Set-of-Mark Prompting")
    print("=" * 60)
    
    agent = MobileGUIAgent(
        vlm=GeminiVLM(
            model="gemini-2.0-flash-thinking-exp",
            use_reasoning=True,
            api_key=os.getenv("GOOGLE_API_KEY")
        ),
        prompting_strategy="set_of_mark",
        verbose=True
    )
    
    result = agent.execute_step(
        screenshot="screenshot.png",
        task_instruction="Click on the search icon"
    )
    
    print(f"\nResult: {result['action'].to_dict()}")


def example_gpt4_multi_step():
    print("\n\nExample 3: GPT-4o Multi-Step Task")
    print("=" * 60)
    
    agent = MobileGUIAgent(
        vlm=OpenAIVLM(
            model="gpt-4o",
            use_reasoning=False,
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        prompting_strategy="direct",
        verbose=True
    )
    
    result = agent.execute_task(
        screenshot="screenshot.png",
        task_instruction="Open the settings and enable dark mode",
        max_steps=5
    )
    
    print(f"\nFinal Result: {result['action'].to_dict()}")
    print(f"Action History: {agent.get_action_history()}")


def example_with_accessibility_tree():
    print("\n\nExample 4: Using Accessibility Tree")
    print("=" * 60)
    
    agent = MobileGUIAgent(
        vlm=ClaudeVLM(
            model="claude-3-7-sonnet-20250219",
            use_reasoning=True
        ),
        prompting_strategy="accessibility_tree",
        verbose=True
    )
    
    accessibility_tree = {
        "class": "FrameLayout",
        "bounds": [0, 0, 1080, 2400],
        "children": [
            {
                "class": "TextView",
                "text": "Settings",
                "bounds": [40, 100, 200, 150],
                "clickable": True
            },
            {
                "class": "Button",
                "text": "Dark Mode",
                "bounds": [40, 200, 300, 250],
                "clickable": True
            }
        ]
    }
    
    result = agent.execute_step(
        screenshot="screenshot.png",
        task_instruction="Enable dark mode",
        accessibility_tree=accessibility_tree
    )
    
    print(f"\nResult: {result['action'].to_dict()}")


if __name__ == "__main__":
    print("Mobile GUI Agent Examples")
    print("=" * 60)
    print("\nNote: Make sure you have:")
    print("1. Set up API keys in environment variables")
    print("2. A screenshot.png file in the current directory")
    print("3. Installed all dependencies: pip install -r requirements.txt")
    print("\n" + "=" * 60)
    
    screenshot_path = Path("screenshot.png")
    if not screenshot_path.exists():
        print("\nWarning: screenshot.png not found. Creating a dummy image...")
        from PIL import Image
        img = Image.new('RGB', (1080, 2400), color='white')
        img.save('screenshot.png')
    
    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            example_claude_with_reasoning()
        else:
            print("\nSkipping Claude example (ANTHROPIC_API_KEY not set)")
        
        if os.getenv("GOOGLE_API_KEY"):
            example_gemini_with_som()
        else:
            print("\nSkipping Gemini example (GOOGLE_API_KEY not set)")
        
        if os.getenv("OPENAI_API_KEY"):
            example_gpt4_multi_step()
        else:
            print("\nSkipping OpenAI example (OPENAI_API_KEY not set)")
        
        if os.getenv("ANTHROPIC_API_KEY"):
            example_with_accessibility_tree()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have set up your API keys and installed dependencies")
