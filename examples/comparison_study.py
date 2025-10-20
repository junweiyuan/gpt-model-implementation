import os
import json
from pathlib import Path
from typing import List, Dict, Any
from mobile_gui_agent import MobileGUIAgent
from mobile_gui_agent.vlm import ClaudeVLM, GeminiVLM, OpenAIVLM


class ComparisonStudy:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def run_comparison(
        self,
        screenshot_path: str,
        task_instruction: str,
        models_config: List[Dict[str, Any]]
    ):
        print(f"\n{'='*80}")
        print(f"Task: {task_instruction}")
        print(f"{'='*80}\n")
        
        for config in models_config:
            print(f"\nTesting: {config['name']}")
            print("-" * 80)
            
            try:
                vlm = self._create_vlm(config)
                agent = MobileGUIAgent(
                    vlm=vlm,
                    prompting_strategy=config.get('prompting_strategy', 'direct'),
                    verbose=True
                )
                
                result = agent.execute_step(
                    screenshot=screenshot_path,
                    task_instruction=task_instruction
                )
                
                self.results.append({
                    'model': config['name'],
                    'task': task_instruction,
                    'action': result['action'].to_dict(),
                    'reasoning': result['reasoning'],
                    'input_tokens': result['vlm_response'].input_tokens,
                    'output_tokens': result['vlm_response'].output_tokens,
                    'total_tokens': result['vlm_response'].get_total_tokens(),
                    'success': result['success']
                })
                
                print(f"\n✓ Completed: {result['action'].action_type}")
                print(f"  Tokens: {result['vlm_response'].input_tokens} in, {result['vlm_response'].output_tokens} out")
                
            except Exception as e:
                print(f"\n✗ Error: {str(e)}")
                self.results.append({
                    'model': config['name'],
                    'task': task_instruction,
                    'error': str(e)
                })
    
    def _create_vlm(self, config: Dict[str, Any]):
        vlm_type = config['type']
        model = config['model']
        use_reasoning = config.get('use_reasoning', False)
        
        if vlm_type == 'claude':
            return ClaudeVLM(model=model, use_reasoning=use_reasoning)
        elif vlm_type == 'gemini':
            return GeminiVLM(model=model, use_reasoning=use_reasoning)
        elif vlm_type == 'openai':
            return OpenAIVLM(model=model, use_reasoning=use_reasoning)
        else:
            raise ValueError(f"Unknown VLM type: {vlm_type}")
    
    def save_results(self, filename: str = "comparison_results.json"):
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n\nResults saved to: {output_path}")
    
    def print_summary(self):
        print("\n\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        for result in self.results:
            print(f"\nModel: {result['model']}")
            if 'error' in result:
                print(f"  Status: ✗ Error - {result['error']}")
            else:
                print(f"  Action: {result['action']['action_type']}")
                print(f"  Tokens: {result['total_tokens']} total ({result['output_tokens']} output)")
                if result.get('reasoning'):
                    print(f"  Reasoning: Yes ({len(result['reasoning'])} chars)")
                else:
                    print(f"  Reasoning: No")


def main():
    study = ComparisonStudy()
    
    models_config = [
        {
            'name': 'Claude 3.7 Sonnet (No Reasoning)',
            'type': 'claude',
            'model': 'claude-3-7-sonnet-20250219',
            'use_reasoning': False,
            'prompting_strategy': 'direct'
        },
        {
            'name': 'Claude 3.7 Sonnet (With Reasoning)',
            'type': 'claude',
            'model': 'claude-3-7-sonnet-20250219',
            'use_reasoning': True,
            'prompting_strategy': 'direct'
        },
        {
            'name': 'Gemini 2.0 Flash',
            'type': 'gemini',
            'model': 'gemini-2.0-flash-exp',
            'use_reasoning': False,
            'prompting_strategy': 'direct'
        },
        {
            'name': 'Gemini 2.0 Flash Thinking',
            'type': 'gemini',
            'model': 'gemini-2.0-flash-thinking-exp',
            'use_reasoning': True,
            'prompting_strategy': 'direct'
        },
        {
            'name': 'GPT-4o',
            'type': 'openai',
            'model': 'gpt-4o',
            'use_reasoning': False,
            'prompting_strategy': 'direct'
        }
    ]
    
    screenshot_path = "screenshot.png"
    if not Path(screenshot_path).exists():
        print("Creating dummy screenshot...")
        from PIL import Image
        img = Image.new('RGB', (1080, 2400), color='white')
        img.save(screenshot_path)
    
    tasks = [
        "Click on the search icon",
        "Set my DM Spam filter to 'Do not filter direct messages' on Discord app",
        "Open settings and enable dark mode"
    ]
    
    for task in tasks:
        study.run_comparison(screenshot_path, task, models_config)
    
    study.print_summary()
    study.save_results()


if __name__ == "__main__":
    print("Mobile GUI Agent Comparison Study")
    print("Based on: 'Does Chain-of-Thought Reasoning Help Mobile GUI Agent?'")
    print("\nThis script compares reasoning vs non-reasoning VLMs")
    print("Make sure you have set up your API keys:\n")
    print("  export ANTHROPIC_API_KEY=your_key")
    print("  export GOOGLE_API_KEY=your_key")
    print("  export OPENAI_API_KEY=your_key")
    print("\n" + "="*80)
    
    main()
