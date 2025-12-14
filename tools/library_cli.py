import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Dict
import questionary

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"

def parse_metadata(file_path: Path) -> Dict:
    """Same metadata parser as before"""
    try:
        with open(file_path, 'r') as f:
            nb = json.load(f)
        if not nb['cells']: return {}
        first_source = nb['cells'][0]['source']
        if isinstance(first_source, list): first_source = "".join(first_source)
        match = re.search(r'# ---\n(.*?)# ---', first_source, re.DOTALL)
        if not match: return {}
        metadata_str = match.group(1)
        metadata = {}
        for line in metadata_str.strip().split('\n'):
            line = line.strip().lstrip('#').strip()
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip().lower()
                val = val.strip()
                if val.startswith('[') and val.endswith(']'):
                    val = [x.strip() for x in val[1:-1].split(',')]
                metadata[key] = val
        metadata['filename'] = file_path.name
        metadata['path'] = str(file_path)
        return metadata
    except Exception as e:
        return {'filename': file_path.name, 'error': str(e)}

def get_notebooks():
    notebooks = []
    for f in NOTEBOOKS_DIR.glob("*.ipynb"):
        meta = parse_metadata(f)
        if meta and 'title' in meta:
            notebooks.append(meta)
    return sorted(notebooks, key=lambda x: x.get('title', ''))

def get_all_tags(notebooks):
    tags = set()
    for nb in notebooks:
        for t in nb.get('tags', []):
            tags.add(t)
    return sorted(list(tags))

def interactive_tui():
    """Main Interactive Function"""
    print("\n📚 Market-Mind Knowledge Library\n")
    
    notebooks = get_notebooks()
    if not notebooks:
        print("No notebooks found.")
        return

    # Main Menu
    main_action = questionary.select(
        "What would you like to do?",
        choices=["Browse All Notebooks", "Filter by Tag", "Exit"],
        qmark="🏠"
    ).ask()
    
    if main_action == "Exit":
        return

    filtered_notebooks = notebooks
    
    if main_action == "Filter by Tag":
        all_tags = get_all_tags(notebooks)
        if not all_tags:
            print("No tags found.")
            interactive_tui()
            return
            
        selected_tag = questionary.select(
            "Select a Tag to filter by:",
            choices=all_tags + ["< Back"],
            qmark="🏷️"
        ).ask()
        
        if selected_tag == "< Back":
            interactive_tui()
            return
            
        filtered_notebooks = [nb for nb in notebooks if selected_tag in nb.get('tags', [])]
        print(f"\nShowing {len(filtered_notebooks)} notebooks tagged '{selected_tag}'")

    # Notebook Selection Loop
    while True:
        choices = []
        for nb in filtered_notebooks:
            title = nb.get('title', 'Untitled')
            tags = ", ".join(nb.get('tags', []))
            diff = nb.get('difficulty', '')
            
            display_text = f"{title} [{diff}] ({tags})"
            choices.append(questionary.Choice(display_text, value=nb))
            
        choices.append(questionary.Choice("< Back to Main Menu", value="BACK"))

        selected_nb = questionary.select(
            "Select a module to explore:",
            choices=choices,
            use_indicator=True,
            qmark="🔭"
        ).ask()

        if selected_nb == "BACK":
            interactive_tui()
            return

        if selected_nb:
            action = questionary.select(
                f"You selected: {selected_nb['title']}. What now?",
                choices=["Open in Jupyter", "View Metadata", "Cancel"],
                qmark="🚀"
            ).ask()

            if action == "Open in Jupyter":
                print(f"Launching {selected_nb['filename']}...")
                try:
                    subprocess.run(["jupyter", "notebook", selected_nb['path']], check=True)
                    return # Exit after launch? Or loop? Let's exit to let user work.
                except Exception as e:
                    print(f"Error launching jupyter: {e}")
            
            elif action == "View Metadata":
                print(json.dumps(selected_nb, indent=2))
                input("Press Enter to continue...")

if __name__ == "__main__":
    interactive_tui()
