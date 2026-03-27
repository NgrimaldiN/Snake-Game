"""
Probe the Google Snake game internals — Path A then Path B.

Path A: Intercept game JS source via page.route(), save it, inject a hook.
Path B: If A fails, hook the canvas drawing API to capture exact draw calls.
"""
import time, sys, os, json, re
from playwright.sync_api import sync_playwright

GAME_URL = "https://googlesnakemods.com/v/current/"
SAVE_DIR = os.path.dirname(__file__)

game_sources = {}

def intercept_js(route):
    """Intercept JS requests, save large scripts (likely the game source)."""
    response = route.fetch()
    body = response.text()
    url = route.request.url

    if len(body) > 5000:
        fname = re.sub(r'[^a-zA-Z0-9_.]', '_', url.split('/')[-1] or 'script') + '.js'
        fpath = os.path.join(SAVE_DIR, '_intercepted_' + fname)
        with open(fpath, 'w') as f:
            f.write(body)
        game_sources[url] = fpath
        print(f"  Intercepted {len(body):>8} bytes  {url[:80]}")
        print(f"    → saved to {fpath}")

    route.fulfill(response=response)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # Intercept all JS files
        page.route("**/*.js", intercept_js)
        page.route("**/*.js?*", intercept_js)

        print("Navigating to game (intercepting JS files)...")
        page.goto(GAME_URL, wait_until="networkidle")

        print(f"\nIntercepted {len(game_sources)} JS file(s).")

        if not game_sources:
            print("No JS files intercepted. The game may load JS differently.")
            print("Trying to capture inline scripts...")
            scripts = page.evaluate("""
            () => {
                const scripts = document.querySelectorAll('script');
                const info = [];
                for (const s of scripts) {
                    info.push({
                        src: s.src || '(inline)',
                        len: (s.textContent || '').length,
                        preview: (s.textContent || '').substring(0, 200)
                    });
                }
                return info;
            }
            """)
            print(json.dumps(scripts, indent=2))

        # Now let the user set up the game
        print("\n" + "=" * 60)
        print("  Configure the game:")
        print("    1. Settings -> 5 apples, wall, small map")
        print("    2. Press Play, then start moving the snake")
        print("=" * 60)
        try:
            input("\nPress Enter when the game is running... ")
        except EOFError:
            print("Waiting 30s...")
            time.sleep(30)

        # Analyze the largest intercepted JS file (most likely the game)
        if game_sources:
            largest = max(game_sources.items(), key=lambda kv: os.path.getsize(kv[1]))
            print(f"\nAnalyzing largest source: {largest[0]}")
            with open(largest[1]) as f:
                src = f.read()
            print(f"  Size: {len(src)} chars")

            # Search for game-state patterns
            patterns = {
                'board/grid dims': r'(?:width|height|grid|board)\s*[=:]\s*\d+',
                'snake array': r'(?:snake|body|train|worm)\s*[=:]\s*\[',
                'apple/food': r'(?:apple|food|fruit|target|passenger)',
                'direction constants': r'(?:ArrowUp|ArrowDown|ArrowLeft|ArrowRight)',
                'keydown handler': r'addEventListener\s*\(\s*["\']keydown',
                'game over/dead': r'(?:gameOver|game_over|dead|alive|active)',
                'score': r'(?:score|points)\s*[=+]',
                'this.width/height': r'this\.\w+\s*=\s*(?:this\.)?\w+\s*=\s*(?:10|20|9)',
                'coordinate objects': r'\{[^}]*(?:x|col)\s*:\s*\w+[^}]*(?:y|row)\s*:\s*\w+',
                'array push (body)': r'\.push\(\s*(?:new\s+)?\w+\(',
            }

            print("\n  Pattern search results:")
            for name, pat in patterns.items():
                matches = re.findall(pat, src, re.IGNORECASE)
                if matches:
                    print(f"    {name}: {len(matches)} matches")
                    for m in matches[:5]:
                        print(f"      → {m[:120]}")
                else:
                    print(f"    {name}: NO matches")

            # Look for the game constructor — search for functions that set up board dimensions
            print("\n  Searching for game constructor patterns...")
            # Pattern: something.width = something.height = NUMBER
            dim_matches = list(re.finditer(r'(\w+)\.(\w+)\s*=\s*\1\.(\w+)\s*=\s*(\d+)', src))
            for m in dim_matches[:10]:
                ctx = src[max(0, m.start()-50):m.end()+50]
                print(f"    dim: {m.group()} | context: ...{ctx}...")

            # Pattern: this.X = this.Y = NUMBER (board init in constructor)
            this_matches = list(re.finditer(r'this\.(\w+)\s*=\s*this\.(\w+)\s*=\s*(\d+)', src))
            for m in this_matches[:10]:
                ctx = src[max(0, m.start()-80):m.end()+80]
                print(f"    this-dim: {m.group()} | context: ...{ctx}...")

            # Find the keydown handler to locate the game object
            key_matches = list(re.finditer(r'addEventListener\s*\(\s*["\']keydown["\']', src))
            for m in key_matches[:5]:
                ctx = src[max(0, m.start()-200):m.end()+300]
                print(f"\n  keydown handler context:\n    ...{ctx[:500]}...")

            # Find switch cases for arrow keys (direction handler)
            arrow_matches = list(re.finditer(r'case\s*["\']Arrow\w+["\']', src))
            if arrow_matches:
                start = max(0, arrow_matches[0].start() - 200)
                end = min(len(src), arrow_matches[-1].end() + 200)
                ctx = src[start:end]
                print(f"\n  Arrow key switch block:\n    ...{ctx[:800]}...")

        # Also try deep probe of _s
        print("\n--- Deep probe of window._s ---")
        result = page.evaluate("""
        () => {
            if (!window._s) return {error: '_s not found'};
            const out = {};
            for (const k of Object.keys(window._s)) {
                const v = window._s[k];
                const t = typeof v;
                if (t === 'function') {
                    out[k] = `[fn(${v.length} args) name=${v.name||'?'}]`;
                    // Check if it's a constructor with useful prototype
                    try {
                        const proto = v.prototype;
                        if (proto && Object.keys(proto).length > 0) {
                            out[k + '_proto'] = Object.keys(proto).slice(0, 15);
                        }
                    } catch(e) {}
                } else if (t === 'object' && v !== null) {
                    try {
                        const keys = Object.keys(v);
                        out[k] = {type: 'object', keys: keys.slice(0, 20), len: keys.length};
                    } catch(e) {
                        out[k] = '[object, inaccessible]';
                    }
                } else {
                    out[k] = v;
                }
            }
            return out;
        }
        """)
        print(json.dumps(result, indent=2, default=str))

        print("\nDone. Browser stays open.")
        print("Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        browser.close()

if __name__ == "__main__":
    main()
