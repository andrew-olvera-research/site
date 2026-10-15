"""Dependency-free portfolio regression checks. Run from any working directory."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import json
ROOT = Path(__file__).resolve().parents[1]
PAGES = ['index.html', 'starscream/index.html', 'microrts/index.html', 'token-level-rewards/index.html', 'flowdreamer-microrts/index.html']
class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.ids=[]; self.links=[]; self.h1=0; self.errors=[]
        self.feed(text)
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if 'id' in a: self.ids.append(a['id'])
        if tag=='h1': self.h1+=1
        for key in ['href','src','poster']:
            if a.get(key): self.links.append(a[key])
        if tag=='img' and not a.get('alt'): self.errors.append('image missing alt')
        if tag=='video' and ('autoplay' in a or 'controls' not in a): self.errors.append('video must have user controls without autoplay')
pages={p:Page((ROOT/p).read_text(encoding='utf-8')) for p in PAGES}
errors=[]; checked=0
for path,page in pages.items():
    errors += [f'{path}: {x}' for x in page.errors]
    if len(page.ids)!=len(set(page.ids)): errors.append(f'{path}: duplicate IDs')
    if path!='flowdreamer-microrts/index.html' and page.h1!=1: errors.append(f'{path}: expected one h1')
    for link in page.links:
        u=urlsplit(link)
        if u.scheme or u.netloc: continue
        target=((ROOT/path).parent/unquote(u.path)).resolve() if u.path else ROOT/path
        if target.is_dir(): target=target/'index.html'
        if not target.exists(): errors.append(f'{path}: missing {link}')
        elif u.fragment and target.suffix=='.html':
            target_page=Page(target.read_text(encoding='utf-8'))
            if unquote(u.fragment) not in target_page.ids: errors.append(f'{path}: missing anchor {link}')
        checked+=1
    if path!='flowdreamer-microrts/index.html':
        text=(ROOT/path).read_text(encoding='utf-8')
        for phrase in ['Preprint forthcoming','Paper (coming soon)','href="../manippo','href="manippo']:
            if phrase in text: errors.append(f'{path}: retired claim/link {phrase}')
star=pages['starscream/index.html']
for hook in ['fig-suites','fig-ladder','fig-outcomes','fig-families','fig-courses','fig-learning','fig-quality','fig-laps','fig-ratio','fig-bytype','fig-speed','fig-tails','clips','rtable']:
    if hook not in star.ids: errors.append(f'missing visual hook {hook}')
assert 'manippo' in (ROOT/'_config.yml').read_text()
assert not (ROOT/'.nojekyll').exists(), 'Jekyll exclusion requires Jekyll deployment'
data=json.loads((ROOT/'starscream/assets/data/development-comparison.json').read_text())
assert data['training_seeds_per_setup']==3 and data['courses']==100
assert data['results']['Transformer']['mean']==66.72
if errors: raise SystemExit('\n'.join(errors))
print(f'PASS: {len(pages)} routes; {checked} local links/assets/anchors; chart hooks, image alt text, video controls, and editorial guardrails.')
