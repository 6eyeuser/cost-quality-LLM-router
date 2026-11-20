"""List the models your API key can use (run this before picking a cheap/strong pair)."""
from openai import OpenAI

from router.config import CHEAP, JUDGE_MODEL, STRONG, UPSTREAM_BASE_URL, get_api_key

client = OpenAI(api_key=get_api_key(), base_url=UPSTREAM_BASE_URL)
available = sorted(m.id for m in client.models.list().data)
print("Models available on your API key:")
for m in available:
    print(f"  - {m}")
print()
for label, name in (("cheap", CHEAP.name), ("strong", STRONG.name), ("judge", JUDGE_MODEL)):
    print(f"{label:7s} {name:35s} {'OK' if name in available else 'NOT AVAILABLE - set the env var'}")
print(f"\nConfigured price ratio strong/cheap (input) = {STRONG.input_per_m / CHEAP.input_per_m:.1f}x, "
      f"(output) = {STRONG.output_per_m / CHEAP.output_per_m:.1f}x.  Verify prices at https://groq.com/pricing")
