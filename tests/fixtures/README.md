# Generation form fixtures

`model_families/` contains four checkpoints (SDXL, FLUX.1-dev, WAN 2.2 I2V,
and MiniMax H3) and eight LoRAs with defaults and saved user overrides. Referenced local weights
are deliberately absent. These files support rendering and request previews,
not generation. They are separate from your actual model library.

From the project root, preview all four families without GPU dependencies:

```bash
IMAGE_GENERATOR_CATALOG_ROOT=tests/fixtures/model_families .venv/bin/python main.py
```

Open `http://127.0.0.1:5000/` and select **Configure generation**, or open
`/generate/fixture-sdxl`, `/generate/fixture-flux`, `/generate/fixture-wan`,
or `/generate/fixture-minimax`.
Enter a prompt and click **Preview request**. WAN and MiniMax additionally require selecting
a source image. Previews validate form settings only and do not save images,
files, overrides, or jobs. Reloading restores the fixture's saved settings.

Krea 2 RAW has a `/generate/fixture-krea2` form and two synthetic adapter entries.
Its weights are intentionally absent, like the other image family fixtures.

Qwen 2.1 has `/generate/fixture-qwen_2_1` and two synthetic adapters. Its form
supports true CFG, KV caching, an optional sigma list, and signed adapter
strengths. Preview works without weights; generation requires real assets.
